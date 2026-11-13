#!/usr/bin/env python3
"""Migration: old monolithic mdat -> standalone MDat datasets + linking container.

Old format: one folder with a single ``mdat.db`` catalog holding everything
(submdats, storage configs, strategies, views, shuffles) and all data under
``datasets/<name>`` + ``pretok/<strategy>/<name>``.

New format: each internal submdat becomes a standalone dataset folder (itself
a minimal container: catalog + datasets/ + pretok/ + functions/), and one
container links them all via relative paths (resolved against
``$MATFORMER_MDAT_ROOT`` first, the container folder second).

Usage:
    python -m matformer.data.mdat.migrate OLD [--dest DIR] [--mode copy|move]
        [--inplace | --container PATH] [--overwrite] [--quiet]

Phases:
  1. EXTRACT (always): for each submdat of OLD, build ``<dest>/<name>/`` as a
     standalone dataset (new ``mdat_container.db`` catalog with the dataset's
     rows: manifest, storage dbs, strategies applied to it, stats,
     precomputed lengths), then copy/move its ``datasets/<name>`` and every
     ``pretok/<strategy>/<name>`` folder (+ the custom splitter code).
     OLD is never modified by this phase (unless --mode move).
  2. CONTAINER:
     --inplace: rewrite OLD's catalog (backed up as mdat.db.pre_migration)
       into a linking container. Submdat rows keep their ds_map_ids, so the
       existing shuffle files stay valid byte-for-byte; mirrored config rows
       (storage dbs, strategies, stats) are cleared and refreshed
       automatically at the next load_dataset().
     --container PATH: build a NEW linking container at PATH, preserving
       ds_map ids (shuffle files copied), views and selections.

Idempotent per dataset: datasets whose standalone catalog already exists are
skipped unless --overwrite.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import uuid as uuid_lib

# Allow running as a plain script: python path/to/migrate.py
if __package__ in (None, ''):
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, repo_root)

from matformer.data.mdat.sql_database import DatabaseManager
from matformer.data.mdat.container import MDatContainer


def _fetch_all(conn, q, params=()):
    c = conn.cursor()
    c.execute(q, params)
    names = [d[0] for d in c.description]
    return [dict(zip(names, r)) for r in c.fetchall()]


def _old_db_path(old_path: str) -> str:
    candidate = os.path.join(old_path, 'mdat.db')
    if not os.path.exists(candidate):
        candidate = os.path.join(old_path, MDatContainer.DB_FILENAME)
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"No mdat catalog found at {old_path}")
    return candidate


def _transfer(src, dst, mode, mkdir_parent=False):
    if not os.path.exists(src):
        if mkdir_parent:
            os.makedirs(dst, exist_ok=True)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == 'move':
        shutil.move(src, dst)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def extract_dataset(old_path, name, dest_root, mode='copy', overwrite=False, quiet=False):
    """Create the standalone dataset folder for one internal submdat of the
    old catalog. Returns (dataset_path, uuid)."""
    dataset_path = os.path.join(dest_root, name)
    dataset_db = os.path.join(dataset_path, MDatContainer.DB_FILENAME)
    src_db_path = _old_db_path(old_path)

    if os.path.exists(dataset_db) and not overwrite:
        if not quiet:
            print(f"  [skip] {name}: standalone catalog already exists")
        ext = DatabaseManager(dataset_db)
        ext.connect()
        row = ext.get_submdat_row(name)
        return dataset_path, (row or {}).get('uuid')

    src = sqlite3.connect(src_db_path)
    try:
        sub_rows = _fetch_all(src, "SELECT * FROM submdat WHERE name=?", (name,))
        if not sub_rows:
            raise ValueError(f"submdat '{name}' not found in {src_db_path}")
        sub_row = sub_rows[0]
        old_id = sub_row['ds_map_id']
        manifest = _fetch_all(src, "SELECT * FROM manifest LIMIT 1")[0]
        storage = _fetch_all(src, """SELECT dr.*, sd.raw_data_bytes, sd.is_data, sd.is_meta, sd.is_extra
                                     FROM database_ref dr JOIN submdat_database sd ON dr.id = sd.database_id
                                     WHERE sd.submdat_id=?""", (old_id,))
        stats_rows = _fetch_all(src, "SELECT * FROM pretok_strategy_submdat WHERE submdat_id=?", (old_id,))
        strategy_payloads = []
        for stats in stats_rows:
            strat = _fetch_all(src, "SELECT * FROM pretok_strategy WHERE strategy_name=?", (stats['strategy_name'],))
            if not strat:
                continue
            sdbs = _fetch_all(src, """SELECT psd.*, dr.type, dr.compression_level, dr.map_size, dr.disk_size, dr.extra_data
                                      FROM pretok_strategy_db psd JOIN database_ref dr ON psd.db_id = dr.id
                                      WHERE psd.strategy_name=? AND psd.submdat_id=?""",
                              (stats['strategy_name'], old_id))
            precomps = _fetch_all(src, """SELECT * FROM pretok_strategy_submdat_precomputed_length
                                          WHERE strategy_name=? AND submdat_id=?""",
                                  (stats['strategy_name'], old_id))
            strategy_payloads.append((stats, strat[0], sdbs, precomps))
    finally:
        src.close()

    dataset_uuid = uuid_lib.uuid4().hex

    # ---- new standalone catalog
    os.makedirs(dataset_path, exist_ok=True)
    dst = DatabaseManager(dataset_db)
    conn = dst.connect()
    dst._initialize_db()
    dst.create_manifest(manifest['dsmap_bits'], json.loads(manifest['modalities']))
    new_id = conn.execute(
        "INSERT INTO submdat(name, data_type, document_number, modality, data_key, errors_counters,"
        " finalized_at, uuid, generation) VALUES(?,?,?,?,?,?,?,?,0)",
        (name, sub_row['data_type'], sub_row['document_number'], sub_row['modality'],
         sub_row['data_key'], sub_row['errors_counters'], sub_row['finalized_at'], dataset_uuid)).lastrowid
    conn.commit()
    # its own minimal container part: default view holding the dataset
    dst.add_view('default')
    raw_bytes = next((d['raw_data_bytes'] for d in storage if d['is_data']), 0) or 0
    dst.link_submdat_view(new_id, 'default', is_skipped=0, is_partial=0, is_preserved=1,
                          document_number=sub_row['document_number'] or 0, bytes_criteria=raw_bytes)

    for dbs in storage:
        new_db_id = dst.add_database(_type=dbs['type'], compression_level=dbs['compression_level'],
                                     map_size=dbs['map_size'], disk_size=dbs['disk_size'],
                                     extra_data=dbs['extra_data'])
        dst.link_submdat_database(new_id, new_db_id, raw_data_bytes=dbs['raw_data_bytes'],
                                  is_data=dbs['is_data'], is_meta=dbs['is_meta'], is_extra=dbs['is_extra'])

    for stats, strat, sdbs, precomps in strategy_payloads:
        strategy_name = stats['strategy_name']
        dst.upsert_strategy_row(strat)
        dst.upsert_strategy_submdat(strategy_name, new_id, stats)
        for sdb in sdbs:
            new_db_id = dst.add_database(_type=sdb['type'], compression_level=sdb['compression_level'],
                                         map_size=sdb['map_size'], disk_size=sdb['disk_size'],
                                         extra_data=sdb['extra_data'])
            dst.add_strategy_db(new_db_id, strategy_name, new_id, sdb['datatype'],
                                sdb['is_tokens'], sdb['is_chunks'], sdb['is_extra'], sdb['is_complete'])
        for precomp in precomps:
            dst.set_strategy_submdat_precomp(strategy_name, new_id, precomp['multiplier'],
                                             precomp['precomputed_length'])
    conn.close()

    # ---- data folders (raw data + pretokenized outputs)
    _transfer(os.path.join(old_path, MDatContainer.DATASETS_DIR, name),
              os.path.join(dataset_path, MDatContainer.DATASETS_DIR, name), mode, mkdir_parent=True)
    for stats, _strat, _sdbs, _precomps in strategy_payloads:
        src_folder = os.path.join(old_path, MDatContainer.PRETOK_DIR, stats['strategy_name'], name)
        if os.path.isdir(src_folder):
            _transfer(src_folder,
                      os.path.join(dataset_path, MDatContainer.PRETOK_DIR, stats['strategy_name'], name), mode)
    # custom splitter code travels with the dataset
    old_functions = os.path.join(old_path, MDatContainer.FUNCTIONS_DIR)
    if os.path.isdir(old_functions):
        dst_functions = os.path.join(dataset_path, MDatContainer.FUNCTIONS_DIR)
        os.makedirs(dst_functions, exist_ok=True)
        for fn in os.listdir(old_functions):
            if fn.endswith('.py'):
                shutil.copy2(os.path.join(old_functions, fn), os.path.join(dst_functions, fn))

    if not quiet:
        print(f"  [extract] {name}: {sub_row['document_number']} docs, "
              f"{len(strategy_payloads)} strategies -> {dataset_path}")
    return dataset_path, dataset_uuid


def _relative_link(dataset_path, container_root):
    """Prefer a $MATFORMER_MDAT_ROOT-relative representation (portable across
    machines), fall back to container-relative."""
    dataset_path = os.path.realpath(dataset_path)
    env_root = os.environ.get('MATFORMER_MDAT_ROOT')
    if env_root and os.path.isdir(env_root):
        env_root = os.path.realpath(env_root)
        if dataset_path.startswith(env_root + os.sep):
            return os.path.relpath(dataset_path, env_root)
    return os.path.relpath(dataset_path, os.path.realpath(container_root))


def build_linking_container(old_path, extracted, container_path=None, inplace=False, quiet=False):
    """Create (or rewrite, with inplace=True) the container that links the
    extracted standalone datasets. ds_map ids are preserved, so shuffle
    pointer files stay valid byte-for-byte."""
    src_db_path = _old_db_path(old_path)

    if inplace:
        container_root = old_path
        container_db_path = src_db_path
        backup = container_db_path + '.pre_migration'
        if not os.path.exists(backup):
            shutil.copy2(container_db_path, backup)
            if not quiet:
                print(f"Backed up catalog to {backup}")
        # raw sqlite3 connection: FKs are off, mirror rows can be cleared freely
        conn = sqlite3.connect(container_db_path)
        DatabaseManager(container_db_path).connect().close()  # ensure linking columns exist
        for table in ('submdat_database', 'pretok_strategy_db',
                      'pretok_strategy_submdat_precomputed_length',
                      'pretok_strategy_view_submdat_precomputed_length',
                      'pretok_strategy_view_distributed_length',
                      'pretok_strategy_submdat', 'pretok_strategy', 'database_ref'):
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
    else:
        if not container_path:
            raise ValueError("--container PATH is required when not using --inplace")
        if os.path.exists(os.path.join(container_path, MDatContainer.DB_FILENAME)):
            raise ValueError(f"A container already exists at {container_path}")
        container_root = container_path
        container_db_path = os.path.join(container_path, MDatContainer.DB_FILENAME)
        for d in (MDatContainer.DATASETS_DIR, MDatContainer.SHUFFLING_DIR, MDatContainer.VIEWS_DIR,
                  MDatContainer.PRETOK_DIR, MDatContainer.FUNCTIONS_DIR):
            os.makedirs(os.path.join(container_path, d), exist_ok=True)
        dm = DatabaseManager(container_db_path)
        dm.connect()
        dm._initialize_db()
        old_manifest = _fetch_all(sqlite3.connect(src_db_path), "SELECT * FROM manifest LIMIT 1")[0]
        dm.create_manifest(old_manifest['dsmap_bits'], json.loads(old_manifest['modalities']))
        # copy submdat rows (same ids) and views/selections from the old catalog
        old = sqlite3.connect(src_db_path)
        for row in _fetch_all(old, "SELECT * FROM submdat ORDER BY ds_map_id"):
            conn = dm.connect()
            conn.execute(
                "INSERT INTO submdat(ds_map_id, name, data_type, document_number, modality, data_key,"
                " errors_counters, finalized_at) VALUES(?,?,?,?,?,?,?,?)",
                (row['ds_map_id'], row['name'], row['data_type'], row['document_number'],
                 row['modality'], row['data_key'], row['errors_counters'], row['finalized_at']))
        for view in _fetch_all(old, "SELECT * FROM view"):
            dm.connect().execute("INSERT OR REPLACE INTO view(view_name, shuffle_struct_format) VALUES(?,?)",
                                 (view['view_name'], view['shuffle_struct_format']))
        for vs in _fetch_all(old, "SELECT * FROM view_submdat"):
            dm.connect().execute(
                "INSERT OR REPLACE INTO view_submdat(view_name,submdat_id,is_skipped,is_preserved,is_partial,document_number,bytes_criteria) VALUES(?,?,?,?,?,?,?)",
                (vs['view_name'], vs['submdat_id'], vs['is_skipped'], vs['is_preserved'],
                 vs['is_partial'], vs['document_number'], vs['bytes_criteria']))
        has_dist = _fetch_all(old, "SELECT COUNT(*) AS n FROM pretok_strategy_view_distributed_length")[0]['n']
        old.close()
        if has_dist:
            print(f"  [note] {has_dist} distributed precompute rows are not migrated: rerun "
                  "precompute_chunks_distributed() after the first load of the new container")
        dm.connect().commit()
        conn = sqlite3.connect(container_db_path)

    # turn the submdat rows into links (ids preserved)
    rows = _fetch_all(conn, "SELECT ds_map_id, name FROM submdat ORDER BY ds_map_id")
    linked = 0
    for row in rows:
        name = row['name']
        if name not in extracted:
            if not quiet:
                print(f"  [note] '{name}' missing from extraction, left unlinked")
            continue
        dataset_path, dataset_uuid = extracted[name]
        stored_rel = _relative_link(dataset_path, container_root)
        conn.execute("UPDATE submdat SET source_rel_path=?, uuid=?, generation=0 WHERE ds_map_id=?",
                     (stored_rel, dataset_uuid, row['ds_map_id']))
        linked += 1
    conn.commit()

    # views' shuffle pointer files: ids unchanged => byte-for-byte copies
    shuffled_views = []
    if not inplace:
        for view in _fetch_all(conn, "SELECT view_name, shuffle_struct_format FROM view"):
            if view['shuffle_struct_format']:
                src_file = os.path.join(old_path, MDatContainer.SHUFFLING_DIR, f"{view['view_name']}.mdat")
                dst_file = os.path.join(container_root, MDatContainer.SHUFFLING_DIR, f"{view['view_name']}.mdat")
                if os.path.exists(src_file):
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                shuffled_views.append(view['view_name'])
    else:
        shuffled_views = [v['view_name'] for v in
                          _fetch_all(conn, "SELECT view_name FROM view WHERE shuffle_struct_format IS NOT NULL")]
        # mirrored rows were cleared: force a re-sync of strategy data on next load
    conn.close()

    if not quiet:
        print(f"Container ready at {container_root}: {linked} linked datasets, "
              f"shuffled views: {shuffled_views or 'none'}")
        print("On next load the container refreshes the mirrors of every linked dataset.")
    return container_root


def migrate(old_path, dest=None, mode='copy', container=None, inplace=False, overwrite=False, quiet=False):
    old_path = os.path.abspath(old_path.rstrip('/'))
    src_db_path = _old_db_path(old_path)
    dest = os.path.abspath(dest or (old_path + '.mdats'))
    os.makedirs(dest, exist_ok=True)

    conn = sqlite3.connect(src_db_path)
    names = [r[0] for r in conn.execute("SELECT name FROM submdat ORDER BY ds_map_id").fetchall()]
    conn.close()
    print(f"Migrating {len(names)} datasets from {old_path} -> {dest} (mode={mode})")

    extracted = {}
    for name in names:
        extracted[name] = extract_dataset(old_path, name, dest, mode=mode, overwrite=overwrite, quiet=quiet)

    if inplace:
        build_linking_container(old_path, extracted, inplace=True, quiet=quiet)
    elif container:
        build_linking_container(old_path, extracted, container_path=os.path.abspath(container), quiet=quiet)
    else:
        print("Extraction done. No container phase requested (use --inplace or --container PATH).")
    return extracted


def main(argv=None):
    parser = argparse.ArgumentParser(description="Migrate an old monolithic mdat to standalone datasets + linking container")
    parser.add_argument('old_mdat', help="Path of the old monolithic mdat folder")
    parser.add_argument('--dest', default=None, help="Root for standalone dataset folders (default: <old>.mdats)")
    parser.add_argument('--mode', choices=['copy', 'move'], default='copy', help="Copy or move the data folders")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--inplace', action='store_true',
                       help="Rewrite the old catalog into a linking container (a .pre_migration backup is taken)")
    group.add_argument('--container', default=None, help="Path of the new linking container to build")
    parser.add_argument('--overwrite', action='store_true', help="Re-extract datasets whose catalog already exists")
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    migrate(args.old_mdat, dest=args.dest, mode=args.mode, container=args.container,
            inplace=args.inplace, overwrite=args.overwrite, quiet=args.quiet)


if __name__ == '__main__':
    main()
