#!/usr/bin/env python3
"""MDat command interpreter: a cmd2 shell + a programmatic facade over the
mdat system (containers, standalone datasets, views, pretokenization...).

The real work happens in :class:`MDatCommands`, which is UI-agnostic: every
method takes explicit arguments and returns a human-readable string (or data).
:class:`MDatShell` is a thin cmd2 layer over it, and future GUIs / MCP servers
/ scripts should target MDatCommands directly, not the shell.

Run interactively:
    python -m matformer.data.mdat.tui [PATH]        # PATH = container or standalone dataset
Or use it programmatically:
    from matformer.data.mdat.tui import MDatCommands
    cmds = MDatCommands()
    cmds.open('/data/mycontainer')
    print(cmds.ls_datasets())

Typical "from scratch" session:
    create /data/exp1.container            # or: new /data/mdats/oscar_it
    link /data/mdats/oscar_it --alias oscar
    ingest oscar jsonl /data/oscar_it.jsonl --key text
    strat-add minerva512 /configs/minerva_512.json
    pretok oscar --strategy minerva512
    view-add full --selection "oscar=preserved" --shuffle
    use full
    shuffle full
    peek --n 3
    precompute-dist 8
"""
from __future__ import annotations

import argparse
import glob as glob_module
import json
import os
import shlex
from typing import Any, Dict, List, Optional

import cmd2

from .container import MDatContainer
from .mdat import MDat
from .mdat_common import MDatNotFound, SubmDatNotFound


class _ShlexArgumentParser(argparse.ArgumentParser):
    """argparse that accepts a raw command string (cmd2 hands lines as str)
    and raises instead of calling sys.exit on bad usage."""

    def parse_args(self, args=None, namespace=None):  # type: ignore[override]
        if isinstance(args, str):
            args = shlex.split(args)
        return super().parse_args(args, namespace)

    def error(self, message):  # type: ignore[override]
        raise ValueError(f"usage error: {message}")


def _fmt_bytes(n) -> str:
    if not isinstance(n, (int, float)):
        return str(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


class MDatCommands:
    """UI-agnostic facade: one open container at a time, every operation
    returns a printable summary string."""

    def __init__(self) -> None:
        self.ds: Optional[MDatContainer] = None
        self.path: Optional[str] = None
        self.readonly: bool = False

    # ------------------------------------------------------------- session
    def _require(self) -> MDatContainer:
        if self.ds is None:
            raise ValueError("No container open. Use: open PATH")
        return self.ds

    def open(self, path: str, readonly: bool = False, view: str = None,
             shuffle: bool = False, distributed: bool = False, create: bool = False) -> str:
        """Open a container OR a standalone dataset folder (its own mini-container)."""
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path) and create:
            self.ds = MDatContainer.create_new(path)
        elif not os.path.exists(path):
            raise MDatNotFound(f"No such path: {path} (use `create {path}` to make a container)")
        else:
            self.ds = MDatContainer.load_dataset(path, readonly=readonly, ds_view=view,
                                                 shuffle=shuffle, distributed=distributed)
        self.path = path
        self.readonly = readonly
        names = self.ds.list_submdat()
        kind = "standalone dataset (used through its own mini-container)" if len(names) == 1 else "container"
        return (f"Opened {kind}: {path}\n"
                f"  datasets: {names or '(none)'}\n"
                f"  views: {self.ds.list_views()} (active: {self.ds.current_view})\n"
                f"  documents in active view: {len(self.ds)}")

    def create(self, path: str, dsmap_bits: int = 8, modalities: List[str] = None) -> str:
        path = os.path.abspath(os.path.expanduser(path))
        ds = MDatContainer.create_new(path, dsmap_bits=dsmap_bits,
                                      modalities=modalities or ['text'])
        self.ds, self.path, self.readonly = ds, path, False
        return f"Created container at {path}"

    def new_dataset(self, path: str, compression: int = 0) -> str:
        """Create a standalone dataset folder (self-contained, linkable anywhere)."""
        path = os.path.abspath(os.path.expanduser(path))
        sbm = MDat.create_standalone(path, compression_levels={'data': compression, 'meta': compression})
        return (f"Created standalone dataset '{sbm.submdat_name}' at {path} "
                f"(use `link {path}` from any container)")

    def close(self) -> str:
        self.ds = None
        self.path = None
        return "Closed"

    def status(self) -> str:
        ds = self._require()
        out = [f"path: {self.path}", f"readonly: {self.readonly}",
               f"active view: {getattr(ds, 'current_view', None)} "
               f"(shuffled: {getattr(ds, '_active_view_shuffled', False)})",
               f"strategy: {ds.current_strategy.strategy_name if ds.current_strategy else None} "
               f"mult: {ds.chunk_multiplier}",
               f"modality: {getattr(ds, 'current_iteration_modality', None)}",
               f"documents: {len(ds)}  tokens: {ds.total_tokens}  chunks: {ds.total_chunks}",
               f"iteration count: {getattr(ds, '_iteration_count', 0)} "
               f"(document_index: {getattr(ds, 'document_index', 0)}, "
               f"chunk_step: {getattr(ds, 'current_chunk_step', 0)})"]
        return "\n".join(out)

    # ------------------------------------------------------------- datasets
    def _default_dataset(self, name: Optional[str]) -> str:
        ds = self._require()
        names = ds.list_submdat()
        if name:
            if name not in names:
                raise SubmDatNotFound(f"'{name}' not in this container ({names})")
            return name
        if len(names) == 1:
            return names[0]
        raise ValueError(f"Datasets available: {names} — specify one")

    def ls_datasets(self) -> str:
        ds = self._require()
        lines = []
        for row in ds.db.list_submdats():
            kind = f"LINK -> {row['source_rel_path']}" if row.get('source_rel_path') else 'internal'
            strategies = [s for s in ds.list_strategies()
                          if ds.db.get_strategy_submdats(s, row['ds_map_id'])]
            info = ds.db.get_submdat_info(row['ds_map_id']) or {}
            lines.append(f"  {row['name']:<28} {row.get('document_number') or 0:>8} docs  "
                         f"{_fmt_bytes(info.get('raw_data_bytes')):>10}  [{kind}]  strategies: {strategies or '-'}")
        return "\n".join(lines) or "(no datasets)"

    def info_dataset(self, name: str = None) -> str:
        ds = self._require()
        name = self._default_dataset(name)
        sbm = ds._ensure_loaded(name)
        row = sbm.get_row()
        lines = [f"name: {name} ({'linked' if sbm.linked else 'internal'})",
                 f"documents: {len(sbm)}  generation: {sbm.generation}  uuid: {sbm.uuid}",
                 f"folder: {sbm.submdat_path}"]
        for strategy in ds.list_strategies():
            stats = ds.db.get_strategy_submdats(strategy, ds.db.get_dsmap('name', name))
            if stats:
                lines.append(f"  strategy {strategy}: tokens={stats.get('total_tokens')} "
                             f"chunks={stats.get('total_chunks')} finalized={stats.get('finalized_at')}")
        return "\n".join(lines)

    def link(self, path: str, alias: str = None) -> str:
        row = self._require().link_dataset(os.path.abspath(os.path.expanduser(path)), alias=alias)
        return (f"Linked dataset '{row['name']}' -> {row['source_rel_path']} "
                f"({row['document_number'] or 0} documents, uuid {str(row['uuid'])[:8]})")

    def unlink(self, alias: str) -> str:
        self._require().unlink_dataset(alias)
        return f"Unlinked '{alias}'. Run `repair` to rebuild views/shuffle files."

    def remove(self, name: str, delete_data: bool = False) -> str:
        self._require().remove_submdat(name, delete_data=delete_data)
        return f"Removed '{name}' from the container. Run `repair` to rebuild views/shuffle files."

    def add(self, name: str, compression: int = 0) -> str:
        """BACKWARD-COMPAT(internal): prefer new_dataset + link."""
        self._require().add_submdat(name, compression_levels={'data': compression, 'meta': compression})
        return f"Added internal dataset '{name}' (legacy layout; prefer: new + link)"

    # --------------------------------------------------------------- ingest
    def ingest(self, dataset: Optional[str], type: str, source: str, data_key: str = 'text') -> str:
        ds = self._require()
        if self.readonly:
            raise PermissionError("Container opened readonly: reopen without --ro to ingest")
        dataset = self._default_dataset(dataset)
        sbm = ds._ensure_loaded(dataset)
        res = sbm.convert_to_submdat(type, os.path.abspath(os.path.expanduser(source)),
                                     data_key=data_key, progress_bar=True)
        ds.set_view(ds.current_view)  # refresh totals from the catalog
        return f"ingest: {res}"

    def batch_import(self, folder: str, data_key: str = 'text', pattern: str = '*.jsonl',
                     recursive: bool = False, compression: int = 0) -> str:
        """One dataset per file (named after the file stem), ingested as jsonl."""
        ds = self._require()
        folder = os.path.abspath(os.path.expanduser(folder))
        files = (glob_module.glob(os.path.join(folder, '**', pattern), recursive=True)
                 if recursive else glob_module.glob(os.path.join(folder, pattern)))
        ok = 0
        for fp in sorted(files):
            name = os.path.splitext(os.path.basename(fp))[0]
            if name in ds.list_submdat():
                print(f"  [skip] dataset '{name}' already exists")
                continue
            sbm = ds.add_submdat(name, compression_levels={'data': compression, 'meta': compression})
            res = sbm.convert_to_submdat('jsonl', fp, data_key=data_key, progress_bar=True)
            ok += 1 if res else 0
        ds.set_view(ds.current_view)
        return f"Batch import: {ok}/{len(files)} files done"

    # ------------------------------------------------------------ strategies
    def strategies(self) -> str:
        return "\n".join(f"  {s}" for s in self._require().list_strategies()) or "(none)"

    def strat_add(self, name: str, json_path: str) -> str:
        with open(os.path.abspath(os.path.expanduser(json_path))) as f:
            self._require().register_strategy(name, json.load(f))
        return f"Strategy '{name}' registered"

    def strat_del(self, name: str, delete_files: bool = False) -> str:
        self._require().deregister_strategy(name, delete_files=delete_files)
        return (f"Strategy '{name}' deregistered"
                + (" (output folders deleted)" if delete_files else "")
                + "; linked datasets keep their own copy (reappears on next sync).")

    def set_strategy(self, name: str, seq_len: int = None) -> str:
        ds = self._require()
        ds.set_strategy(name, max_seq_len=seq_len)
        return (f"Strategy: {name}  multiplier: {ds.chunk_multiplier}  max_seq_len: {seq_len}  "
                f"tokens: {ds.total_tokens}  chunks: {ds.total_chunks}  "
                f"iterations (divided chunks): {ds.total_divided_chunks}")

    def pretok(self, dataset: Optional[str] = None, strategy: str = None, parallel: bool = True,
               processes: int = None, batch: int = 256, compression: int = 0) -> str:
        ds = self._require()
        if self.readonly:
            raise PermissionError("Container opened readonly: reopen without --ro to pretokenize")
        if strategy is None:
            raise ValueError("--strategy is required")
        dataset = self._default_dataset(dataset)
        sbm = ds._ensure_loaded(dataset)
        sbm.pretokenize_submdat(strategy, parallel=parallel, num_processes=processes,
                                batch_size=batch, compression_level=compression, progress_bar=True)
        return f"Pretokenized '{dataset}' with '{strategy}'"

    def pretok_all(self, strategy: str, parallel: bool = True, processes: int = None,
                   batch: int = 256, compression: int = 0) -> str:
        ds = self._require()
        done = []
        for name in ds.list_submdat():
            sbm = ds._ensure_loaded(name)
            sbm.pretokenize_submdat(strategy, parallel=parallel, num_processes=processes,
                                    batch_size=batch, compression_level=compression, progress_bar=True)
            done.append(name)
        return f"Pretokenized {len(done)} datasets with '{strategy}': {done}"

    # ----------------------------------------------------------------- views
    def views(self) -> str:
        ds = self._require()
        lines = []
        for v in ds.list_views():
            row = ds.db.get_view_row(v)
            totals = ds.db.get_view_totals(view_name=v)
            lines.append(f"  {v:<24} {totals['document_number']:>8} docs  "
                         f"{_fmt_bytes(totals['bytes_criteria']):>10}  "
                         f"{'shuffled' if row.get('shuffle_struct_format') else 'NOT shuffled'}"
                         + ('  (active)' if v == getattr(ds, 'current_view', None) else ''))
        return "\n".join(lines) or "(no views)"

    def view_template(self, output: str = 'view.json') -> str:
        ds = self._require()
        template = {"view_name": "example",
                    "selection": {n: "preserved" for n in ds.list_submdat()},
                    "shuffle": True}
        with open(output, 'w') as f:
            json.dump(template, f, indent=2)
        return (f"Template written to {output}. Selection values: 'preserved', 'skipped', "
                "N (documents), 0.N (proportion), '80M'/'2G' (bytes)")

    def view_add(self, name: str = None, selection: str = None, json_path: str = None,
                 shuffle: bool = True) -> str:
        ds = self._require()
        if json_path:
            with open(os.path.abspath(os.path.expanduser(json_path))) as f:
                view_dict = json.load(f)
        else:
            if not name or not selection:
                raise ValueError("provide --selection 'ds=value,...' or --json FILE")
            view_dict = {"view_name": name,
                         "selection": self._parse_selection(selection),
                         "shuffle": shuffle}
        view_dict.setdefault('view_name', name)
        name = view_dict.get('view_name')
        if name is None:
            raise ValueError("view name required (view-add NAME --selection ... or --json FILE)")
        ds.register_new_view(view_dict)
        return f"View '{name}' created. Use `views` to list."

    @staticmethod
    def _parse_selection(spec: str) -> Dict[str, Any]:
        out = {}
        for pair in spec.split(','):
            if '=' not in pair:
                raise ValueError(f"bad selection item '{pair}' (expected name=value)")
            k, v = pair.split('=', 1)
            v = v.strip()
            if v.lower() in ('preserved', 'skipped'):
                out[k.strip()] = v.lower()
            else:
                try:
                    out[k.strip()] = int(v)
                except ValueError:
                    try:
                        out[k.strip()] = float(v)
                    except ValueError:
                        out[k.strip()] = v  # byte spec like 2G / 500M
        return out

    def view_del(self, name: str) -> str:
        self._require().remove_view(name)
        return f"View '{name}' removed (shuffle file and precomputes included)"

    def head_view(self, name: str, n: int, source: str = None) -> str:
        ds = self._require()
        ds.create_head_view(n, name, source_view=source or ds.current_view)
        return f"Head view '{name}' created ({n} docs from '{source or ds.current_view}')"

    def range_view(self, name: str, start: int, end: int = None, source: str = None) -> str:
        ds = self._require()
        ds.create_range_view(name, source_view=source or ds.current_view, start=start, end=end)
        return f"Range view '{name}' created ([{start}:{end}] of '{source or ds.current_view}')"

    def use(self, view: str) -> str:
        ds = self._require()
        ds.set_view(view)
        return f"Active view: {view} ({len(ds)} documents, shuffled: {ds._active_view_shuffled})"

    def shuffle(self, view: str = None) -> str:
        ds = self._require()
        view = view or ds.current_view
        fmt = ds.shuffle(view)
        ds.set_view(view)
        return f"View '{view}' shuffled (pointer struct: {fmt}, {len(ds)} documents)"

    def repair(self) -> str:
        return f"Repair: {self._require().repair()}"

    def check(self) -> str:
        issues = self._require().check_linked_datasets(verbose=False)
        return "All linked datasets in sync" if not issues else json.dumps(issues, indent=2)

    # -------------------------------------------------------------- training
    def modality(self, name: str, meta: bool = False, raw: bool = False) -> str:
        ds = self._require()
        ds.set_iteration_modality(name, with_meta=meta, return_raw=raw)
        return f"Modality: {name} (len: {len(ds)})"

    def peek(self, dataset: str = None, key: int = None, n: int = 3, chunks: bool = False) -> str:
        """Print documents: a specific one (`--key`) or n random ones."""
        ds = self._require()
        name = self._default_dataset(dataset)
        sbm = ds._ensure_loaded(name)
        if chunks:
            strategy_name = ds.current_strategy.strategy_name if ds.current_strategy else None
            if strategy_name is None:
                raise ValueError("run `set-strategy NAME` first (peek --chunks needs it)")
            sbm._attach_strategy(strategy_name)
            sbm.set_default_wanted(from_dbs=['meta'], from_strategy=['chunked_tokens'])
        lines = []
        if key is not None:
            keys = [key]
        else:
            import random
            keys = random.sample(range(len(sbm)), min(n, len(sbm)))
        for k in keys:
            doc = sbm[k]
            data = doc.get('data')
            if isinstance(data, bytes):
                data = data[:200].decode('utf-8', errors='replace') + '…'
            elif isinstance(data, str):
                data = data[:200] + '…'
            entry = f"  [{name} #{k}]"
            if chunks:
                ct = doc.get('chunked_tokens', [])
                entry += f" {len(ct)} chunks; first chunk ({len(ct[0]) if ct else 0} tok): {ct[0][:12] if ct else '…'}"
            elif data is not None:
                entry += f" {data}"
            lines.append(entry)
        return "\n".join(lines)

    def tokens(self) -> str:
        ds = self._require()
        return (f"view: {ds.current_view}  documents: {ds.total_documents}  "
                f"tokens: {ds.total_tokens}  chunks: {ds.total_chunks}  "
                f"divided chunks (iterations @ mult {ds.chunk_multiplier}): {ds.total_divided_chunks}")

    def precompute_dist(self, num_workers: int) -> str:
        return str(self._require().precompute_chunks_distributed(num_workers=num_workers))

    def dist_len(self, num_devices: int) -> str:
        return f"Distributed length for {num_devices} devices: " \
               f"{self._require().get_distributed_length_before_training(num_devices)}"

    def export(self, view: str, output: str, limit: int = None, wanted: str = 'document') -> str:
        n = self._require().export_view(output, view=view, wanted=wanted, limiter=limit)
        return f"Exported view '{view}' -> {output} ({n} documents)"

    def state_save(self, output: str) -> str:
        with open(output, 'w') as f:
            json.dump(self._require().get_state_dict(), f, indent=1)
        return f"State saved to {output}"

    def state_load(self, input_file: str) -> str:
        with open(input_file) as f:
            self._require().restore_state_dict(json.load(f))
        return "State restored: next `iterate`/training continues from that document"

    def iterate(self, n: int = 5) -> str:
        ds = self._require()
        out = []
        try:
            for i, item in enumerate(ds):
                obj = item['object']
                size = len(obj) if isinstance(obj, (list, bytes)) else 'doc'
                out.append(f"  step {i}: modality={ds.current_iteration_modality} size={size}")
                if i + 1 >= n:
                    break
        finally:
            ds.stop_prefetch()  # don't let the prefetch thread fill its queue after we stop reading
        return "\n".join(out)


def _parser(*specs, **flags):
    """Build an argparse parser for one shell command.

    Positional specs: 'name', optionally colon-suffixed:
        'n?'        optional positional
        'n?:int'    optional + type
        'files*'    multiple
    Flag specs: opt=('dest', {kwargs}) or opt=('dest', 'store_true'|'int'|'list').
    """
    p = _ShlexArgumentParser(add_help=False)
    for spec in specs:
        parts = spec.split(':')
        name, kwargs = parts[0], {}
        while name and name[-1] in '?*':
            kwargs['nargs'] = name[-1]
            name = name[:-1]
        for extra in parts[1:]:
            if extra == '?':
                kwargs['nargs'] = '?'
            elif extra == '*':
                kwargs['nargs'] = '*'
            elif extra in ('int', 'float'):
                kwargs['type'] = int if extra == 'int' else float
        p.add_argument(name, **kwargs)
    for opt, spec in flags.items():
        dest, extra = spec[0], spec[1] if len(spec) > 1 else {}
        if extra == 'store_true':
            extra = {'action': 'store_true'}
        elif extra == 'int':
            extra = {'type': int}
        elif extra == 'list':
            extra = {'nargs': '+'}
        p.add_argument('--' + opt, dest=dest, **(extra or {}))
    return p


class MDatShell(cmd2.Cmd):
    """Interactive mdat shell (cmd2): a thin layer over MDatCommands so that
    GUIs / MCP servers / scripts can drive the very same operations."""
    intro = ("MDat shell - type `help` for the command list, `help <cmd>` for usage. "
             "Start with: open PATH | create PATH | new PATH")
    prompt = '(mdat) '

    def __init__(self, path: Optional[str] = None, readonly: bool = False):
        super().__init__()
        self.cmds = MDatCommands()
        if path:
            self.poutput(self._run(self.cmds.open, path, readonly=readonly))

    def _run(self, fn, *args, **kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"ERROR ({type(e).__name__}): {e}"

    # ---------------------------------------------------------------- session
    def do_open(self, line):
        """Open a container or a standalone dataset: open PATH [--ro] [--view V] [--shuffle] [--distributed] [--create]"""
        a = _parser('path', ro=('readonly', 'store_true'), view=('ds_view',), shuffle=('shuffle', 'store_true'),
                    distributed=('distributed', 'store_true'), create=('create', 'store_true')).parse_args(line)
        self.poutput(self._run(self.cmds.open, a.path, a.readonly, a.ds_view, a.shuffle, a.distributed, a.create))

    def do_create(self, line):
        """Create an empty container: create PATH [--dsmap-bits N] [--modalities text audio ...]"""
        a = _parser('path', dsmap_bits=('dsmap_bits', 'int'), modalities=('modalities', 'list')).parse_args(line)
        self.poutput(self._run(self.cmds.create, a.path, a.dsmap_bits or 8, a.modalities))

    def do_new(self, line):
        """Create a STANDALONE dataset folder: new PATH [--compression N]"""
        a = _parser('path', compression=('compression', 'int')).parse_args(line)
        self.poutput(self._run(self.cmds.new_dataset, a.path, a.compression or 0))

    def do_close(self, line):
        """Close the current container."""
        self.poutput(self.cmds.close())

    def do_status(self, line):
        """What is open: path, active view, strategy, modality, counters, iteration state."""
        self.poutput(self._run(self.cmds.status))

    # --------------------------------------------------------------- datasets
    def do_ls(self, line):
        """List datasets: documents, bytes, internal/linked, strategies."""
        self.poutput(self._run(self.cmds.ls_datasets))

    def do_info(self, line):
        """Dataset details: info [DATASET]"""
        a = _parser('dataset?').parse_args(line)
        self.poutput(self._run(self.cmds.info_dataset, a.dataset))

    def do_link(self, line):
        """Link a standalone dataset: link PATH [--alias NAME]"""
        a = _parser('path', alias=('alias',)).parse_args(line)
        self.poutput(self._run(self.cmds.link, a.path, a.alias))

    def do_unlink(self, line):
        """Drop a linked dataset (external folder untouched): unlink ALIAS"""
        a = _parser('alias').parse_args(line)
        self.poutput(self._run(self.cmds.unlink, a.alias))

    def do_remove(self, line):
        """Remove a dataset from the container: remove NAME [--delete-data]"""
        a = _parser('name', delete_data=('delete_data', 'store_true')).parse_args(line)
        self.poutput(self._run(self.cmds.remove, a.name, a.delete_data))

    def do_add(self, line):
        """(Legacy internal dataset - prefer new + link) add NAME [--compression N]"""
        a = _parser('name', compression=('compression', 'int')).parse_args(line)
        self.poutput(self._run(self.cmds.add, a.name, a.compression or 0))

    # ----------------------------------------------------------------- ingest
    def do_ingest(self, line):
        """Ingest data: ingest DATASET TYPE SOURCE [--key text]   (TYPE: jsonl|lmdb|hf|atlas)"""
        a = _parser('dataset', 'type', 'source', key=('key',)).parse_args(line)
        self.poutput(self._run(self.cmds.ingest, a.dataset, a.type, a.source, a.key or 'text'))

    def do_batch_import(self, line):
        """One dataset per file: batch-import FOLDER [--key text] [--pattern '*.jsonl'] [--recursive]"""
        a = _parser('folder', key=('key',), pattern=('pattern',), recursive=('recursive', 'store_true')).parse_args(line)
        self.poutput(self._run(self.cmds.batch_import, a.folder, a.key or 'text',
                               a.pattern or '*.jsonl', a.recursive))

    # ------------------------------------------------------------- strategies
    def do_strategies(self, line):
        """List registered strategies."""
        self.poutput(self._run(self.cmds.strategies))

    def do_strat_add(self, line):
        """Register a strategy from its JSON config: strat-add NAME config.json"""
        a = _parser('name', 'json_path').parse_args(line)
        self.poutput(self._run(self.cmds.strat_add, a.name, a.json_path))

    def do_strat_del(self, line):
        """Deregister a strategy: strat-del NAME [--delete-files]"""
        a = _parser('name', delete_files=('delete_files', 'store_true')).parse_args(line)
        self.poutput(self._run(self.cmds.strat_del, a.name, a.delete_files))

    def do_set_strategy(self, line):
        """Activate a strategy and compute totals: set-strategy NAME [--seq-len N]"""
        a = _parser('name', seq_len=('seq_len', 'int')).parse_args(line)
        self.poutput(self._run(self.cmds.set_strategy, a.name, a.seq_len))

    def do_pretok(self, line):
        """Pretokenize one dataset: pretok DATASET --strategy NAME [--procs N] [--batch N] [--compression N] [--no-parallel]"""
        a = _parser('dataset', strategy=('strategy',), procs=('processes', 'int'), batch=('batch', 'int'),
                    compression=('compression', 'int'), no_parallel=('parallel', 'store_true')).parse_args(line)
        self.poutput(self._run(self.cmds.pretok, a.dataset, a.strategy, not a.parallel,
                               a.processes, a.batch or 256, a.compression or 0))

    def do_pretok_all(self, line):
        """Pretokenize every dataset: pretok-all --strategy NAME [--procs N] [--batch N] [--compression N]"""
        a = _parser(strategy=('strategy',), procs=('processes', 'int'), batch=('batch', 'int'),
                    compression=('compression', 'int')).parse_args(line)
        if not a.strategy:
            self.poutput("ERROR: --strategy is required")
            return
        self.poutput(self._run(self.cmds.pretok_all, a.strategy, True, a.processes, a.batch or 256, a.compression or 0))

    # ------------------------------------------------------------------ views
    def do_views(self, line):
        """List views with totals and shuffled status."""
        self.poutput(self._run(self.cmds.views))

    def do_view_template(self, line):
        """Write a view JSON template: view-template [OUTFILE]"""
        a = _parser('output?').parse_args(line)
        self.poutput(self._run(self.cmds.view_template, a.output or 'view.json'))

    def do_view_add(self, line):
        """Create a view: view-add NAME --selection 'ds=preserved,other=2G,x=0.5,n=800' [--no-shuffle]  (or: view-add --json FILE)"""
        a = _parser('name?', selection=('selection',), json_path=('json_path',), no_shuffle=('no_shuffle', 'store_true')).parse_args(line)
        self.poutput(self._run(self.cmds.view_add, a.name, a.selection, a.json_path, not a.no_shuffle))

    def do_view_del(self, line):
        """Delete a view (with its shuffle file): view-del NAME"""
        a = _parser('name').parse_args(line)
        self.poutput(self._run(self.cmds.view_del, a.name))

    def do_head(self, line):
        """Head view (first n docs, no reshuffle): head NAME N [--source VIEW]"""
        a = _parser('name', 'n:int', source=('source',)).parse_args(line)
        self.poutput(self._run(self.cmds.head_view, a.name, a.n, a.source))

    def do_range(self, line):
        """Range view (documents [START,END) of a shuffled view): range NAME START [END] [--source VIEW]"""
        a = _parser('name', 'start:int', 'end?:int', source=('source',)).parse_args(line)
        self.poutput(self._run(self.cmds.range_view, a.name, a.start, a.end, a.source))

    def do_use(self, line):
        """Activate a view: use VIEW"""
        self.poutput(self._run(self.cmds.use, line.strip()))

    def do_shuffle(self, line):
        """Shuffle a view (requires writable): shuffle [VIEW]"""
        self.poutput(self._run(self.cmds.shuffle, line.strip() or None))

    def do_repair(self, line):
        """Re-sync linked datasets, drop distributed rows, reshuffle all affected views."""
        self.poutput(self._run(self.cmds.repair))

    def do_check(self, line):
        """Check linked datasets against their container mirrors."""
        self.poutput(self._run(self.cmds.check))

    # -------------------------------------------------------------- training
    def do_modality(self, line):
        """Set iteration modality: modality document|tokens|chunks|chunked_tokens|chunked_for_recurrence [--meta] [--raw]"""
        a = _parser('name', meta=('meta', 'store_true'), raw=('raw', 'store_true')).parse_args(line)
        self.poutput(self._run(self.cmds.modality, a.name, a.meta, a.raw))

    def do_peek(self, line):
        """Inspect documents: peek [DATASET] [--key N] [--n 3] [--chunks]"""
        a = _parser('dataset?', key=('key', 'int'), n=('n', 'int'), chunks=('chunks', 'store_true')).parse_args(line)
        self.poutput(self._run(self.cmds.peek, a.dataset, a.key, a.n or 3, a.chunks))

    def do_iterate(self, line):
        """Consume N items through the training iteration path: iterate [N]"""
        a = _parser('n?:int').parse_args(line)
        self.poutput(self._run(self.cmds.iterate, a.n or 5))

    def do_tokens(self, line):
        """Totals for the active view (documents/tokens/chunks/iterations)."""
        self.poutput(self._run(self.cmds.tokens))

    def do_precompute_dist(self, line):
        """Precompute distributed lengths: precompute-dist NUM_WORKERS (use a highly composite number)"""
        a = _parser('num_workers:int').parse_args(line)
        self.poutput(self._run(self.cmds.precompute_dist, a.num_workers))

    def do_dist_len(self, line):
        """Distributed length for NUM_DEVICES devices: dist-len NUM_DEVICES"""
        a = _parser('num_devices:int').parse_args(line)
        self.poutput(self._run(self.cmds.dist_len, a.num_devices))

    def do_export(self, line):
        """Export a view to JSONL: export VIEW OUTFILE [--limit N]"""
        a = _parser('view', 'output', limit=('limit', 'int')).parse_args(line)
        self.poutput(self._run(self.cmds.export, a.view, a.output, a.limit))

    def do_state_save(self, line):
        """Save the iteration state (document index, chunk step...): state-save FILE"""
        a = _parser('output').parse_args(line)
        self.poutput(self._run(self.cmds.state_save, a.output))

    def do_state_load(self, line):
        """Restore an iteration state: state-load FILE"""
        a = _parser('input_file').parse_args(line)
        self.poutput(self._run(self.cmds.state_load, a.input_file))


def main(argv=None):
    ap = argparse.ArgumentParser(description="MDat interactive shell")
    ap.add_argument('path', nargs='?', help="Container or standalone dataset to open")
    ap.add_argument('--ro', action='store_true', help="open read-only")
    ap.add_argument('--command', '-c', help="run a single command non-interactively and exit")
    args = ap.parse_args(argv)
    shell = MDatShell(path=args.path, readonly=args.ro)
    if args.command:
        shell.onecmd_plus_hooks(args.command)
        return
    shell.cmdloop()


if __name__ == '__main__':
    main()
