"""SQLite metadata catalog shared by MDat (single dataset) and MDatContainer.

Two kinds of catalogs use this manager:

* A *dataset catalog* (``mdat.db`` inside a standalone dataset folder): it
  describes exactly one dataset row in the ``submdat`` table, its storage
  databases, the pretokenization strategies applied to it and the precomputed
  lengths of those strategies.
* A *container catalog* (``mdat_container.db`` inside a container folder): it
  describes views, shuffle files, distributed precomputes and the local
  (aliased) registrations of datasets. A registered dataset is either
  *internal* (its folder lives under ``datasets/`` of this container,
  ``source_rel_path IS NULL``) or *linked* (``source_rel_path`` points, as a
  relative path, to an external standalone dataset folder; the rows describing
  its strategies/stats are refreshed mirrors of the external catalog).

The old ambiguous ``get_manifest(submdat=...|strategy=...|view=...)`` /
``update_manifest(...)`` multiplexers are gone: every table now has its own
explicit accessor.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

# Columns added to the "submdat" table by the linking redesign. Older
# databases get them via ALTER TABLE in _ensure_schema_upgrades().
_SUBMDAT_EXTRA_COLUMNS = {
    'source_rel_path': 'TEXT',   # NULL => internal submdat; else relative link
    'uuid': 'TEXT',              # stable identity of the external dataset
    'generation': 'INTEGER DEFAULT 0',  # bumped whenever documents change
}


class DatabaseManager:
    def __init__(self, db_path: str):
        """
        DatabaseManager manages the SQLite connection and schema initialization.
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, isolation_level=None, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self._ensure_schema_upgrades()
        return self.conn

    def _ensure_schema_upgrades(self):
        """Add the linking-era columns to pre-existing databases (idempotent)."""
        tables = {r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if 'submdat' not in tables:
            return  # fresh database: _initialize_db() creates the full schema
        existing = {x[1] for x in self.conn.execute("PRAGMA table_info('submdat')").fetchall()}
        for col, decl in _SUBMDAT_EXTRA_COLUMNS.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE submdat ADD COLUMN {col} {decl}")

    def _initialize_db(self):
        """
        Initializes the SQLite database schema
        """
        conn = self.connect()
        cur = conn.cursor()

        # Tabella "MANIFEST" (in realtà, sono semplicemente attributi)
        #     dsmap_bits //SHORTINT (8,16 or 32), it indicates with how many bits the id of the subdataset should be represented. If 8 is chosen, it will be much more efficient, but the subdatasets must be limited to 255
        #     modalities //JSON ex {"text"} or {"text","audio","image"}...
        cur.execute("""
        CREATE TABLE IF NOT EXISTS manifest (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dsmap_bits INTEGER NOT NULL CHECK (dsmap_bits IN (8,16,32,64)),
            modalities TEXT NOT NULL -- JSON
        );
        """)

        # Tabella "SUBMDAT"
        # * One row per dataset known to this catalog. In a standalone dataset
        #   catalog there is exactly one row (its name is the dataset name). In
        #   a container catalog a row is either internal (source_rel_path IS
        #   NULL, folder under datasets/) or linked (source_rel_path set:
        #   relative path of the external dataset folder;
        #   document_number/uuid/generation are refreshed mirrors of the
        #   external catalog).
        cur.execute("""
        CREATE TABLE IF NOT EXISTS submdat (
            ds_map_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            data_type TEXT,              -- JSON
            document_number INTEGER CHECK (document_number >= 0),
            modality TEXT,               -- JSON
            data_key TEXT,
            errors_counters TEXT,       -- JSON (can be NULL)
            finalized_at TEXT,          -- DATETIME stored as TEXT
            source_rel_path TEXT,       -- NULL = internal, else relative link
            uuid TEXT,                  -- stable identity of the dataset
            generation INTEGER DEFAULT 0
        );
        """)

        # Tabella "DATABASE"
        # * Actually, not even the dataset holds the data. Data is contained in
        #   highly efficient KV stores, such as LMDB. This table contains all the
        #   useful references to directly call a database. It is used also by
        #   other parts of the mdat, for example, the pretok strategy manager.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS database_ref (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            compression_level INTEGER CHECK (compression_level BETWEEN 0 AND 9),
            map_size INTEGER,
            disk_size INTEGER,
            extra_data TEXT  -- JSON
        );
        """)

        # Tabella "SUBMDAT-DATABASE"
        # *This table maps the relevant databases of the submdat. Usually, we will just have one dataset with raw data (data) and one with metadata (meta). However, for further expansion (classification, multimodality...) it is possible to map an arbitrary amount of datasets
        cur.execute("""
        CREATE TABLE IF NOT EXISTS submdat_database (
            submdat_id INTEGER NOT NULL,
            database_id INTEGER NOT NULL,
            raw_data_bytes INTEGER CHECK (raw_data_bytes >= 0),
            is_data INTEGER DEFAULT 0 CHECK (is_data IN (0,1)),
            is_meta INTEGER DEFAULT 0 CHECK (is_meta IN (0,1)),
            is_extra TEXT, -- JSON stored as TEXT when extra metadata is present; NULL otherwise
            PRIMARY KEY (submdat_id, database_id),
            FOREIGN KEY (submdat_id) REFERENCES submdat(ds_map_id) ON DELETE CASCADE,
            FOREIGN KEY (database_id) REFERENCES database_ref(id) ON DELETE CASCADE
        );
        """)

        # Tabella "PRETOK_STRATEGY"
        # * The dataset can store, alongside with the data, many different
        #   "pretokenizations" of the data, for faster retrieval during model
        #   training. Each strategy can call a "splitter" (built-in or
        #   user-defined) that handles the data processing, ex. splitting in chunks.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy (
            strategy_name TEXT PRIMARY KEY,
            modality TEXT,             -- JSON stored as TEXT
            tokenizer_type TEXT,
            tokenizer_name TEXT,
            tokenizer_args TEXT,       -- JSON stored as TEXT
            bos_token_id INTEGER,
            eos_token_id INTEGER,
            mask_token_id INTEGER,
            wants_raw INTEGER DEFAULT 1, --BOOL STORED AS INT
            splitter_class TEXT,
            splitter_init TEXT,        -- JSON stored as TEXT
            splitter_arguments TEXT,   -- JSON stored as TEXT
            chunk_size INTEGER CHECK (chunk_size >= 0),
            wants_from_db TEXT,        -- JSON stored as TEXT (list of required db names)
            returns TEXT,              -- JSON stored as TEXT
            chunks_datatype TEXT, 
            vocab_size INT,                           
            tokens_datatype TEXT,              
            required_databases TEXT    -- JSON stored as TEXT
        );
        """)

        # Tabella "PRETOK_STRATEGY-SUBMDAT"
        # Stats of each pretokenization applied to each dataset.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy_submdat (
            strategy_name TEXT NOT NULL,
            submdat_id INTEGER NOT NULL,
            finalized_at TEXT,  -- ISO8601 DATETIME stored as TEXT (can be NULL)
            total_tokens INTEGER CHECK (total_tokens >= 0),
            max_tokens_per_doc INTEGER CHECK (max_tokens_per_doc >= 0),
            total_chunks INTEGER CHECK (total_chunks >= 0),
            max_chunks_per_doc INTEGER CHECK (max_chunks_per_doc >= 0),
            processed_docs INTEGER CHECK (processed_docs >= 0),
            PRIMARY KEY (strategy_name, submdat_id),
            FOREIGN KEY (strategy_name) REFERENCES pretok_strategy(strategy_name) ON DELETE CASCADE,
            FOREIGN KEY (submdat_id) REFERENCES submdat(ds_map_id) ON DELETE CASCADE
        );
        """)

        # Tabella "PRETOK_STRATEGY-SUBMDAT-PRECOMPUTED-LENGTH"
        # * Iteration lengths, per chunk_multiplier, when entire datasets are selected.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy_submdat_precomputed_length (
            strategy_name TEXT NOT NULL,
            submdat_id INTEGER NOT NULL,
            multiplier INTEGER NOT NULL CHECK (multiplier >= 0),
            precomputed_length INTEGER CHECK (precomputed_length >= 0),
            PRIMARY KEY (strategy_name, submdat_id, multiplier),
            FOREIGN KEY (strategy_name) REFERENCES pretok_strategy(strategy_name) ON DELETE CASCADE,
            FOREIGN KEY (submdat_id) REFERENCES submdat(ds_map_id) ON DELETE CASCADE
        );
        """)

        # Tabella "PRETOK_STRATEGY-DB"
        # * Which storage databases hold the outputs of a strategy on a dataset.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy_db (
            db_id INTEGER NOT NULL,
            strategy_name TEXT NOT NULL,
            submdat_id INTEGER NOT NULL,
            is_complete INTEGER DEFAULT 0 CHECK (is_complete IN (0,1)),
            datatype TEXT,
            is_tokens INTEGER DEFAULT 0 CHECK (is_tokens IN (0,1)),
            is_chunks INTEGER DEFAULT 0 CHECK (is_chunks IN (0,1)),
            is_extra TEXT, -- JSON stored as TEXT (nullable)
            FOREIGN KEY (strategy_name) REFERENCES pretok_strategy(strategy_name) ON DELETE CASCADE,
            FOREIGN KEY (db_id) REFERENCES database_ref(id) ON DELETE CASCADE,
            FOREIGN KEY (submdat_id) REFERENCES submdat(ds_map_id) ON DELETE CASCADE,
            CHECK (
                (COALESCE(is_tokens,0) + COALESCE(is_chunks,0) + CASE WHEN is_extra IS NOT NULL THEN 1 ELSE 0 END) = 1
            )
        );
        """)

        # Tabella "PRETOK_STRATEGY-VIEW-SUBMDAT-PRECOMPUTED-LENGTH"
        # * Iteration lengths, per chunk_multiplier, for partially selected
        #   datasets inside a specific view. Container-owned.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy_view_submdat_precomputed_length (
            strategy_name TEXT NOT NULL,
            view_name TEXT NOT NULL,
            submdat_id INTEGER NOT NULL,
            multiplier INTEGER NOT NULL CHECK (multiplier >= 0),
            precomputed_length INTEGER CHECK (precomputed_length >= 0),
            PRIMARY KEY (strategy_name, view_name, submdat_id, multiplier),
            FOREIGN KEY (strategy_name) REFERENCES pretok_strategy(strategy_name) ON DELETE CASCADE,
            FOREIGN KEY (view_name) REFERENCES view(view_name) ON DELETE CASCADE,
            FOREIGN KEY (submdat_id) REFERENCES submdat(ds_map_id) ON DELETE CASCADE
        );
        """)

        # Tabella "VIEW" (container-owned)
        #     view_name //STR (KEY)
        #     shuffle_struct_format //STRING (NULL = view not shuffled)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS view (
            view_name TEXT PRIMARY KEY,
            shuffle_struct_format TEXT
        );
        """)

        # Tabella "VIEW-SUBMDAT" (container-owned)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS view_submdat (
            view_name TEXT NOT NULL,
            submdat_id INTEGER NOT NULL,
            is_skipped INTEGER DEFAULT 0 CHECK (is_skipped IN (0,1)),
            is_preserved INTEGER DEFAULT 0 CHECK (is_preserved IN (0,1)),
            is_partial INTEGER DEFAULT 0 CHECK (is_partial IN (0,1)),
            document_number INTEGER,  -- NULL if skipped or preserved, otherwise number of selected documents
            bytes_criteria INTEGER,   -- reference if selection was based on data size
            PRIMARY KEY (view_name, submdat_id)      
        );
        """)

        # Tabella "PRETOK_STRATEGY-VIEW-DISTRIBUTED-LENGTH" (container-owned)
        # * Precomputed chunk counts for distributed training configurations
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy_view_distributed_length (
            strategy_name TEXT NOT NULL,
            view_name TEXT NOT NULL,
            num_workers INTEGER NOT NULL CHECK (num_workers > 0),
            worker_id INTEGER NOT NULL CHECK (worker_id >= 0),
            chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
            chunk_multiplier INTEGER NOT NULL CHECK (chunk_multiplier >= 0),
            PRIMARY KEY (strategy_name, view_name, num_workers, worker_id, chunk_multiplier),
            FOREIGN KEY (strategy_name) REFERENCES pretok_strategy(strategy_name) ON DELETE CASCADE,
            FOREIGN KEY (view_name) REFERENCES view(view_name) ON DELETE CASCADE,
            CHECK (worker_id < num_workers)
        );
        """)

        # Add index
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_pretok_distributed_lookup 
        ON pretok_strategy_view_distributed_length(view_name, strategy_name, num_workers);
        """)
        # Indexes to speed up common access patterns
        cur.execute("CREATE INDEX IF NOT EXISTS idx_submdat_dsmap ON submdat(ds_map_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_submdat_name ON submdat(name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dbref_type ON database_ref(type);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pretok_strategy_modality ON pretok_strategy(modality);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pretok_strategy_submdat ON pretok_strategy_submdat(submdat_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pretok_strategy_db_strategy ON pretok_strategy_db(strategy_name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_submdat_database_submdat ON submdat_database(submdat_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_view_submdat_view ON view_submdat(view_name);")

        conn.commit()

    # ------------------------------------------------------------------ helpers
    def _dict_from_row(self, cursor, row):
        if not row:
            return {}
        return dict(zip([x[0] for x in cursor.description], row))

    def _fetchone(self, q, params=()):
        c = self.connect().cursor()
        r = c.execute(q, params).fetchone()
        return self._dict_from_row(c, r)

    def _fetchall(self, q, params=()):
        c = self.connect().cursor()
        rows = c.execute(q, params).fetchall()
        return [dict(zip([x[0] for x in c.description], r)) for r in rows]

    def _update_table(self, table, data, where_clause=None, where_params=()):
        if not data:
            return
        cols = ",".join([f"{k}=?" for k in data.keys()])
        q = f"UPDATE {table} SET {cols}"
        params = list(data.values())
        if where_clause:
            q += f" WHERE {where_clause}"
            params.extend(where_params)
        c = self.connect().cursor()
        c.execute(q, tuple(params))
        self.connect().commit()

    def _insert(self, table, data):
        """Generic insert. `data` is dict of col->value. Returns lastrowid."""
        cols = ",".join(data.keys())
        vals = ",".join(["?" for _ in data])
        q = f"INSERT INTO {table}({cols}) VALUES({vals})"
        c = self.connect().cursor()
        c.execute(q, tuple(data.values()))
        self.connect().commit()
        return c.lastrowid

    # ------------------------------------------------------------- row accessors
    # One explicit accessor pair per table. (The old get_manifest/update_manifest
    # multiplexers used to hide which of the four tables was touched.)

    def get_manifest(self) -> Dict[str, Any]:
        """The single catalog-level manifest row."""
        return self._fetchone("SELECT * FROM manifest LIMIT 1")

    def create_manifest(self, dsmap_bits, modalities):
        return self._insert("manifest", {"dsmap_bits": dsmap_bits, "modalities": json.dumps(modalities)})

    def update_manifest(self, data) -> None:
        self._update_table('manifest', dict(data))

    def get_submdat_row(self, submdat) -> Dict[str, Any]:
        """One row of the submdat table, addressed by int id or by name."""
        if isinstance(submdat, int):
            return self._fetchone("SELECT * FROM submdat WHERE ds_map_id=?", (submdat,))
        return self._fetchone("SELECT * FROM submdat WHERE name=?", (submdat,))

    def update_submdat_row(self, submdat, data) -> None:
        if isinstance(submdat, int):
            self._update_table('submdat', dict(data), 'ds_map_id=?', (submdat,))
        else:
            self._update_table('submdat', dict(data), 'name=?', (submdat,))

    def get_strategy_row(self, strategy_name) -> Dict[str, Any]:
        return self._fetchone("SELECT * FROM pretok_strategy WHERE strategy_name=?", (strategy_name,))

    def update_strategy_row(self, strategy_name, data) -> None:
        self._update_table('pretok_strategy', dict(data), 'strategy_name=?', (strategy_name,))

    def get_view_row(self, view_name) -> Dict[str, Any]:
        return self._fetchone("SELECT * FROM view WHERE view_name=?", (view_name,))

    def update_view_row(self, view_name, data) -> None:
        self._update_table('view', dict(data), 'view_name=?', (view_name,))

    def bump_generation(self, submdat_id) -> int:
        """Record that the document set of this dataset changed. Returns new value."""
        c = self.connect().cursor()
        c.execute("UPDATE submdat SET generation = COALESCE(generation,0) + 1 WHERE ds_map_id=?", (submdat_id,))
        self.connect().commit()
        r = c.execute("SELECT generation FROM submdat WHERE ds_map_id=?", (submdat_id,)).fetchone()
        return int(r[0]) if r and r[0] is not None else 0

    # -------------------------------------------------------------------- dsmap
    def get_dsmap(self, key, value=None):
        """Unified accessor for dsmap-related lookups. Returns either scalar or dict as original."""
        c = self.connect().cursor()
        if key == "name" and value is not None:
            r = c.execute("SELECT ds_map_id FROM submdat WHERE name=?", (value,)).fetchone()
            return r[0] if r else None
        if key == "id" and value is not None:
            r = c.execute("SELECT name FROM submdat WHERE ds_map_id=?", (value,)).fetchone()
            return r[0] if r else None
        if key == "max_id":
            r = c.execute("SELECT MAX(ds_map_id) FROM submdat").fetchone()
            return (r[0] or 0)
        if key == "name":
            rows = c.execute("SELECT ds_map_id,name FROM submdat").fetchall()
            return {r[1]: r[0] for r in rows}
        rows = c.execute("SELECT ds_map_id,name FROM submdat").fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------ views
    def ask_view(self, view, keys):
        c = self.connect().cursor()
        if isinstance(keys, str):
            keys = [keys]
        mapping = {
            'preserved_submdats': 'is_preserved',
            'partial_submdats': 'is_partial',
            'skipped_submdats': 'is_skipped'
        }
        res = {}
        for k in keys:
            col = mapping[k]
            rows = c.execute(f"SELECT submdat_id FROM view_submdat WHERE view_name=? AND {col}=1", (view,)).fetchall()
            res[k] = [r[0] for r in rows]
        return res if len(keys) > 1 else res[keys[0]]

    def add_view(self, view_name):
        c = self.connect().cursor()
        c.execute("INSERT INTO view(view_name,shuffle_struct_format) VALUES (?,?)", (view_name, None))
        self.connect().commit()
        return view_name

    def remove_view(self, view_name):
        """Delete a view and every row that depends on it."""
        c = self.connect().cursor()
        c.execute("DELETE FROM view_submdat WHERE view_name=?", (view_name,))
        c.execute("DELETE FROM pretok_strategy_view_submdat_precomputed_length WHERE view_name=?", (view_name,))
        c.execute("DELETE FROM pretok_strategy_view_distributed_length WHERE view_name=?", (view_name,))
        c.execute("DELETE FROM view WHERE view_name=?", (view_name,))
        self.connect().commit()

    def list_views(self):
        rows = self._fetchall("SELECT view_name FROM view")
        return [row['view_name'] for row in rows]

    # -------------------------------------------------------------- submdats
    def add_submdat(self, submdat_name):
        return self._insert('submdat', {'name': submdat_name})

    def add_submdat_link(self, submdat_name, source_rel_path, uuid=None,
                         data_type=None, modality=None, data_key=None,
                         document_number=0, generation=0):
        """Register an *external* dataset folder under a local alias."""
        return self._insert('submdat', {
            'name': submdat_name,
            'source_rel_path': source_rel_path,
            'uuid': uuid,
            'data_type': data_type,
            'modality': modality,
            'data_key': data_key,
            'document_number': document_number,
            'generation': generation,
        })

    def list_submdats(self) -> List[Dict[str, Any]]:
        """Every dataset row, internal and linked."""
        return self._fetchall("SELECT * FROM submdat ORDER BY ds_map_id")

    def remove_submdat(self, submdat_id):
        """Delete a dataset row.

        Removes its links to views and its strategy associations (cascades also
        delete per-strategy precomputed lengths and strategy_db rows) and then
        garbage-collects database_ref entries that nothing references anymore.
        """
        submdat_id = int(submdat_id)
        c = self.connect().cursor()
        c.execute("DELETE FROM view_submdat WHERE submdat_id=?", (submdat_id,))
        c.execute("DELETE FROM pretok_strategy_view_submdat_precomputed_length WHERE submdat_id=?", (submdat_id,))
        c.execute("DELETE FROM submdat WHERE ds_map_id=?", (submdat_id,))
        # database_ref rows not referenced by any remaining association:
        c.execute("""DELETE FROM database_ref WHERE id NOT IN (
                        SELECT database_id FROM submdat_database
                        UNION SELECT db_id FROM pretok_strategy_db)""")
        self.connect().commit()

    def get_submdat_info(self, submdat_id):
        submdat_id = int(submdat_id)
        r = self._fetchone(
            """
            SELECT s.ds_map_id AS submdat_id, s.document_number,
                   sd.raw_data_bytes
            FROM submdat AS s
            LEFT JOIN submdat_database AS sd ON sd.submdat_id = s.ds_map_id
            WHERE s.ds_map_id = ? AND sd.is_data = ?
            """,
            (submdat_id, 1,)
        )
        return r

    # --------------------------------------------------------------- databases
    def add_database(self, _type, compression_level, map_size, disk_size, extra_data):
        return self._insert('database_ref', {
            'type': _type,
            'compression_level': compression_level,
            'map_size': map_size,
            'disk_size': disk_size,
            'extra_data': extra_data
        })

    def link_submdat_database(self, submdat_id, database_id, raw_data_bytes, is_data, is_meta, is_extra):
        return self._insert('submdat_database', {
            'submdat_id': submdat_id,
            'database_id': database_id,
            'raw_data_bytes': raw_data_bytes,
            'is_data': int(bool(is_data)),
            'is_meta': int(bool(is_meta)),
            'is_extra': is_extra  # JSON string or NULL (the column is TEXT)
        })

    def get_storage_db_info(self, database_id):
        q = "SELECT * FROM database_ref WHERE database_ref.id = ?"
        return self._fetchone(q, (int(database_id),))

    def get_submdat_databases(self, submdat_id, database_id=None, database_type=None):
        q = """
            SELECT dr.*, sd.raw_data_bytes, sd.is_data, sd.is_meta, sd.is_extra
            FROM database_ref dr
            JOIN submdat_database sd ON dr.id = sd.database_id
            WHERE sd.submdat_id = ?
        """
        params = [submdat_id]
        if database_id is not None:
            q += " AND sd.database_id = ?"
            params.append(database_id)
        result = self._fetchall(q, tuple(params))
        if database_type in ('data', 'meta'):
            flag = 'is_data' if database_type == 'data' else 'is_meta'
            for r in result:
                if r[flag] == 1:
                    return r
            return None
        return result[0] if database_id is not None and result else result

    def update_submdat_database(self, submdat_id, database_id, data):
        dr_allowed = {'type', 'compression_level', 'map_size', 'disk_size', 'extra_data'}
        sd_allowed = {'raw_data_bytes', 'is_data', 'is_meta', 'is_extra'}
        dr_fields = {k: v for k, v in data.items() if k in dr_allowed}
        sd_fields = {k: v for k, v in data.items() if k in sd_allowed}
        if dr_fields:
            self._update_table('database_ref', dr_fields, 'id=?', (database_id,))
        if sd_fields:
            self._update_table('submdat_database', sd_fields, 'submdat_id=? AND database_id=?', (submdat_id, database_id))

    # ---------------------------------------------------- strategy <-> submdat
    def get_strategy_submdats(self, strategy_name, submdat_id=None):
        q = "SELECT * FROM pretok_strategy_submdat WHERE strategy_name = ?"
        params = [strategy_name]
        if submdat_id is not None:
            q += " AND submdat_id = ?"
            params.append(submdat_id)
        rows = self._fetchall(q, tuple(params))
        return rows[0] if submdat_id is not None and rows else rows

    def update_strategy_submdat(self, strategy_name, submdat_id, data):
        self._update_table('pretok_strategy_submdat', data, 'strategy_name=? AND submdat_id=?', (strategy_name, submdat_id))

    def add_strategy_submdat(self, strategy_name, submdat_id, data=None):
        data = data or {}
        payload = {'strategy_name': strategy_name, 'submdat_id': submdat_id}
        payload.update(data)
        return self._insert('pretok_strategy_submdat', payload)

    def remove_strategy_submdat(self, strategy_name, submdat_id):
        c = self.connect().cursor()
        c.execute("DELETE FROM pretok_strategy_submdat WHERE strategy_name=? AND submdat_id=?", (strategy_name, submdat_id))
        self.connect().commit()

    # ------------------------------------------------------------ strategy dbs
    def get_strategy_dbs(self, strategy_name, submdat_id, db_id=None):
        q = """SELECT psd.db_id, psd.strategy_name, psd.submdat_id, psd.is_complete,
                      psd.datatype, psd.is_tokens, psd.is_chunks, psd.is_extra,
                      dr.id, dr.type, dr.compression_level, dr.map_size, dr.disk_size, dr.extra_data
               FROM pretok_strategy_db psd
               JOIN database_ref dr ON psd.db_id = dr.id
               WHERE psd.strategy_name=? AND psd.submdat_id=?"""
        params = [strategy_name, submdat_id]
        if db_id is not None:
            q += " AND psd.db_id=?"
            params.append(db_id)
        rows = self._fetchall(q, tuple(params))
        return rows[0] if db_id is not None and rows else rows

    def update_strategy_db(self, database_id, data):
        psd_allowed = {'is_complete', 'datatype', 'is_tokens', 'is_chunks', 'is_extra'}
        dr_allowed = {'type', 'compression_level', 'map_size', 'disk_size', 'extra_data'}

        psd_fields = {k: v for k, v in data.items() if k in psd_allowed}
        dr_fields = {k: v for k, v in data.items() if k in dr_allowed}

        if psd_fields:
            self._update_table('pretok_strategy_db', psd_fields, 'db_id=?', (database_id,))

        if dr_fields:
            self._update_table('database_ref', dr_fields, 'id=?', (database_id,))

    def add_strategy_db(self, database_id, strategy_name, submdat_id, datatype, is_tokens, is_chunks, is_extra, is_complete=1):
        c = self.connect().cursor()
        c.execute("INSERT INTO pretok_strategy_db(db_id,strategy_name,submdat_id,datatype,is_tokens,is_chunks,is_extra,is_complete) VALUES(?,?,?,?,?,?,?,?)", (database_id, strategy_name, submdat_id, datatype, is_tokens, is_chunks, is_extra, is_complete))
        db_id = c.lastrowid
        self.connect().commit()
        return db_id

    def remove_strategy_db(self, db_id):
        c = self.connect().cursor()
        c.execute("DELETE FROM pretok_strategy_db WHERE db_id=?", (db_id,))
        self.connect().commit()

    # ------------------------------------------------------------ strategies
    def list_strategies(self):
        rows = self._fetchall("SELECT strategy_name FROM pretok_strategy")
        return [row['strategy_name'] for row in rows]

    def remove_strategy(self, strategy_name):
        """Delete a strategy row; FK cascades remove its associations, precomputed
        lengths and (view/distributed) dependents."""
        c = self.connect().cursor()
        c.execute("DELETE FROM pretok_strategy WHERE strategy_name = ?", (strategy_name,))
        self.connect().commit()

    # ------------------------------------------------------- precomputed lengths
    def set_strategy_view_precomp(self, strategy_name, view_name, submdat_id, multiplier, precomputed_length):
        c = self.connect().cursor()
        c.execute("""INSERT OR REPLACE INTO pretok_strategy_view_submdat_precomputed_length
                     (strategy_name,view_name,submdat_id,multiplier,precomputed_length) VALUES(?,?,?,?,?)""",
                  (strategy_name, view_name, submdat_id, multiplier, precomputed_length))
        self.connect().commit()

    def set_strategy_view_precomp_many(self, rows):
        """Batch version of set_strategy_view_precomp: rows is an iterable of
        (strategy_name, view_name, submdat_id, multiplier, precomputed_length).
        One transaction instead of thousands of commits."""
        if not rows:
            return
        c = self.connect().cursor()
        c.executemany("""INSERT OR REPLACE INTO pretok_strategy_view_submdat_precomputed_length
                         (strategy_name,view_name,submdat_id,multiplier,precomputed_length) VALUES(?,?,?,?,?)""",
                      list(rows))
        self.connect().commit()

    def get_strategy_precomp(self, strategy_name, view_name=None, submdat_id=None, multiplier=None):
        if view_name:
            table = 'pretok_strategy_view_submdat_precomputed_length'
        else:
            table = 'pretok_strategy_submdat_precomputed_length'
        q = f"SELECT * FROM {table} WHERE strategy_name=?"
        params = [strategy_name]
        if view_name:
            q += " AND view_name=?"
            params.append(view_name)
        if submdat_id:
            q += " AND submdat_id=?"
            params.append(submdat_id)
        if multiplier:
            q += " AND multiplier=?"
            params.append(multiplier)
        return self._fetchall(q, tuple(params))

    def set_strategy_submdat_precomp(self, strategy_name, submdat_id, multiplier, precomputed_length):
        c = self.connect().cursor()
        c.execute("""INSERT OR REPLACE INTO pretok_strategy_submdat_precomputed_length
                     (strategy_name,submdat_id,multiplier,precomputed_length) VALUES(?,?,?,?)""",
                  (strategy_name, submdat_id, multiplier, precomputed_length))
        self.connect().commit()

    # --------------------------------------------------------- view <-> submdat
    def link_submdat_view(self, submdat_id, view_name, is_skipped, is_partial, is_preserved, document_number, bytes_criteria):
        """Insert or update link into view_submdat (UPSERT semantics)."""
        flags = (int(bool(is_skipped)), int(bool(is_partial)), int(bool(is_preserved)))
        if sum(flags) != 1:
            raise ValueError("Exactly one of is_skipped, is_partial, is_preserved must be true")

        c = self.connect().cursor()

        # Check if the row already exists
        exists = c.execute(
            "SELECT 1 FROM view_submdat WHERE submdat_id=? AND view_name=?",
            (submdat_id, view_name)
        ).fetchone()

        if exists:
            # Update existing entry
            c.execute("""
                UPDATE view_submdat
                SET is_skipped=?, is_partial=?, is_preserved=?, document_number=?, bytes_criteria=?
                WHERE submdat_id=? AND view_name=?
            """, (flags[0], flags[1], flags[2], document_number, bytes_criteria, submdat_id, view_name))
        else:
            # Insert new entry
            c.execute("""
                INSERT INTO view_submdat(submdat_id, view_name, is_skipped, is_partial, is_preserved, document_number, bytes_criteria)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (submdat_id, view_name, flags[0], flags[1], flags[2], document_number, bytes_criteria))

        self.connect().commit()

    def get_view_submdat(self, view_name, submdat_id):
        return self._fetchone(
            "SELECT view_name, submdat_id, is_skipped, is_preserved, is_partial, document_number, bytes_criteria FROM view_submdat WHERE view_name=? AND submdat_id=?",
            (view_name, submdat_id)
        )

    def update_view_submdat(self, view_name, submdat_id, data):
        allowed = {'is_skipped', 'is_preserved', 'is_partial', 'document_number', 'bytes_criteria'}
        upd = {k: v for k, v in (data or {}).items() if k in allowed}
        if not upd:
            return
        self._update_table('view_submdat', upd, 'view_name=? AND submdat_id=?', (view_name, submdat_id))

    def get_view_totals(self, view_name, criterion='active'):
        """Return aggregated document_number and bytes_criteria for a view.
        criterion: 'active' (default) : rows where is_skipped=0
                   'preserved' : rows where is_preserved=1
                   'all' : all rows
        Returns dict { 'document_number': int, 'bytes_criteria': int }
        """
        c = self.connect().cursor()
        if criterion == 'preserved':
            where = 'WHERE view_name=? AND is_preserved=1'
        elif criterion == 'all':
            where = 'WHERE view_name=?'
        else:
            where = 'WHERE view_name=? AND is_skipped=0'
        q = f"SELECT SUM(COALESCE(document_number,0)) as document_number, SUM(COALESCE(bytes_criteria,0)) as bytes_criteria FROM view_submdat {where}"
        r = c.execute(q, (view_name,)).fetchone()
        if not r:
            return {'document_number': 0, 'bytes_criteria': 0}
        return {'document_number': int(r[0] or 0), 'bytes_criteria': int(r[1] or 0)}

    # ------------------------------------------------- linked-dataset mirrors
    # A container stores refreshed mirrors of the external catalogs of its
    # linked datasets so that every existing query path (strategy stats,
    # precomputed lengths, storage-db configuration) works unchanged.

    STRATEGY_COLUMNS = ['strategy_name', 'modality', 'tokenizer_type', 'tokenizer_name',
                        'tokenizer_args', 'bos_token_id', 'eos_token_id', 'mask_token_id',
                        'wants_raw', 'splitter_class', 'splitter_init', 'splitter_arguments',
                        'chunk_size', 'wants_from_db', 'returns', 'chunks_datatype',
                        'vocab_size', 'tokens_datatype', 'required_databases']

    def upsert_strategy_row(self, row: Dict[str, Any]):
        """Mirror a full pretok_strategy row (insert or overwrite)."""
        payload = {col: row.get(col) for col in self.STRATEGY_COLUMNS}
        cols = ",".join(payload.keys())
        vals = ",".join(["?"] * len(payload))
        updates = ",".join([f"{c}=excluded.{c}" for c in payload if c != 'strategy_name'])
        q = (f"INSERT INTO pretok_strategy({cols}) VALUES({vals}) "
             f"ON CONFLICT(strategy_name) DO UPDATE SET {updates}")
        c = self.connect().cursor()
        c.execute(q, tuple(payload.values()))
        self.connect().commit()

    def upsert_strategy_submdat(self, strategy_name, submdat_id, stats: Dict[str, Any]):
        """Mirror per (strategy, dataset) stats (insert or overwrite)."""
        payload = {'strategy_name': strategy_name, 'submdat_id': submdat_id,
                   'finalized_at': stats.get('finalized_at'),
                   'total_tokens': stats.get('total_tokens'),
                   'max_tokens_per_doc': stats.get('max_tokens_per_doc'),
                   'total_chunks': stats.get('total_chunks'),
                   'max_chunks_per_doc': stats.get('max_chunks_per_doc'),
                   'processed_docs': stats.get('processed_docs')}
        cols = ",".join(payload.keys())
        vals = ",".join(["?"] * len(payload))
        updates = ",".join([f"{c}=excluded.{c}" for c in payload if c not in ('strategy_name', 'submdat_id')])
        q = (f"INSERT INTO pretok_strategy_submdat({cols}) VALUES({vals}) "
             f"ON CONFLICT(strategy_name,submdat_id) DO UPDATE SET {updates}")
        c = self.connect().cursor()
        c.execute(q, tuple(payload.values()))
        self.connect().commit()

    def upsert_mirrored_database_ref(self, db_row: Dict[str, Any]) -> int:
        """Mirror a database_ref row of an external catalog; returns the local id.

        The local extra_data column carries a tag pointing to the external
        database id, so repeated refreshes update the same local row.
        """
        tag = json.dumps({'mirrored_source_db_id': db_row.get('id')})
        existing = self._fetchone("SELECT id FROM database_ref WHERE extra_data=?", (tag,))
        data = {'type': db_row.get('type'),
                'compression_level': db_row.get('compression_level'),
                'map_size': db_row.get('map_size'),
                'disk_size': db_row.get('disk_size'),
                'extra_data': tag}
        if existing:
            self._update_table('database_ref', data, 'id=?', (existing['id'],))
            return existing['id']
        return self._insert('database_ref', data)

    def upsert_mirrored_strategy_db(self, db_id, strategy_name, submdat_id, sdb_row: Dict[str, Any]):
        """Mirror a pretok_strategy_db association for the given (strategy, dataset)."""
        c = self.connect().cursor()
        c.execute("DELETE FROM pretok_strategy_db WHERE db_id=? AND strategy_name=? AND submdat_id=?",
                  (db_id, strategy_name, submdat_id))
        c.execute("INSERT INTO pretok_strategy_db(db_id,strategy_name,submdat_id,is_complete,datatype,is_tokens,is_chunks,is_extra) VALUES(?,?,?,?,?,?,?,?)",
                  (db_id, strategy_name, submdat_id, sdb_row.get('is_complete'), sdb_row.get('datatype'),
                   sdb_row.get('is_tokens'), sdb_row.get('is_chunks'), sdb_row.get('is_extra')))
        self.connect().commit()

    # -------------------------------------------------- distributed precomputes

    def _store_distributed_precomputed_length(self, view_name, strategy_name, num_workers, worker_chunk_counts, chunk_multiplier):
        """
        Store precomputed chunk counts for distributed training.

        Args:
            view_name: Name of the view
            strategy_name: Name of the pretokenization strategy
            num_workers: Number of workers this was precomputed for
            worker_chunk_counts: List of chunk counts, one per worker
            chunk_multiplier: the base sequence length multiplier
        """
        conn = self.connect()
        cur = conn.cursor()

        # Delete existing entries
        cur.execute("""
            DELETE FROM pretok_strategy_view_distributed_length 
            WHERE view_name = ? AND strategy_name = ? AND chunk_multiplier = ? AND num_workers = ?
        """, (view_name, strategy_name, chunk_multiplier, num_workers))

        # Insert new entries
        for worker_id, chunk_count in enumerate(worker_chunk_counts):
            cur.execute("""
                INSERT INTO pretok_strategy_view_distributed_length 
                (view_name, strategy_name, chunk_multiplier, num_workers, worker_id, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (view_name, strategy_name, chunk_multiplier, num_workers, worker_id, chunk_count))
        conn.commit()
        print(f"Stored precomputed lengths for {num_workers} workers at chunk multiplier {chunk_multiplier} in database")

    def get_worker_length_distributed(self, target_num_workers, target_worker_id, view_name, strategy_name, chunk_multiplier):
        """
        Get the precomputed chunk count for a specific worker.
        If exact match exists, return it. If precomputed for a multiple, calculate from that.

        Args:
            target_num_workers: Number of workers in current training setup
            target_worker_id: ID of this worker (rank)

        Returns:
            int: Number of chunks this worker should process
        """
        conn = self.connect()
        cur = conn.cursor()

        # Try exact match first
        result = cur.execute("""
            SELECT chunk_count FROM pretok_strategy_view_distributed_length
            WHERE view_name = ? AND strategy_name = ? 
            AND num_workers = ? AND worker_id = ? AND chunk_multiplier = ?
        """, (view_name, strategy_name, target_num_workers, target_worker_id, chunk_multiplier)).fetchone()

        if result:
            return result[0]

        # No exact match - try to find a precomputed multiple
        available = cur.execute("""
            SELECT DISTINCT num_workers FROM pretok_strategy_view_distributed_length
            WHERE view_name = ? AND strategy_name = ? AND chunk_multiplier = ?
            ORDER BY num_workers DESC
        """, (view_name, strategy_name, chunk_multiplier)).fetchall()

        for (precomputed_workers,) in available:
            if precomputed_workers % target_num_workers == 0:
                # Fetch all worker counts from the precomputed configuration
                precomputed_counts = cur.execute("""
                    SELECT worker_id, chunk_count FROM pretok_strategy_view_distributed_length
                    WHERE view_name = ? AND strategy_name = ? AND chunk_multiplier = ? AND num_workers = ? 
                    ORDER BY worker_id
                """, (view_name, strategy_name, chunk_multiplier, precomputed_workers)).fetchall()

                # Sum the relevant workers
                total = 0
                for worker_id, chunk_count in precomputed_counts:
                    if worker_id % target_num_workers == target_worker_id:
                        total += chunk_count

                return total

        raise ValueError(
            f"No precomputed length found for {target_num_workers} workers at chunk multiplier {chunk_multiplier} "
            f"Available configurations: {[w[0] for w in available]}. "
            f"Please run precompute_chunks_distributed() with a number divisible by {target_num_workers} and suitable for sequence length {chunk_multiplier} the base sequence length"
        )
