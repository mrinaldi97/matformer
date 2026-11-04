"""SQLite schema and metadata queries for existing MDAT datasets."""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

class DatabaseManager:
    def __init__(self, db_path: str, readonly: bool = False):
        """
        DatabaseManager manages the SQLite connection and schema initialization.
        """
        self.db_path = db_path
        self.readonly = readonly
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        if self.conn is None:
            path = Path(self.db_path).resolve().as_uri() + '?mode=ro' if self.readonly else self.db_path
            self.conn = sqlite3.connect(path, uri=self.readonly, isolation_level=None,
                                        detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON;")
        return self.conn

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

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
        # * A Mdat doesn't directly contain data. Data is contained in the sub-mdat objects, subclasses of the parent dataset. In this way, it is possible to easily separe logically different dataset to be used to train a model
        #
        #     name //STR
        #     ds_map_id //PROGRESSIVE-ID
        #     data_type //JSON ex {"text"} or {"text","audio","image"}
        #     document_number //LONG-UNSIGNED-INT how many documents in the dataset
        #     modality //JSON ex {"text"} or {"text","audio","image"}...
        #     data_key //Short string, the original key of the data (ex. 'text')
        #     errors_counters //JSON
        #     finalized_at //DATETIME
        cur.execute("""
        CREATE TABLE IF NOT EXISTS submdat (
            ds_map_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            data_type TEXT,              -- JSON
            document_number INTEGER CHECK (document_number >= 0),
            modality TEXT,               -- JSON
            data_key TEXT,
            errors_counters TEXT,       -- JSON (can be NULL)
            finalized_at TEXT           -- DATETIME stored as TEXT
        );
        """)

        # Tabella "DATABASE"
        # * Actually, not even the Submdat holds the data. Data is contained in highly efficient KV stores, such as LMDB. This table contains all the useful references to directly call a database. It is used also by other parts of the mdat, for example, the pretok strategy manager.
        #
        #     ID //PROGRESSIVE-ID
        #     Type //STRING ex LMDB
        #     Compression_level //SHORTINT (0-9), level of compression
        #     map_size //LONGINT lmdb requires a "map_size" parameter
        #     disk_size //LONGINT the actual size on disk
        #     extra_data //JSON (for further expansions)
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
        #     submdat_id //linked to submdat
        #     database_id //linked to database
        #     raw_data_bytes // LONGINT the amount of bytes of the original data (for example, chars in case of documents)
        #     is_data //BOOL (mututally exclusive)
        #     is_meta //BOOL (mututally exclusive)
        #     is_extra //JSON or NULL (mututally exclusive)
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
        # * The MDAT can store, alongside with the data, many different "pretokenizations" of the data, for faster retrieval during model training.
        # * Each pretok strategy can call a "splitter" (either built-in or user-defined) that handles the data processing, ex, splitting in chunks
        #
        #     strategy_name //STR
        #     modality  //JSON ex {"text"} or {"text","audio","image"}...
        #     tokenizer_type //STR ex "huggingface"
        #     tokenizer_name //STR ex "sapienzanlp/Minerva-350M-base-v1.0"
        #     tokenizer_args //JSON, extra arguments for the tokenizer
        #     bos_token_id //SINGLE INT
        #     eos_token_id //SINGLE INT
        #     mask_token_id //SINGLE INT
        #     splitter_class //STR ex "split_and_tokenize_by_nltk_sentences"
        #     splitter_init //JSON, arguments for the splitter's init
        #     splitter_arguments //JSON, arguments for the splitter's call
        #     chunk_size //INT, ex 1024 if the base chunk are 1024tok long
        #     wants_from_db //JSON, a list of required databases from the SUMBDAT to process the data, example, a text pretokenization strategy would just require "data" that contains raw text, thus ["data"]
        #     returns //JSON ex {"tokens","chunked_tokens"}
        #     required_databases //JSON ex {"tokens","chunks"}
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
        # The table that maps each pretokenization strategy to a submdat
        #
        #     strategy_name //linked to strategy
        #     submdat_id //linked to submdat
        #     finalized_at //DATETIME
        #     total_tokens //LONGINT
        #     max_tokens_per_doc //LONGINT
        #     total_chunks //LONGINT
        #     max_chunks_per_doc //LONGINT
        #     processed_docs //LONGINT
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
        # * In case of chunked tokens, it is useful to precompute the length of iteration in case a multiplier of base_chunk_size is set. These data are used when entire subdatasets are selected.
        #     strategy_name //linked to strategy
        #     submdat_id //linked to submdat
        #     multiplier //SHORT
        #     precomputed_length //LONGINT
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
        # The table that maps each pretokenization strategy to a submdat through one or more db
        #     strategy_name //linked to strategy
        #     submdat_id //linked to submdat
        #     id //PROGRESSIVE ID
        #     is_complete //BOOL
        #     datatype //str ex. 'uint16', 'uint32, 'float', 'tensor'...
        #     is_tokens //BOOL (mututally exclusive)
        #     is_chunks //BOOL (mututally exclusive)
        #     is_extra //JSON or NULL   (mututally exclusive)
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
        # * In case of chunked tokens, it is useful to precompute the length of iteration in case a multiplier of base_chunk_size is set. These data are used when a partial submdat is included in a view.
        #     strategy_name //linked to strategy
        #     view_id //linked to view
        #     submdat_id //linked to submdat
        #     multiplier //SHORT
        #     precomputed_length //LONGINT
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

        # Tabella "VIEW"
        # * In a Mdat, it is possible to build a "view" over a dataset, for example choosing only some example to build a smaller set for ablation studies
        #     view_name //STR (KEY)
        #     shuffle_struct_format //STRING
        cur.execute("""
        CREATE TABLE IF NOT EXISTS view (
            view_name TEXT PRIMARY KEY,
            shuffle_struct_format TEXT
        );
        """)

        # Tabella "VIEW-SUMBDAT"
        # * For each view-submdat couple, how it is considered in the view
        #     view_name KEY
        #     sumbdat_id KEY
        #     is_skipped BOOL (mututally exclusive) //in case the submdat is skipped in this view
        #     is_preserved BOOL (mututally exclusive) //in case the submdat is entirely taken
        #     is_partial BOOL (mututally exclusive) //in case only a part of the submdat is taken
        #     document_number //NULL if skipped or preserved, or the number of selected documents
        #     bytes_criteria //LONGINT if the criteria to choose the view was data size, that number is shown here for reference
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
        # Table "PRETOK_STRATEGY-VIEW-DISTRIBUTED-LENGTH"
        # * Precomputed chunk counts for distributed training configurations
        #     strategy_name //linked to strategy
        #     view_name //linked to view
        #     num_workers //INT number of workers this was precomputed for
        #     worker_id //INT the worker rank (0 to num_workers-1)
        #     chunk_count //LONGINT number of chunks this worker processes
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


    def get_manifest(self, submdat=None, strategy=None, view=None):
        """Return a single dict from manifest OR a row from submdat/pretok_strategy/view.
        Maintains original call signatures for compatibility.
        """
        if not submdat and not strategy and not view:
            return self._fetchone("SELECT * FROM manifest LIMIT 1")

        if submdat:
            where = ("ds_map_id=?" if isinstance(submdat, int) else "name=?", (submdat,))
            return self._fetchone(f"SELECT * FROM submdat WHERE {where[0]}", where[1])
        if strategy:
            return self._fetchone("SELECT * FROM pretok_strategy WHERE strategy_name=?", (strategy,))
        return self._fetchone("SELECT * FROM view WHERE view_name=?", (view,))

    def create_manifest(self, dsmap_bits, modalities):
        """Keep original signature: inserts into manifest."""
        return self._insert("manifest", {"dsmap_bits": dsmap_bits, "modalities": json.dumps(modalities)})

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

    def update_manifest(self, data, submdat=None, strategy=None, view=None):
        data = dict(data)
        c = self.connect().cursor()
        cols_info = [x[1] for x in c.execute("PRAGMA table_info('manifest')").fetchall()]
        if 'finalized_at' in cols_info:
            data['finalized_at'] = datetime.utcnow().isoformat()

        if submdat:
            table = 'submdat'
            where_clause = ('ds_map_id=?' if isinstance(submdat, int) else 'name=?')
            where_params = (submdat,)
        elif strategy:
            table = 'pretok_strategy'
            where_clause = 'strategy_name=?'
            where_params = (strategy,)
        elif view:
            table = 'view'
            where_clause = 'view_name=?'
            where_params = (view,)
        else:
            table = 'manifest'
            where_clause = None
            where_params = ()

        self._update_table(table, data, where_clause, where_params)

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

    def add_database(self, _type, compression_level, map_size, disk_size, extra_data):
        return self._insert('database_ref', {
            'type': _type,
            'compression_level': compression_level,
            'map_size': map_size,
            'disk_size': disk_size,
            'extra_data': extra_data
        })

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



    def link_submdat_database(self, submdat_id, database_id, raw_data_bytes, is_data, is_meta, is_extra):
        return self._insert('submdat_database', {
            'submdat_id': submdat_id,
            'database_id': database_id,
            'raw_data_bytes': raw_data_bytes,
            'is_data': int(bool(is_data)),
            'is_meta': int(bool(is_meta)),
            'is_extra': int(bool(is_extra))
        })

    def add_submdat(self, submdat_name):
        return self._insert('submdat', {'name': submdat_name})

    def add_view(self, view_name):
        c = self.connect().cursor()
        c.execute("INSERT INTO view(view_name,shuffle_struct_format) VALUES (?,?)", (view_name, None))
        self.connect().commit()
        return c.lastrowid

    def remove_submdat(self, submdat_id):
        c = self.connect().cursor()
        c.execute("DELETE FROM submdat WHERE ds_map_id=?", (submdat_id,))
        self.connect().commit()
    def get_storage_db_info(self,database_id):
        q="SELECT * FROM database_ref WHERE database_ref.id = ?"
        return self._fetchone(q,(int(database_id),))


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
        if database_type=='data':
            for r in result:
                if r['is_data']==1:
                    return r
        elif database_type=='meta':
            for r in result:
                if r['is_meta']==1:
                    return r
        return result[0] if database_id is not None and result else result

    def update_submdat_database(self, submdat_id, database_id, data):
        dr_allowed = {'type','compression_level','map_size','disk_size','extra_data'}
        sd_allowed = {'raw_data_bytes','is_data','is_meta','is_extra','doc_count'}
        dr_fields = {k: v for k, v in data.items() if k in dr_allowed}
        sd_fields = {k: v for k, v in data.items() if k in sd_allowed}
        if dr_fields:
            self._update_table('database_ref', dr_fields, 'id=?', (database_id,))
        if sd_fields:
            self._update_table('submdat_database', sd_fields, 'submdat_id=? AND database_id=?', (submdat_id, database_id))

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

    def get_strategy_dbs(self, strategy_name, submdat_id, db_id=None):
        q = """SELECT psd.*, dr.* FROM pretok_strategy_db psd
               JOIN database_ref dr ON psd.db_id = dr.id
               WHERE psd.strategy_name=? AND psd.submdat_id=?"""
        params = [strategy_name, submdat_id]
        if db_id is not None:
            q += " AND psd.id=?"
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


    def add_strategy_db(self, database_id, strategy_name, submdat_id, datatype, is_tokens,is_chunks,is_extra,is_complete=1):
        c = self.connect().cursor()
        c.execute("INSERT INTO pretok_strategy_db(db_id,strategy_name,submdat_id,datatype,is_tokens,is_chunks,is_extra,is_complete) VALUES(?,?,?,?,?,?,?,?)", (database_id,strategy_name, submdat_id,datatype,is_tokens,is_chunks,is_extra,is_complete))
        db_id = c.lastrowid
        self.connect().commit()
        return db_id

    def remove_strategy_db(self, db_id):
        c = self.connect().cursor()
        c.execute("DELETE FROM pretok_strategy_db WHERE id=?", (db_id,))
        self.connect().commit()

    def set_strategy_view_precomp(self, strategy_name, view_name, submdat_id, multiplier, precomputed_length):
        c = self.connect().cursor()
        c.execute("""INSERT OR REPLACE INTO pretok_strategy_view_submdat_precomputed_length
                     (strategy_name,view_name,submdat_id,multiplier,precomputed_length) VALUES(?,?,?,?,?)""",
                  (strategy_name, view_name, submdat_id, multiplier, precomputed_length))
        self.connect().commit()

    def get_strategy_precomp(self, strategy_name, view_name=None, submdat_id=None, multiplier=None):
        if view_name:
            table='pretok_strategy_view_submdat_precomputed_length'
        else:
            table='pretok_strategy_submdat_precomputed_length'
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

    def list_views(self):
        rows = self._fetchall("SELECT view_name FROM view")
        return [row['view_name'] for row in rows]
    def list_strategies(self):
        rows = self._fetchall("SELECT strategy_name FROM pretok_strategy")
        return [row['strategy_name'] for row in rows]

    def get_submdat_info(self, submdat_id):
        submdat_id=int(submdat_id)
        r = self._fetchone(
            """
            SELECT s.ds_map_id AS submdat_id, s.document_number,
                   sd.raw_data_bytes
            FROM submdat AS s
            LEFT JOIN submdat_database AS sd ON sd.submdat_id = s.ds_map_id
            WHERE s.ds_map_id = ? AND sd.is_data = ?
            """,
            (submdat_id,1,)
        )
        return r



    def get_view_submdat(self, view_name, submdat_id):
        return self._fetchone(
            "SELECT view_name, submdat_id, is_skipped, is_preserved, is_partial, document_number, bytes_criteria FROM view_submdat WHERE view_name=? AND submdat_id=?",
            (view_name, submdat_id)
        )

    def update_view_submdat(self, view_name, submdat_id, data):
        allowed = {'is_skipped','is_preserved','is_partial','document_number','bytes_criteria'}
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
