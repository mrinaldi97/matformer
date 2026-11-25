# matformer/utils/catalog.py
"""An offline index of run metadata, not the run store.

Each server owns its runs in the SQLite run database (``matformer/utils/run_db.py``): that is the
source of truth, and every metric, parameter and task result is read from it. This catalog is a
small laptop-side cache of the *identities* of runs seen on several servers, so a machine can still
list history while a server is unreachable. It holds no metric and no task data.

Sync is deliberately dumb and conflict-free: each server's rows are replaced wholesale on every
sync (a server is always the source of truth for its own runs, which never migrate), so the merged
view converges. The same table can be pushed back to any server (`catalog.merge`) so each machine
also learns about runs made elsewhere. Remove this module and its three RPC commands if offline
browsing across servers is not wanted: no run is lost, because the run databases stay untouched."""
import sqlite3
import time
from pathlib import Path

DB = Path.home() / ".matformer" / "catalog.db"
SCHEMA = """CREATE TABLE IF NOT EXISTS runs (
    server TEXT, id TEXT, project TEXT, run TEXT, version TEXT, host TEXT,
    started TEXT, dir TEXT, steps INTEGER, total_steps INTEGER, synced REAL,
    PRIMARY KEY (server, id))"""


class Catalog:
    def __init__(self, path=DB):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._con() as c:
            c.execute(SCHEMA)

    def _con(self):
        return sqlite3.connect(self.path)

    def replace_server(self, server, rows):
        """The server owns its rows: wipe and re-insert, so deletions propagate naturally."""
        with self._con() as c:
            c.execute("DELETE FROM runs WHERE server=?", (server,))
            for r in rows:
                c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                          (server, r.get("id"), r.get("project"), r.get("run"), r.get("version"),
                           r.get("host"), r.get("started"), r.get("dir"), r.get("steps"),
                           r.get("total_steps"), time.time()))
        return len(rows)

    def upsert(self, rows):
        """Absorb rows that came from another machine's catalog, each keeping the server it was
        seen on. Nothing is wiped: this is additive knowledge, not a mirror of one server."""
        with self._con() as c:
            for r in rows:
                c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                          (r.get("server") or "?", r.get("id"), r.get("project"), r.get("run"),
                           r.get("version"), r.get("host"), r.get("started"), r.get("dir"),
                           r.get("steps"), r.get("total_steps"), time.time()))
        return len(rows)

    def forget(self, server, run_id):
        """Drop one remembered run (the local view only: the server keeps its own files)."""
        with self._con() as c:
            c.execute("DELETE FROM runs WHERE server=? AND id=?", (server, run_id))

    def all(self):
        with self._con() as c:
            cols = ("server id project run version host started dir steps total_steps synced"
                    .split())
            return [dict(zip(cols, row)) for row in
                    c.execute("SELECT * FROM runs ORDER BY started DESC").fetchall()]

    def as_rows(self):
        """Server-shaped dicts (what runs.list returns), for pushing via catalog.merge."""
        return [{k: v for k, v in r.items() if k in
                 ("id", "project", "run", "version", "host", "started", "dir", "steps",
                  "total_steps")} for r in self.all()]
