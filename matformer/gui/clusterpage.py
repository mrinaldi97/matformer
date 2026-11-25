# matformer/gui/clusterpage.py
"""The cluster as the submission host sees it: queue, jobs, nodes, GPUs, partitions, environments —
each one a table the server's SLURM interpreter produced, so nothing here knows what `sbatch` is.
Right-click a job for the actions that apply to it (checkpoint, stop, requeue, cancel, log, the
address of the node running it); right-click a node to copy its name. A job that changes state on
the cluster pushes a `slurm_job` event, and the Jobs tab follows without a refresh."""
import tkinter as tk
from tkinter import ttk

from matformer.gui.widgets import Console, Page, TableList, popup

VIEWS = [("jobs", "Jobs", ["job", "slurm", "state", "partition", "nodes", "gpus", "step", "loss",
                           "saved", "left", "venv", "host"]),
         ("queue", "Queue", ["id", "user", "name", "partition", "state", "nodes", "gpus", "limit",
                             "left", "nodelist", "reason"]),
         ("nodes", "Nodes", ["node", "state", "arch", "gpu", "gpus_free", "cpus_free", "mem_mb",
                             "partitions", "reason"]),
         ("gres", "GPUs", ["gpu_model", "arch", "nodes", "nodes_idle", "gpus", "gpus_free",
                           "gpus_per_node", "partitions"]),
         ("shapes", "Node sets", ["arch", "gpu_model", "gpus_per_node", "usable", "total",
                                 "partitions", "free_nodes"]),
         ("partitions", "Partitions", ["name", "avail", "state", "nodes", "usable", "max_time",
                                       "gres", "gpus_free", "exclusive", "hidden"]),
         ("venvs", "Environments", ["name", "arch", "python", "torch", "flash_attn", "path"])]
STATUS = {"nodes": "state", "gres": "avail", "partitions": "avail", "queue": "state",
          "jobs": "state"}


class ClusterPage(Page):
    """Read-only dashboards over the cluster, with the actions a job needs."""

    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Cluster", "")
        bar = ttk.Frame(self.actions)
        bar.pack(side="right")
        ttk.Button(bar, text="Doctor", width=7, style="Ghost.TButton",
                   command=self.doctor).pack(side="left", padx=3)
        ttk.Button(bar, text="Refresh", width=7, style="Ghost.TButton",
                   command=self.refresh).pack(side="left")
        split = ttk.PanedWindow(self.body, orient="vertical")
        split.pack(fill="both", expand=True)
        top = ttk.Frame(split)
        split.add(top, weight=3)
        self.tabs = ttk.Notebook(top)
        self.tabs.pack(fill="both", expand=True)
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self._load_view(self._tab_key()))
        self.tables = {}
        for key, title, columns in VIEWS:
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=title)
            table = TableList(app, frame, status_col=STATUS.get(key), on_menu=self._menu(key),
                              on_double=self._double(key), height=16,
                              on_select=self._picked if key in ("jobs", "queue") else None)
            table.headings(columns)
            table.fill([], f"no {title.lower()} to show")
            table.pack(fill="both", expand=True)
            self.tables[key] = table
        bottom = ttk.Frame(split)
        split.add(bottom, weight=2)
        inside = ttk.PanedWindow(bottom, orient="horizontal")
        inside.pack(fill="both", expand=True)
        self.detail = TableList(app, inside, height=5)
        self.detail.headings(["property", "value"])
        self.detail.fill([], "select a job for its detail")
        inside.add(self.detail, weight=1)
        self.log = Console(app, inside)
        self.log.put("Select a job to read its detail; right-click it for log, checkpoint, "
                     "requeue or cancel.\n", "meta")
        inside.add(self.log, weight=2)
        self.view = "jobs"

    # ── transport ────────────────────────────────────────────────────────────────
    def call(self, cmd, args=None, cb=None):
        self.app.server.call("slurm.call", {"cmd": cmd, "args": args or {}}, cb)

    def activated(self):
        self.app.server.call("slurm.watch")
        self.refresh()

    def server_changed(self):
        self.refresh()

    def _tab_key(self):
        try:
            return VIEWS[self.tabs.index(self.tabs.select())][0]
        except (ValueError, tk.TclError):
            return "jobs"

    def _load_view(self, key):
        self.view = key
        self.tables[key].fill([], "loading…")
        self.call(key, None, self._filled(key))

    def refresh(self):
        self._load_view(self._tab_key())

    def _filled(self, key):
        def got(payload):
            table = self.tables[key]
            if not payload or isinstance(payload, tuple):
                return table.fill([], f"{key}: server did not answer")
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            table.fill(rows, f"no {key} on this cluster")
            if key == self.view:
                self.set_subtitle(f"{len(rows)} {key}" + ("s" if len(rows) != 1 and key != "jobs" else ""))
        return got

    def on_event(self, server, name, data):
        if name == "slurm_job" and server == self.app.server_var.get() and self.view == "jobs":
            self.tables["jobs"].upsert("job", data)

    # ── per-view menus ───────────────────────────────────────────────────────────
    def _menu(self, key):
        def build(menu, row):
            if not row:
                return
            if key in ("jobs", "queue"):
                job = row.get("job") or row.get("matformer_id") or row.get("id")
                menu.add_command(label="Follow log", command=lambda: self._tail(job))
                menu.add_command(label="Detail", command=lambda: self._info(job))
                menu.add_separator()
                for label, cmd in (("Checkpoint now", "save"), ("Stop after checkpoint", "stop"),
                                   ("Requeue", "requeue"), ("Cancel", "cancel"),
                                   ("Boost priority", "prioritize")):
                    menu.add_command(label=label, command=lambda c=cmd, j=job: self._act(c, j))
                menu.add_separator()
                menu.add_command(label="Node address", command=lambda: self._attach(job))
            elif key == "nodes":
                menu.add_command(label="Copy node", command=lambda: self.app.copy(row.get("node", "")))
                menu.add_command(label="Exclude from next job",
                                 command=lambda: self.app.copy("exclude=" + str(row.get("node", ""))))
            else:
                menu.add_command(label="Copy row", command=lambda: self.app.copy(
                    "\n".join(f"{k}: {v}" for k, v in row.items())))
        return build

    def _double(self, key):
        def open_row(row):
            if row and key in ("jobs", "queue"):
                self._tail(row.get("job") or row.get("matformer_id") or row.get("id"))
        return open_row

    def _picked(self, row):
        """Selecting a row is enough to see what it is: no menu round-trip for the common case."""
        if not row:
            self.detail.fill([], "select a job for its detail")
            return
        job = row.get("job") or row.get("matformer_id") or row.get("id")
        if job:
            self._info(job)

    def _act(self, cmd, job):
        self.call(cmd, {"job": job}, lambda answer: self._acted(cmd, job, answer))

    def _acted(self, cmd, job, answer):
        if isinstance(answer, tuple):
            return
        self.log.put(f"{cmd} {job}: {(answer or {}).get('msg', 'done')}\n")
        self.call("status", {"job": job}, self._filled("jobs"))

    def _tail(self, job):
        def got(answer):
            if answer and not isinstance(answer, tuple):
                self.log.clear()
                text = answer if isinstance(answer, str) else str(answer)
                self.log.put(text if text.endswith("\n") else text + "\n")
        self.call("tail", {"job": job, "lines": 300}, got)

    def _info(self, job):
        def got(answer):
            if not answer or isinstance(answer, tuple):
                return
            rows = [{"property": key, "value": _text(value)} for key, value in
                    sorted((answer.get("entry") or {}).items())]
            rows += [{"property": "heartbeat", "value": _text(answer.get("heartbeat"))},
                     {"property": "checkpoint", "value": _text(answer.get("saved"))},
                     {"property": "control", "value": _text(answer.get("control"))}]
            self.detail.fill(rows, "no detail")
        self.call("info", {"job": job}, got)

    def _attach(self, job):
        def got(answer):
            if not answer or isinstance(answer, tuple):
                return
            self.log.put(f"host {answer.get('host')}\n{answer.get('tunnel')}\n")
            self.app.toast(f"node: {answer.get('host')}")
        self.call("attach", {"job": job}, got)

    def doctor(self):
        def got(payload):
            self.tabs.select(self.tabs.tab(self.tabs.index("end"), "value"))
            self.log.clear()
            if not payload or isinstance(payload, tuple):
                return self.log.put("doctor: the server did not answer\n")
            rows = payload.get("rows", [])
            self.log.put("nothing to report\n" if not rows
                         else "\n".join(f"[{r['severity']}] {r.get('job', '')} {r['problem']}\n"
                                        f"    → {r['fix']}" for r in rows) + "\n")
        self.call("doctor", {}, got)


def _text(value):
    if value in (None, "", {}, []):
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return ", ".join(f"{k}={_text(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(one) for one in value)
    return str(value)
