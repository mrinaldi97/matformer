# matformer/gui/clusterpage.py
"""The cluster as the submission host sees it: queue, jobs, nodes, GPUs, partitions, environments —
each one a table the server's SLURM interpreter produced, so nothing here knows what `sbatch` is.
Right-click a job for the actions that apply to it (checkpoint, stop, requeue, cancel, log, the
address of the node running it); right-click a node to copy its name. A job that changes state on
the cluster pushes a `slurm_job` event, and the Jobs tab follows without a refresh."""
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
        self.tabs = ttk.Notebook(self.body)
        self.tabs.pack(fill="both", expand=True)
        self.tables = {}
        for key, title, columns in VIEWS:
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=title)
            table = TableList(app, frame, status_col=STATUS.get(key), on_menu=self._menu(key),
                              on_double=self._double(key), height=16)
            table.headings(columns)
            table.fill([], f"no {title.lower()} to show")
            table.pack(fill="both", expand=True)
            self.tables[key] = table
        bottom = ttk.PanedWindow(self.body, orient="vertical")
        bottom.pack(fill="both", expand=True, pady=(6, 0))
        self.detail = TableList(app, bottom, height=5)
        self.detail.headings(["property", "value"])
        self.detail.fill([], "select a job for its detail")
        bottom.add(self.detail, weight=1)
        self.log = Console(app, bottom)
        bottom.add(self.log, weight=2)
        self.view = "jobs"

    # ── transport ────────────────────────────────────────────────────────────────
    def call(self, cmd, args=None, cb=None):
        self.app.server.call("slurm.call", {"cmd": cmd, "args": args or {}}, cb)

    def activated(self):
        self.refresh()

    def server_changed(self):
        self.refresh()

    def refresh(self):
        for key, _, _ in VIEWS:
            self.tables[key].fill([], "loading…")
            self.call(key, None, self._filled(key))

    def _filled(self, key):
        def got(payload):
            table = self.tables[key]
            if not payload or isinstance(payload, tuple):
                return table.fill([], f"{key}: server did not answer")
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            table.fill(rows, f"no {key} on this cluster")
            if key == self.view:
                self.set_subtitle(f"{len(rows)} {key}")
        return got

    def on_event(self, server, name, data):
        if name == "slurm_job" and server == self.app.server_var.get() and self.view == "jobs":
            self.tables["jobs"].upsert("matformer_id", data)

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
                self.log.put(answer if isinstance(answer, str) else str(answer))
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
    return "—" if value in (None, "", {}, []) else (
        value if isinstance(value, str) else ", ".join(f"{k}={v}" for k, v in value.items()))
