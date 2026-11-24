# matformer/gui/serverpage.py
"""Server: what the far side is doing and what lives inside it — the loaded model, the active task,
memory footprint, the chosen attention implementation and mounted plugins (`model.internals`, the
quick check "which kernel am I actually running?"). Commands have their own page, connections and
the run catalog are local matters: this page is the machine's state and nothing else. Job state
lives in the dock."""
from tkinter import ttk

from matformer.gui.widgets import Page, TableList


class ServerPage(Page):
    """Live server state."""

    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Server", "")
        row = ttk.Frame(self.body)
        row.pack(fill="x")
        ttk.Button(row, text="Refresh", style="Accent.TButton", command=self.refresh).pack(side="left")
        ttk.Button(row, text="Unload model", command=self.unload).pack(side="left", padx=6)
        ttk.Button(row, text="Free GPU memory", command=self.free).pack(side="left")
        ttk.Button(row, text="Commands", style="Ghost.TButton",
                   command=lambda: app.select_page("commands")).pack(side="right")
        ttk.Button(row, text="Connections & catalog", style="Ghost.TButton",
                   command=lambda: app.select_page("local")).pack(side="right", padx=4)

        pan = ttk.PanedWindow(self.body, orient="horizontal")
        pan.pack(fill="both", expand=True, pady=6)
        left = ttk.Frame(pan)
        pan.add(left, weight=2)
        ttk.Label(left, text="Loaded", style="Head.TLabel").pack(fill="x")
        self.info = TableList(app, left, height=6)
        self.info.headings(["property", "value"])
        self.info.fill([], "server offline")
        self.info.pack(fill="x")
        ttk.Label(left, text="Memory", style="Head.TLabel").pack(fill="x", pady=(8, 0))
        self.mem = TableList(app, left, height=5)
        self.mem.headings(["pool", "value"])
        self.mem.fill([])
        self.mem.pack(fill="x")

        right = ttk.Frame(pan)
        pan.add(right, weight=3)
        ttk.Label(right, text="Model internals", style="Head.TLabel").pack(fill="x")
        self.internals = TableList(app, right, height=12)
        self.internals.headings(["what", "name", "detail"])
        self.internals.fill([], "no model loaded")
        self.internals.pack(fill="both", expand=True)

    def activated(self):
        self.refresh()

    def server_changed(self):
        self.refresh()

    def task_state(self, state):
        self.refresh()

    def refresh(self):
        srv = self.app.server
        srv.call("task.state", None, self._state)
        srv.call("sys.mem", None, self._mem)
        srv.call("model.internals", None, self._internals)

    def _state(self, st):
        if not st or isinstance(st, tuple):
            return
        rows = [("model", st.get("source") or "none"),
                ("active task", st.get("task")), ("device", st.get("device")),
                ("configured tasks", ", ".join(st.get("tasks") or []))]
        self.info.fill([{"property": k, "value": v} for k, v in rows if v], "server offline")
        self.set_subtitle(f"{self.app.server_var.get()}   ·   "
                          f"{st.get('source') or 'no model loaded'}")

    def _mem(self, mem):
        if not mem or isinstance(mem, tuple):
            return
        self.mem.fill([{"pool": k, "value": f"{v / 2**20:,.0f} MiB"}
                       for k, v in mem.items() if isinstance(v, (int, float))])

    def _internals(self, table):
        if not table or isinstance(table, tuple):
            return
        rows = table.get("rows") if isinstance(table, dict) else table
        self.internals.fill(rows or [], "no model loaded")

    def unload(self):
        self.app.server.call("task.unload", None, lambda r: (self.app.toast(str(r)), self.refresh()))

    def free(self):
        self.app.server.call("sys.gc", None, lambda r: (self.app.toast(str(r)), self._refresh_mem()))

    def _refresh_mem(self):
        self.app.server.call("sys.mem", None, self._mem)
