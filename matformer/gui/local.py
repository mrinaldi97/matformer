# matformer/gui/local.py
"""This machine's side of the suite: which servers exist and whether they are dialed in, the local
run catalog (what `~/.matformer/catalog.db` remembered across every server, offline runs included),
and the console switch. None of it is a server operation — it is the client's own state, so it
lives here and not in the Server page."""
import csv
import tkinter as tk
from tkinter import filedialog, ttk

from matformer.gui.widgets import Page, TableList, popup


class LocalPage(Page):
    """Connections + local run catalog."""

    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Local", "")
        pan = ttk.PanedWindow(self.body, orient="horizontal")
        pan.pack(fill="both", expand=True)

        left = ttk.Frame(pan)
        pan.add(left, weight=2)
        head = ttk.Frame(left)
        head.pack(fill="x")
        ttk.Label(head, text="Servers", style="Head.TLabel").pack(side="left")
        ttk.Button(head, text="Preferences…", style="Ghost.TButton",
                   command=app.do_settings).pack(side="right")
        self.servers = TableList(app, left, on_double=self._toggle, height=6)
        self.servers.headings(["name", "url", "state"])
        self.servers.pack(fill="x", pady=4)
        self.servers.tree.bind("<Button-3>", self._server_menu)
        row = ttk.Frame(left)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Connect", command=lambda: self._pick_server(self.connect)).pack(side="left")
        ttk.Button(row, text="Disconnect", command=lambda: self._pick_server(self.disconnect)).pack(side="left", padx=4)
        ttk.Button(row, text="Make current", style="Ghost.TButton",
                   command=lambda: self._pick_server(self.make_current)).pack(side="left")
        box = ttk.LabelFrame(left, text=" Console ")
        box.pack(fill="x", pady=8)
        self.record = ttk.Checkbutton(box, text="record every command and answer",
                                      command=self._toggle_record)
        self.record.pack(anchor="w", padx=6, pady=2)
        ttk.Button(box, text="Open the Console tab", style="Ghost.TButton",
                   command=app.do_console).pack(anchor="w", padx=6, pady=(0, 6))

        right = ttk.Frame(pan)
        pan.add(right, weight=3)
        head = ttk.Frame(right)
        head.pack(fill="x")
        ttk.Label(head, text="Run catalog", style="Head.TLabel").pack(side="left")
        self.count = ttk.Label(head, style="Muted.TLabel")
        self.count.pack(side="left", padx=8)
        ttk.Button(head, text="Sync from servers", command=app.do_sync_catalog).pack(side="right")
        ttk.Button(head, text="Export CSV…", style="Ghost.TButton",
                   command=self._export).pack(side="right", padx=4)
        self.filter = ttk.Entry(right)
        self.filter.pack(fill="x", pady=4)
        self.filter.bind("<KeyRelease>", lambda e: self._show())
        self.catalog = TableList(app, right, on_double=lambda row: row and app.show_run(row["id"]),
                                 height=14)
        self.catalog.headings(["server", "project", "run", "version", "started", "steps", "host"])
        self.catalog.pack(fill="both", expand=True)
        self.catalog.tree.bind("<Button-3>", self._catalog_menu)
        self.all_rows = []

    def activated(self):
        self.refresh()

    def server_changed(self):
        self.refresh()

    def refresh(self):
        self._paint_servers()
        self._load_catalog()
        self.record.state(["selected"] if self.app.traffic.on else ["!selected"])

    # ── connections ────────────────────────────────────────────────────────────
    def _paint_servers(self):
        current = self.app.server_var.get()
        self.servers.fill([{"name": f"▸ {n}" if n == current else n, "url": s.url,
                            "state": "up" if s.up else "down", "server": n}
                           for n, s in self.app.servers.items()], "no server configured")
        for row in self.servers.rows:
            row["name"] = row["name"].lstrip("▸ ")

    def _selected_server(self):
        row = self.servers.selected()
        return self.app.servers[row["server"]] if row else None

    def _pick_server(self, action):
        srv = self._selected_server()
        if srv:
            action(srv)

    def connect(self, srv):
        srv.connect()
        self.app.toast(f"{srv.name}: dialing")

    def disconnect(self, srv):
        srv.disconnect()
        self.app.toast(f"{srv.name}: disconnected")

    def make_current(self, srv):
        self.app.server_var.set(srv.name)
        self.app._server_changed()

    def _toggle(self, row):
        if row:
            self._toggle_server(self.app.servers[row["server"]])

    def _toggle_server(self, srv):
        (self.disconnect if srv.up else self.connect)(srv)
        self.app.after(400, self._paint_servers)

    def _server_menu(self, e):
        row = self.servers.selected()
        if not row:
            return
        srv = self.app.servers[row["server"]]
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Disconnect" if srv.up else "Connect",
                      command=lambda: self._toggle_server(srv))
        m.add_command(label="Make this the current server", command=lambda: self.make_current(srv))
        m.add_command(label="Server page", command=lambda: self.app.select_page("server"))
        m.add_command(label="Copy url", command=lambda: self.app.copy(srv.url))
        popup(m, e.x_root, e.y_root)

    def _toggle_record(self):
        self.app.traffic.set(self.record.instate(["selected"]))

    # ── the local run catalog ──────────────────────────────────────────────────
    def _load_catalog(self):
        from matformer.utils.catalog import Catalog
        self.all_rows = Catalog().all()
        self._show()

    def _show(self):
        t = self.filter.get().lower()
        rows = [r for r in self.all_rows
                if not t or t in " ".join(str(v) for v in r.values()).lower()]
        self.catalog.fill(rows, "empty catalog: use Sync from servers")
        self.count.configure(text=f"{len(self.all_rows)} runs")

    def _catalog_menu(self, e):
        row = self.catalog.selected()
        if not row:
            return
        self.catalog.tree.selection_set(self.catalog.rows.index(row))
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Open in Runs", command=lambda: self.app.show_run(row["id"]))
        m.add_command(label="Copy directory", command=lambda: self.app.copy(row.get("dir") or ""))
        m.add_command(label="Forget this run", command=lambda: self._forget(row))
        popup(m, e.x_root, e.y_root)

    def _forget(self, row):
        from matformer.utils.catalog import Catalog
        Catalog().forget(row["server"], row["id"])
        self._load_catalog()
        self.app.pages["runs"].catalog_reload()

    def _export(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="runs.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if self.all_rows:
                writer.writerow(self.catalog.columns)
                for r in self.all_rows:
                    writer.writerow([r.get(c, "") for c in self.catalog.columns])
        self.app.toast(f"catalog exported to {path}")
