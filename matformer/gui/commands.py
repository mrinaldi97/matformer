# matformer/gui/commands.py
"""The server's own manual, read from the server (`rpc.list`): every command with the first line of
its docstring, filterable, runnable, and next to it the failures — the commands that raised, with
the traceback the status bar could only shorten. Nothing here is hardcoded: a new server command
appears the moment it is registered."""
import json
import tkinter as tk
from tkinter import ttk

from matformer.gui.widgets import Console, Page, TableList, popup


class CommandsPage(Page):
    """Command reference + server-side tracebacks."""

    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Commands", "the server, asked about itself")
        top = ttk.Frame(self.body)
        top.pack(fill="x")
        self.filter = ttk.Entry(top)
        self.filter.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.filter.bind("<KeyRelease>", lambda e: self._show_cmds())
        ttk.Button(top, text="Refresh", style="Ghost.TButton", command=self.refresh).pack(side="left")
        pan = ttk.PanedWindow(self.body, orient="vertical")
        pan.pack(fill="both", expand=True, pady=6)

        holder = ttk.Frame(pan)
        pan.add(holder, weight=3)
        self.cmds = TableList(app, holder, on_double=self._run_one, height=12)
        self.cmds.headings(["method", "doc"])
        self.cmds.fill([], "no server")
        self.cmds.pack(fill="both", expand=True)
        self.all_cmds = []

        bottom = ttk.Frame(pan)
        pan.add(bottom, weight=2)
        head = ttk.Frame(bottom)
        head.pack(fill="x")
        ttk.Label(head, text="Failures", style="Head.TLabel").pack(side="left")
        ttk.Button(head, text="Clear", style="Ghost.TButton", width=6,
                   command=self._clear_failures).pack(side="right")
        self.failures = TableList(app, bottom, on_double=self._show_failure, height=6)
        self.failures.headings(["at", "method", "error"])
        self.failures.fill([], "no failed command")
        self.failures.pack(fill="both", expand=True)
        self.detail = Console(app, bottom)
        self.detail.pack(fill="both", expand=True)
        self.detail.text.configure(height=6)

    def activated(self):
        self.refresh()

    def server_changed(self):
        self.refresh()

    def refresh(self):
        srv = self.app.server
        srv.call("rpc.list", None, self._cmds)
        self.show_failures()

    def _cmds(self, cmds):
        if cmds and not isinstance(cmds, tuple):
            self.all_cmds = [{"method": c["name"], "doc": c["doc"]} for c in cmds]
            self._show_cmds()

    def _show_cmds(self):
        t = self.filter.get().lower()
        rows = [c for c in self.all_cmds if t in c["method"] or t in c["doc"].lower()]
        self.cmds.fill(rows, "no matching command")

    def _run_one(self, row):
        if not row:
            return
        name = row["method"]

        def got(result):
            if isinstance(result, tuple):
                return self.detail.put(f"{name}  {result[1]}\n", "err")
            self.detail.put(f"{name}  {json.dumps(result, default=str)[:4000]}\n")
        self.app.server.call(name, {}, got)

    def show_failures(self):
        self.app.server.call("rpc.failures", None, self._failures)

    def _failures(self, table):
        if not table or isinstance(table, tuple):
            return
        rows = table.get("rows") if isinstance(table, dict) else table
        self.failures.fill(rows or [], "no failed command")

    def _show_failure(self, row):
        if not row:
            return
        index = self.failures.rows.index(row) if row in self.failures.rows else 0

        def got(result):
            if result and not isinstance(result, tuple):
                self.detail.clear()
                self.detail.put(result.get("text", ""))
        self.app.server.call("rpc.failures", {"index": index}, got)

    def _clear_failures(self):
        self.app.server.call("rpc.failures", {"clear": True}, lambda r: self.show_failures())
