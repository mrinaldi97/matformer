"""Jobs: the queue of long work, separate from the page that creates it.

Training, MDAT work and annotation share one queue. This page follows it across one server or every
connected server, tails the selected job and keeps the abort and log actions one click away.
"""
import json
import tkinter as tk
from tkinter import filedialog, ttk

from matformer.gui import exports
from matformer.gui.widgets import Console, Page, TableList, fit_dialog, popup


class JobsPage(Page):
    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Jobs", "no jobs on this server")
        self.rows = []
        self._selected = None
        self._tail_job = None
        self._tail_server = None
        self._timer = None
        self._want_job = None
        self._build_actions()
        bar = ttk.Frame(self.body)
        bar.pack(fill="x", pady=(0, 4))
        self.all_servers = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="All servers", variable=self.all_servers,
                        command=self.refresh).pack(side="left")
        self.where = ttk.Label(bar, text="", style="Faint.TLabel")
        self.where.pack(side="left", padx=10)
        pan = ttk.PanedWindow(self.body, orient="horizontal")
        pan.pack(fill="both", expand=True)
        left = ttk.Frame(pan)
        pan.add(left, weight=2)
        self.table = TableList(app, left, status_col="status", height=10)
        self.table.headings(["id", "server", "kind", "name", "project", "status", "step", "progress"])
        self.table.tree.bind("<<TreeviewSelect>>", lambda e: self.tail())
        self.table.tree.bind("<Double-1>", lambda e: self._open_run())
        self.table.tree.bind("<Button-3>", self._menu)
        self.table.fill([], "no jobs on any server" if self.all_servers.get() else "no jobs on this server")
        self.table.pack(fill="both", expand=True)
        right = ttk.Frame(pan)
        pan.add(right, weight=3)
        head = ttk.Frame(right)
        head.pack(fill="x")
        ttk.Label(head, text="Log", style="Head.TLabel").pack(side="left")
        ttk.Button(head, text="Clear", style="Ghost.TButton", width=6,
                   command=lambda: self.log.clear()).pack(side="right")
        self.log = Console(app, right)
        self.log.pack(fill="both", expand=True, pady=(2, 0))
        self.log.put("Select a job to follow its log.\n", "meta")

    def _build_actions(self):
        ttk.Button(self.actions, text="Refresh", style="Accent.TButton",
                   command=self.refresh).pack(side="left", padx=3)
        self.abort_btn = ttk.Button(self.actions, text="Abort", style="Danger.TButton",
                                    command=self.abort, state="disabled")
        self.abort_btn.pack(side="left", padx=3)
        self.export_btn = ttk.Button(self.actions, text="Export log…", style="Ghost.TButton",
                                     command=self.export_log, state="disabled")
        self.export_btn.pack(side="left")
        control = ttk.Menubutton(self.actions, text="Control", style="Ghost.TButton",
                                 state="disabled")
        menu = tk.Menu(control, tearoff=0)
        for label, cmd in (("Save checkpoint", "save"), ("Pause", "pause"),
                           ("Resume", "resume"), ("Stop after checkpoint", "stop")):
            menu.add_command(label=label, command=lambda c=cmd: self.command(c))
        menu.add_separator()
        menu.add_command(label="Set a field…", command=self.set_field)
        menu.add_command(label="Run a task now…", command=self.run_task)
        menu.add_command(label="Heartbeat", command=self.heartbeat)
        control["menu"] = menu
        control.pack(side="left", padx=3)
        self.control = control

    def _actions_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for widget in (self.abort_btn, self.export_btn, self.control):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    @property
    def server(self):
        return self.app.server

    def activated(self):
        self.refresh()
        self.server.call("jobs.watch")

    def server_changed(self):
        self._tail_job = self._tail_server = None
        self._selected = None
        self._actions_enabled(False)
        self.log.clear()
        self.log.put("Select a job to follow its log.\n", "meta")
        self.refresh()

    def refresh(self):
        if not self.all_servers.get():
            name = self.app.server_var.get()
            self.app.server_of(name).call("jobs.list", None,
                                          lambda rows, n=name: self._filled(rows, n))
            self.where.configure(text=f"server: {name}")
            return
        self.where.configure(text="reading every server…")
        collected = []

        def one(rows, name):
            if rows and not isinstance(rows, tuple):
                collected.extend({**row, "server": name} for row in rows)
            self._filled(collected, None)
        for name, srv in self.app.servers.items():
            if srv.up:
                srv.call("jobs.list", None, lambda rows, n=name: one(rows, n))

    def _filled(self, rows, server):
        if not rows or isinstance(rows, tuple):
            rows = []
        self.rows = list(rows)
        if server:
            for row in self.rows:
                row.setdefault("server", server)
        self.table.fill(self.rows, "no jobs on any server" if self.all_servers.get()
                        else "no jobs on this server")
        self.set_subtitle(f"{len(self.rows)} job" + ("s" if len(self.rows) != 1 else ""))
        if self._want_job:
            wanted, self._want_job = self._want_job, None
            for row in self.rows:
                if row.get("id") == wanted:
                    self._select(row)
                    break

    def _select(self, row):
        self._selected = row
        if row:
            self._tail_job, self._tail_server = row.get("id"), row.get("server") or self.app.server_var.get()
            self.tail()

    def focus_job(self, job_id, server=None):
        self._want_job = job_id
        if server and server in self.app.servers:
            self.app.server_var.set(server)
            self.app._server_changed()
        self.app.select_page("jobs")
        self.refresh()

    def _current(self):
        row = self.table.selected()
        if row:
            self._selected = row
        return self._selected

    def tail(self):
        row = self._current()
        if not row:
            self._actions_enabled(False)
            if self._tail_job is None:
                self.log.clear()
                self.log.put("Select a job to follow its log.\n", "meta")
            return
        self._actions_enabled(True)
        server = row.get("server") or self.app.server_var.get()
        job_id = row.get("id")
        self._tail_job, self._tail_server = job_id, server

        def got(chunk):
            if chunk and not isinstance(chunk, tuple):
                self.log.clear()
                text = chunk.get("text", "")[-120000:]
                if text and not text.endswith("\n"):
                    text += "\n"
                self.log.put(text or "(no log yet for this job)\n",
                             None if text else "meta")
        self.app.server_of(server).call("jobs.log", {"id": job_id}, got)
        if self._timer:
            self.after_cancel(self._timer)
        self._timer = self.after(1200, self.tail)

    def on_event(self, server, name, data):
        if name != "job":
            return
        if not self.all_servers.get() and server != self.app.server_var.get():
            return
        row = {**data, "server": server}
        self.table.upsert("id", row)
        for index, old in enumerate(self.rows):
            if old.get("id") == row.get("id"):
                self.rows[index] = row
                break
        else:
            self.rows.append(row)
        if self._tail_job == row.get("id") and row.get("status") not in ("cancelled",):
            self.tail()

    def _menu(self, event):
        row = self.table.selected()
        if not row:
            return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Abort", command=self.abort)
        menu.add_command(label="Export log…", command=self.export_log)
        menu.add_command(label="Open in Plots", command=self._open_run)
        menu.add_command(label="Copy id", command=lambda: self.app.copy(row.get("id")))
        popup(menu, event.x_root, event.y_root)

    def command(self, cmd):
        row = self._current()
        if not row:
            return self.app.toast("select a job first", error=True)
        self.app.server_of(row.get("server") or self.app.server_var.get()).call(
            "train.command", {"id": row.get("id"), "cmd": cmd})
        self.app.toast(f"{cmd}: {row.get('id')}")

    def heartbeat(self):
        row = self._current()
        if not row:
            return self.app.toast("select a job first", error=True)
        self.app.server_of(row.get("server") or self.app.server_var.get()).call(
            "train.heartbeat", {"id": row.get("id")}, self._show_heartbeat)

    def _show_heartbeat(self, answer):
        import json
        if not answer or isinstance(answer, tuple):
            return
        self.log.put("heartbeat\n" + json.dumps(answer, indent=1, default=str) + "\n", "meta")

    def set_field(self):
        row = self._current()
        if not row:
            return self.app.toast("select a job first", error=True)
        dlg = tk.Toplevel(self)
        dlg.title("Set a live training field")
        dlg.transient(self.app)
        form = ttk.Frame(dlg); form.pack(fill="x", padx=12, pady=12)
        ttk.Label(form, text="Field:").grid(row=0, column=0, sticky="w")
        field = ttk.Combobox(form, values=["lr", "weight_decay", "max_steps",
                                           "save_every_n_steps", "gradient_clip_val"],
                             state="readonly")
        field.set("lr"); field.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(form, text="Value:").grid(row=1, column=0, sticky="w")
        value = ttk.Entry(form); value.grid(row=1, column=1, sticky="ew", padx=4)
        form.columnconfigure(1, weight=1)

        def apply():
            self.app.server_of(row.get("server") or self.app.server_var.get()).call(
                "train.set", {"id": row.get("id"), "field": field.get(), "value": value.get()})
            dlg.destroy()
        ttk.Button(dlg, text="Set", style="Accent.TButton", command=apply).pack(
            anchor="e", padx=12, pady=(0, 10))
        fit_dialog(dlg, 380, 0)
        dlg.bind("<Escape>", lambda e: dlg.destroy())

    def run_task(self):
        row = self._current()
        if not row:
            return self.app.toast("select a job first", error=True)
        dlg = tk.Toplevel(self)
        dlg.title("Run a task inside the training")
        dlg.transient(self.app)
        form = ttk.Frame(dlg); form.pack(fill="x", padx=12, pady=12)
        ttk.Label(form, text="Task:").grid(row=0, column=0, sticky="w")
        task = ttk.Combobox(form, state="readonly")
        task.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(form, text="Name:").grid(row=1, column=0, sticky="w")
        name = ttk.Entry(form); name.grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Label(form, text="Fields (JSON):").grid(row=2, column=0, sticky="nw")
        fields = tk.Text(form, height=5, font=("TkFixedFont", 9))
        fields.insert("1.0", "{}"); fields.grid(row=2, column=1, sticky="ew", padx=4)
        form.columnconfigure(1, weight=1)
        self.app.server_of(row.get("server") or self.app.server_var.get()).call(
            "task.types", None,
            lambda types: task.configure(values=types or []) or task.set((types or ["evaluation"])[0]))

        def apply():
            try:
                payload = json.loads(fields.get("1.0", "end-1c") or "{}")
            except Exception as problem:
                return self.app.toast(f"fields must be JSON: {problem}", error=True)
            self.app.server_of(row.get("server") or self.app.server_var.get()).call(
                "train.task", {"id": row.get("id"), "type": task.get(),
                               "name": name.get().strip() or None, "fields": payload})
            dlg.destroy()
        ttk.Button(dlg, text="Run task", style="Accent.TButton", command=apply).pack(
            anchor="e", padx=12, pady=(0, 10))
        fit_dialog(dlg, 460, 0)
        dlg.bind("<Escape>", lambda e: dlg.destroy())

    def abort(self):
        row = self._current()
        if not row:
            return self.app.toast("select a job first", error=True)
        self.app.server_of(row.get("server") or self.app.server_var.get()).call(
            "train.abort", {"id": row.get("id")})

    def export_log(self):
        row = self._current()
        if not row:
            return self.app.toast("select a job first", error=True)
        path = exports.ask("Export the job log", f"{row.get('id')}.txt", exports.TEXT_FORMATS)
        if not path:
            return

        def got(chunk):
            if chunk and not isinstance(chunk, tuple):
                exports.write_text(path, chunk.get("text", ""), f"Job {row.get('id')} log")
                self.app.toast(f"log exported: {path}")
        self.app.server_of(row.get("server") or self.app.server_var.get()).call(
            "jobs.log", {"id": row.get("id")}, got)

    def _open_run(self):
        row = self._current()
        if row:
            self.app.show_run(row.get("id"))
