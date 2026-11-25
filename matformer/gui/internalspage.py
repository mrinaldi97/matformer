# matformer/gui/internalspage.py
"""Internals: the sockets a plugin can mount on, the plugins mounted now, and the weights they
can edit. Everything is RPC (plugin.*, param.*): the page never imports the model, it asks the
server what it has, so it works the same on a taple or a V100."""
import json
import tkinter as tk
from tkinter import simpledialog, ttk

from matformer.gui.widgets import (Console, Page, TableList, fit_dialog, popup, show_payload,
                                   tip)


class InternalsPage(Page):
    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Internals", "sockets, plugins and weights of the loaded model")
        bar = ttk.Frame(self.body)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="Sockets matching:").pack(side="left")
        self.pattern = ttk.Entry(bar, width=26)
        self.pattern.insert(0, "*")
        self.pattern.pack(side="left", padx=4)
        self.pattern.bind("<Return>", lambda e: self.refresh())
        ttk.Button(bar, text="Refresh", style="Accent.TButton",
                   command=self.refresh).pack(side="left", padx=4)
        ttk.Button(bar, text="Attach plugin…", command=self.attach_dialog).pack(side="left", padx=4)
        self.detach_btn = ttk.Button(bar, text="Detach", state="disabled", command=self.detach)
        self.detach_btn.pack(side="left", padx=4)
        pan = ttk.PanedWindow(self.body, orient="horizontal")
        pan.pack(fill="both", expand=True)
        left = ttk.Frame(pan)
        pan.add(left, weight=3)
        ttk.Label(left, text="Sockets", style="Head.TLabel").pack(fill="x")
        self.sockets = TableList(app, left, height=10, on_double=self._use_socket,
                                 on_select=lambda row: self._socket_tip(row))
        self.sockets.headings(["address", "mounted", "count", "owner"])
        self.sockets.fill([], "load a model first")
        self.sockets.pack(fill="both", expand=True)
        right = ttk.Frame(pan)
        pan.add(right, weight=2)
        head = ttk.Frame(right)
        head.pack(fill="x")
        ttk.Label(head, text="Mounted plugins", style="Head.TLabel").pack(side="left")
        ttk.Label(head, text="  origin=model is part of the config; runtime is an intervention",
                  style="Muted.TLabel").pack(side="left")
        self.plugins = TableList(app, right, height=10, on_select=self._plugin_picked)
        self.plugins.headings(["id", "socket", "kind", "class", "origin"])
        self.plugins.fill([], "no plugin mounted")
        self.plugins.pack(fill="both", expand=True)

        sheet = ttk.Frame(self.body)
        sheet.pack(fill="x", pady=(8, 0))
        ttk.Label(sheet, text="Weights", style="Head.TLabel").pack(side="left")
        ttk.Label(sheet, text="  filter:").pack(side="left")
        self.param_filter = ttk.Entry(sheet, width=26)
        self.param_filter.insert(0, "*")
        self.param_filter.pack(side="left", padx=4)
        self.param_filter.bind("<Return>", lambda e: self.load_params())
        for label, action in (("Edit…", self.edit_param), ("Stats", self.param_stats),
                              ("Directions", self.param_directions),
                              ("Ablate direction", self.param_ablate),
                              ("Freeze", self.param_freeze), ("Restore", self.param_restore)):
            ttk.Button(sheet, text=label, style="Ghost.TButton",
                       command=action).pack(side="left", padx=2)
        self.params = TableList(app, self.body, height=8)
        self.params.headings(["name", "shape", "dtype", "norm", "requires_grad", "edited"])
        self.params.fill([], "load a model first")
        self.params.pack(fill="both", expand=True, pady=4)
        self.note = Console(app, self.body)
        self.note.text.configure(height=3)
        self.note.pack(fill="x")

    # ── transport ───────────────────────────────────────────────────────────────
    def call(self, method, params, cb):
        self.app.server.call(method, params, cb)

    def activated(self):
        self.refresh()

    def server_changed(self):
        self.refresh()

    def refresh(self):
        pattern = self.pattern.get().strip() or "*"
        self.call("plugin.sockets", {"pattern": pattern}, self._sockets)
        self.call("plugin.list", None, self._plugins)
        self.load_params()

    def load_params(self):
        pattern = self.param_filter.get().strip() or "*"
        self.call("param.list", {"pattern": pattern, "limit": 400}, self._params)

    def _sockets(self, payload):
        if not payload or isinstance(payload, tuple):
            self.sockets.fill([], "load a model first")
            return
        rows = payload.get("sockets") or []
        self.sockets.fill(rows, "no socket matches")
        self.set_subtitle(f"{len(rows)} sockets  ·  {self.app.server_var.get()}")

    def _plugins(self, payload):
        if not payload or isinstance(payload, tuple):
            self.plugins.fill([], "no plugin mounted")
            return
        self.plugins.fill(payload.get("plugins") or [], "no plugin mounted")
        self.detach_btn.configure(state="disabled")

    def _params(self, payload):
        if not payload or isinstance(payload, tuple):
            self.params.fill([], "load a model first")
            return
        rows = payload.get("parameters") or []
        self.params.fill([{"name": r.get("name"), "shape": "x".join(str(x) for x in r.get("shape", [])),
                           "dtype": str(r.get("dtype", "")).replace("torch.", ""),
                           "norm": round(float(r.get("norm", 0)), 4),
                           "requires_grad": "yes" if r.get("requires_grad") else "no",
                           "edited": f"{float(r.get('edited', 0)):.3g}"}
                          for r in rows], "no weight matches")

    def _plugin_picked(self, row):
        self.detach_btn.configure(state="normal" if row else "disabled")

    def _socket_tip(self, row):
        if row:
            self.set_subtitle(row.get("address", ""))

    def _use_socket(self, row):
        if row:
            self.pattern.delete(0, "end")
            self.pattern.insert(0, row["address"])
            self.attach_dialog()

    # ── attach / detach ────────────────────────────────────────────────────────
    def attach_dialog(self):
        top = tk.Toplevel(self)
        top.title("Attach a plugin")
        top.transient(self.app)
        frame = ttk.Frame(top)
        frame.pack(fill="both", expand=True, padx=12, pady=10)
        ttk.Label(frame, text="Socket pattern").grid(row=0, column=0, sticky="w", pady=3)
        sock = ttk.Entry(frame, width=54)
        sock.insert(0, self.pattern.get() or "*")
        sock.grid(row=0, column=1, sticky="we", pady=3)
        ttk.Label(frame, text="Plugin").grid(row=1, column=0, sticky="w", pady=3)
        kind = ttk.Combobox(frame, state="readonly", width=30)
        kind.grid(row=1, column=1, sticky="w", pady=3)
        ttk.Label(frame, text="Config (JSON)").grid(row=2, column=0, sticky="nw", pady=3)
        cfg = tk.Text(frame, width=60, height=10, font=("TkFixedFont", 9))
        cfg.grid(row=2, column=1, sticky="nsew", pady=3)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)
        bottom = ttk.Frame(top)
        bottom.pack(fill="x", padx=12, pady=(0, 10))
        status = ttk.Label(bottom, text="", style="Muted.TLabel")
        status.pack(side="left")

        self._kinds = {}

        def defaults(name):
            cfg.delete("1.0", "end")
            cfg.insert("1.0", json.dumps(self._kinds.get(name) or {}, indent=2))

        def kinds(payload):
            for row in (payload or {}).get("plugins", []):
                self._kinds[row["kind"]] = row.get("defaults") or {}
            names = sorted(self._kinds)
            kind["values"] = names
            if names:
                kind.set(names[0])
                defaults(names[0])
        self.call("plugin.kinds", None, kinds)
        kind.bind("<<ComboboxSelected>>", lambda e: defaults(kind.get()))

        def go():
            try:
                config = json.loads(cfg.get("1.0", "end").strip() or "{}")
            except json.JSONDecodeError as problem:
                status.configure(text=f"config is not JSON: {problem}", foreground=self.app.palette()["err"])
                return
            self.call("plugin.attach", {"socket": sock.get().strip() or "*",
                                        "kind": kind.get(), "config": config}, attached)
        ttk.Button(bottom, text="Attach", style="Accent.TButton", command=go).pack(side="right")
        ttk.Button(bottom, text="Cancel", style="Ghost.TButton", command=top.destroy).pack(side="right", padx=4)

        def attached(answer):
            if isinstance(answer, tuple):
                status.configure(text=str(answer[1]), foreground=self.app.palette()["err"])
                return
            mounted = (answer or {}).get("mounted") or []
            self.app.toast(f"mounted {len(mounted)} plugin(s)")
            top.destroy()
            self.refresh()
        fit_dialog(top, 640, 430)

    def detach(self):
        row = self.plugins.selected()
        if not row:
            return self.app.toast("select a mounted plugin first", error=True)
        self.call("plugin.detach", {"id": row["id"]},
                  lambda answer: self.app.toast(str(answer) if not isinstance(answer, tuple)
                                               else str(answer[1]), error=isinstance(answer, tuple)))
        self.refresh()

    # ── parameter actions ──────────────────────────────────────────────────────
    def _param(self):
        row = self.params.selected()
        if not row:
            self.app.toast("select a weight first", error=True)
            return None
        return row["name"]

    def _show(self, title, payload):
        if not payload or isinstance(payload, tuple):
            return self.app.toast(str(payload[1]) if isinstance(payload, tuple) else "no answer",
                                  error=True)
        top = tk.Toplevel(self)
        top.title(title)
        top.transient(self.app)
        show_payload(self.app, top, payload)
        ttk.Button(top, text="Close", command=top.destroy).pack(anchor="e", padx=8, pady=6)
        fit_dialog(top, 720, 520)

    def edit_param(self):
        name = self._param()
        if not name:
            return
        value = simpledialog.askstring("Edit weight", f"New value for {name}\n(a number, or JSON):",
                                       parent=self)
        if value is None:
            return
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        self.call("param.edit", {"name": name, "value": parsed}, self._acted)

    def param_stats(self):
        name = self._param()
        if name:
            self.call("param.stats", {"name": name},
                      lambda payload: self._show(f"Statistics — {name}", payload))

    def param_directions(self):
        name = self._param()
        if name:
            self.call("param.directions", {"name": name, "k": 5},
                      lambda payload: self._show(f"Directions — {name}", payload))

    def param_ablate(self):
        name = self._param()
        if not name:
            return

        def got(payload):
            if not payload or isinstance(payload, tuple):
                return self.app.toast("cannot read a direction", error=True)
            direction = payload["directions"][0] if payload.get("directions") else None
            if direction is None:
                return self.app.toast("no direction", error=True)
            self.call("param.project_out", {"name": name, "direction": direction, "side": "in"},
                      self._acted)
        self.call("param.directions", {"name": name, "k": 1}, got)

    def param_freeze(self):
        name = self._param()
        if name:
            self.call("param.freeze", {"name": name}, self._acted)

    def param_restore(self):
        row = self.params.selected()
        name = row["name"] if row else None
        self.call("param.restore", {"name": name}, self._acted)

    def _acted(self, answer):
        if isinstance(answer, tuple):
            self.note.put(f"{answer[1]}\n", "err")
            return
        self.note.put(json.dumps(answer, default=str) + "\n", "meta")
        self.load_params()
