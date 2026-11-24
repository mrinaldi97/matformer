# matformer/gui/mdatpage.py
"""MDAT: the dataset side of the suite. Every control here is one MDatCommands operation over RPC —
the same facade the terminal TUI drives — so the page is a transport, not a second implementation.

Operations live in menus, grouped by what they act on (datasets, views, strategies, distribution),
because the facade has forty of them and a toolbar of forty buttons helps nobody. Each menu item
opens a dialog *generated from the method's own signature* (`mdat.methods`: names, kinds,
defaults), path fields get the file picker, booleans get checkboxes, and the long operations offer
to run as a queue job — log, status, abort — instead of freezing this window.

The document browser pages through a dataset in order (peek --offset), shows one document at a time
in a pane with its own controls, and follows the container's current view and strategy: what you
see is what a training would iterate."""
import csv
import tkinter as tk
from tkinter import filedialog, ttk

from matformer.gui.explorer import pick
from matformer.gui.widgets import Console, Page, TableList, popup, show_payload

# Which menu each facade operation belongs to, and the ones worth running in the background.
GROUPS = (("Datasets", ["ls_datasets", "add", "link", "unlink", "remove", "info_dataset",
                        "ingest", "append", "history", "batch_import", "pretok", "pretok_all"]),
          ("Views", ["views", "view_add", "view_del", "view_template", "head_view", "range_view",
                     "use", "shuffle"]),
          ("Strategies", ["strategies", "strat_add", "strat_del", "set_strategy"]),
          ("Distribute", ["precompute_dist", "dist_len"]),
          ("Transfer", ["export", "state_save", "state_load", "iterate"]),
          ("Analysis", ["status", "stats", "tokens", "modality", "iterate"]),
          ("Health", ["check", "desync", "sync", "repair", "state_save", "state_load"]))
BACKGROUND = {"ingest", "append", "sync", "batch_import", "pretok", "pretok_all", "precompute_dist", "export",
              "iterate", "repair"}
PATH_FIELDS = ("path", "source", "output", "json_path", "input_file", "file", "folder")
# Anything that adds, removes or reorganises what the container holds: the three inventory tables
# are re-read after it, so the page never shows a stale view of the disk.
CHANGES_INVENTORY = {"add", "link", "unlink", "remove", "ingest", "append", "sync", "batch_import", "pretok",
                     "pretok_all", "view_add", "view_del", "use", "shuffle", "head_view",
                     "range_view", "strat_add", "strat_del", "set_strategy", "repair"}
HIDDEN = {"cmd"}


class MdatPage(Page):
    """MDAT container workbench: open, inspect, act — through menus, dialogs and a browser."""

    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "MDAT", "no container open")
        self.methods = {}
        self.dataset, self.offset, self.page_size = None, 0, 25
        self._build_bar()
        pan = ttk.PanedWindow(self.body, orient="horizontal")
        pan.pack(fill="both", expand=True)
        self._build_left(pan)
        self._build_right(pan)
        self.server().call("mdat.methods", None, self._methods)
        self.call("status")

    def server(self):
        return self.app.server

    # ── top bar: the container itself ─────────────────────────────────────────
    def _build_bar(self):
        bar = ttk.Frame(self.body)
        bar.pack(fill="x")
        ttk.Label(bar, text="Container:").pack(side="left")
        self.path = ttk.Entry(bar)
        self.path.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(bar, text="…", width=2, command=self._pick).pack(side="left")
        ttk.Button(bar, text="Open", style="Accent.TButton", command=self.open).pack(side="left", padx=3)
        self.readonly = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="read-only", variable=self.readonly).pack(side="left")
        new_button = ttk.Menubutton(bar, text="New…", width=7)
        new_menu = tk.Menu(new_button, tearoff=0)
        new_menu.add_command(label="Container…", command=lambda: self._wizard("create", self.methods.get("create")))
        new_menu.add_command(label="Standalone dataset…",
                             command=lambda: self._wizard("new_dataset", self.methods.get("new_dataset")))
        new_menu.add_command(label="Dataset in this container…",
                             command=lambda: self._wizard("add", self.methods.get("add")))
        new_button["menu"] = new_menu
        new_button.pack(side="left", padx=3)
        ttk.Button(bar, text="Close", style="Ghost.TButton",
                   command=lambda: (self.call("close"), self._blank())).pack(side="left", padx=3)
        self.menus = {}
        for title, operations in GROUPS:
            button = ttk.Menubutton(bar, text=title, style="Ghost.TButton")
            menu = tk.Menu(button, tearoff=0)
            button["menu"] = menu
            button.pack(side="left", padx=1)
            self.menus[title] = (button, menu)
        self._fill_menus()

    def _fill_menus(self):
        for title, operations in GROUPS:
            _, menu = self.menus[title]
            menu.delete(0, "end")
            for name in operations:
                method = self.methods.get(name)
                label = (name.replace("_", " ").capitalize())
                if method and method["args"]:
                    menu.add_command(label=f"{label}…",
                                     command=lambda n=name, m=method: self._wizard(n, m))
                else:
                    menu.add_command(label=label, command=lambda n=name: self.call(n))
            menu.add_separator()
            menu.add_command(label="Type the command instead…",
                             command=lambda t=title: self._focus_line())

    def _focus_line(self):
        self.entry.focus_set()

    def _methods(self, payload):
        if not payload or isinstance(payload, tuple):
            return
        self.methods = {m["name"]: m for m in payload}
        self._fill_menus()

    def _pick(self):
        p = pick(self.app, self, "", (), None, "Choose a container or dataset folder")
        if p:
            self.path.delete(0, "end")
            self.path.insert(0, p)

    def open(self):
        self.call("open", {"path": self.path.get(), "readonly": self.readonly.get()})

    def refresh_all(self):
        for command in ("ls_datasets", "views", "strategies"):
            self.call(command)

    def _blank(self):
        self.dataset = None
        for table in (self.datasets, self.views, self.docs):
            table.fill([], "no container open")
        self.ds_pick.configure(values=[])
        self.view_pick.configure(values=[])
        self.strat_pick.configure(values=[])
        self.set_subtitle("no container open")

    # ── left: what the container holds ────────────────────────────────────────
    def _build_left(self, pan):
        left = ttk.Frame(pan)
        pan.add(left, weight=2)
        head = ttk.Frame(left)
        head.pack(fill="x")
        ttk.Label(head, text="Datasets", style="Head.TLabel").pack(side="left")
        ttk.Button(head, text="⟳", width=3, style="Ghost.TButton",
                   command=lambda: self.call("ls_datasets")).pack(side="right")
        self.datasets = TableList(self.app, left, on_double=self._pick_dataset, height=10)
        self.datasets.headings(["name", "docs", "size", "strategies"])
        self.datasets.fill([], "no container open")
        self.datasets.pack(fill="both", expand=True)
        self.datasets.tree.bind("<Button-3>", self._dataset_menu)
        views = ttk.Frame(left)
        views.pack(fill="x", pady=(6, 0))
        head = ttk.Frame(views)
        head.pack(fill="x")
        ttk.Label(head, text="Views", style="Head.TLabel").pack(side="left")
        ttk.Button(head, text="⟳", width=3, style="Ghost.TButton",
                   command=lambda: self.call("views")).pack(side="right")
        self.views = TableList(self.app, views, on_double=lambda row: row and self.call(
            "use", {"view": row["view"]}), height=6)
        self.views.headings(["view", "docs", "shuffle", "active"])
        self.views.fill([], "no container open")
        self.views.pack(fill="x")

    def _dataset_menu(self, e):
        row = self.datasets.selected()
        if not row:
            return
        name = row["name"]
        m = tk.Menu(self, tearoff=0)
        m.add_command(label=f"Browse documents ({name})", command=lambda: self.browse(name))
        m.add_command(label="Info", command=lambda: self.call("info_dataset", {"name": name}))
        m.add_command(label="Append documents…",
                      command=lambda: self._wizard("append", self.methods.get("append"),
                                                   {"dataset": name}))
        m.add_command(label="Pretokenize…",
                      command=lambda: self._wizard("pretok", self.methods.get("pretok"),
                                                   {"dataset": name}))
        m.add_command(label="Sync strategy…",
                      command=lambda: self._wizard("sync", self.methods.get("sync"),
                                                   {"dataset": name}))
        m.add_command(label="Ingestion history",
                      command=lambda: self.call("history", {"dataset": name}))
        m.add_command(label="Export a view…",
                      command=lambda: self._wizard("export", self.methods.get("export")))
        m.add_separator()
        m.add_command(label="Unlink", command=lambda: self.call("unlink", {"alias": name}))
        m.add_command(label="Remove…",
                      command=lambda: self._wizard("remove", self.methods.get("remove"),
                                                   {"name": name}))
        popup(m, e.x_root, e.y_root)

    def _pick_dataset(self, row):
        if row:
            self.browse(row["name"])

    # ── right: the browser ────────────────────────────────────────────────────
    def _build_right(self, pan):
        right = ttk.Frame(pan)
        pan.add(right, weight=3)
        nav = ttk.Frame(right)
        nav.pack(fill="x")
        ttk.Label(nav, text="Dataset").pack(side="left")
        self.ds_pick = ttk.Combobox(nav, values=[], state="readonly", width=18)
        self.ds_pick.pack(side="left", padx=4)
        self.ds_pick.bind("<<ComboboxSelected>>", lambda e: self.browse(self.ds_pick.get(), 0))
        ttk.Label(nav, text="View").pack(side="left", padx=(8, 0))
        self.view_pick = ttk.Combobox(nav, values=[], state="readonly", width=14)
        self.view_pick.pack(side="left", padx=4)
        self.view_pick.bind("<<ComboboxSelected>>", lambda e: self.call("use", {"view": self.view_pick.get()}))
        ttk.Label(nav, text="Strategy").pack(side="left", padx=(8, 0))
        self.strat_pick = ttk.Combobox(nav, values=[], state="readonly", width=14)
        self.strat_pick.pack(side="left", padx=4)
        self.strat_pick.bind("<<ComboboxSelected>>",
                             lambda e: self.call("set_strategy", {"name": self.strat_pick.get()}))
        self.chunks = tk.BooleanVar(value=False)
        ttk.Checkbutton(nav, text="chunks", variable=self.chunks,
                        command=lambda: self.browse(self.dataset, self.offset)).pack(side="left", padx=6)
        pager = ttk.Frame(right)
        pager.pack(fill="x", pady=4)
        ttk.Button(pager, text="⏮", width=3, style="Ghost.TButton",
                   command=lambda: self.browse(self.dataset, 0)).pack(side="left")
        ttk.Button(pager, text="←", width=3, style="Ghost.TButton",
                   command=lambda: self.browse(self.dataset, max(0, self.offset - self.page_size))).pack(side="left")
        self.where = ttk.Label(pager, text="0", style="Muted.TLabel")
        self.where.pack(side="left", padx=6)
        ttk.Button(pager, text="→", width=3, style="Ghost.TButton",
                   command=lambda: self.browse(self.dataset, self.offset + self.page_size)).pack(side="left")
        self.jump = ttk.Entry(pager, width=7)
        self.jump.pack(side="left", padx=4)
        self.jump.bind("<Return>", lambda e: self.browse(self.dataset, int(self.jump.get() or 0)))
        ttk.Label(pager, text="rows").pack(side="left", padx=(8, 0))
        self.size = ttk.Combobox(pager, values=[10, 25, 50, 100], width=4, state="readonly")
        self.size.set(self.page_size)
        self.size.bind("<<ComboboxSelected>>",
                       lambda e: setattr(self, "page_size", int(self.size.get()))
                       and self.browse(self.dataset, self.offset))
        self.docs = TableList(self.app, right, on_double=self._show_document, height=12)
        self.docs.headings(["key", "document"])
        self.docs.fill([], "open a container, pick a dataset")
        self.docs.pack(fill="both", expand=True)
        self.docs.tree.bind("<Button-3>", self._document_menu)
        line = ttk.Frame(right)
        line.pack(fill="x", pady=(6, 0))
        ttk.Label(line, text="MDAT>").pack(side="left")
        self.entry = ttk.Entry(line)
        self.entry.pack(side="left", fill="x", expand=True, padx=6)
        self.entry.bind("<Return>", lambda e: self.run_line())
        self.out = ttk.Frame(right)
        self.out.pack(fill="both", expand=True)
        self.doc = Console(self.app, self.out)
        self.doc.pack(fill="both", expand=True)
        self.doc.text.configure(height=8)

    def browse(self, dataset=None, offset=0):
        self.dataset = dataset or self.dataset
        if not self.dataset:
            return self.app.toast("pick a dataset first", error=True)
        self.offset = max(0, int(offset))
        self.call("peek", {"dataset": self.dataset, "n": self.page_size, "offset": self.offset,
                           "chunks": self.chunks.get()})

    def _document_menu(self, e):
        row = self.docs.selected()
        if not row:
            return
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Open document", command=lambda: self._show_document(row))
        m.add_command(label="Copy key", command=lambda: self.app.copy(row.get("key")))
        m.add_command(label="Export this page as CSV…", command=lambda: self._export(self.docs))
        popup(m, e.x_root, e.y_root)

    def _show_document(self, row):
        if not row:
            return
        key = int(row["key"])
        self.doc.clear()
        self.doc.put(f"loading {self.dataset} #{key}…\n", "meta")
        self.server().call("mdat.call", {"cmd": "doc", "args": {"dataset": self.dataset, "key": key}},
                           lambda t: t and not isinstance(t, tuple)
                           and (self.doc.clear(), self.doc.put(str(t))))
        self._doc_key = key

    # ── the one call path ─────────────────────────────────────────────────────
    def activated(self):
        self.call("status")

    def server_changed(self):
        self.current = None
        self.call("status")

    def call(self, cmd, args=None):
        self.server().call("mdat.call", {"cmd": cmd, "args": args or {}},
                           lambda payload: self.render(payload, cmd, args or {}))

    def run_line(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self.server().call("mdat.line", {"line": text}, lambda payload: self.render(payload, "line", {}))

    def render(self, payload, cmd, args):
        if payload is None or isinstance(payload, tuple):
            return
        if cmd == "open":
            self.set_subtitle(str(payload).splitlines()[0] if payload else "")
            self.refresh_all()
            return
        if cmd == "close":
            self._blank()
            return
        if isinstance(payload, dict) and payload.get("__table__"):
            columns = payload["columns"]
            rows = payload["rows"]
            if columns == ["name", "docs", "size", "kind", "strategies"]:
                self.datasets.headings(["name", "docs", "size", "strategies"])
                self.datasets.fill(rows, "empty container")
                self.ds_pick.configure(values=[r["name"] for r in rows])
                if rows and not self.dataset:
                    self.browse(rows[0]["name"], 0)
            elif columns == ["view", "docs", "size", "shuffle", "active"]:
                self.views.fill(rows, "no views")
                self.view_pick.configure(values=[r["view"] for r in rows])
                active = next((r["view"] for r in rows if r.get("active")), None)
                if active:
                    self.view_pick.set(active)
            elif columns == ["strategy"]:
                names = [r["strategy"] for r in rows]
                self.strat_pick.configure(values=names)
            elif columns == ["key", "document"]:
                self.docs.fill(rows, "no documents in this range")
                self.where.configure(text=(f"{self.offset} – {self.offset + len(rows) - 1}"
                                           if rows else "empty"))
            else:
                self._show_table(payload)
            if cmd == "stats":
                self._show_table(payload)
            if cmd in CHANGES_INVENTORY:
                self.refresh_all()
            return
        self.doc.clear()
        self.doc.put(str(payload))

    def _show_table(self, payload):
        dlg = tk.Toplevel(self.app)
        dlg.title("MDAT")
        dlg.geometry("820x480")
        dlg.transient(self.app)
        table = show_payload(self.app, dlg, payload)
        bar = ttk.Frame(dlg)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Close", command=dlg.destroy).pack(side="right")
        ttk.Button(bar, text="Export CSV…", style="Ghost.TButton",
                   command=lambda: self._export(table)).pack(side="right", padx=4)
        dlg.bind("<Escape>", lambda e: dlg.destroy())

    def _export(self, table):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(table.columns)
            for row in table.rows:
                writer.writerow(row if not isinstance(row, dict)
                                else [row.get(c, "") for c in table.columns])
        self.app.toast(f"exported {path}")

    # ── generated wizards ─────────────────────────────────────────────────────
    def _wizard(self, name, method, preset=None):
        if method is None:
            return self.app.toast(f"the server does not expose mdat.{name}", error=True)
        MdatWizard(self, name, method, preset or {})


class MdatWizard(tk.Toplevel):
    """A dialog generated from one MDatCommands signature: argument names, kinds and defaults come
    from the facade, path-ish ones get the picker, and the long operations can be handed to the
    queue instead of blocking this window."""

    def __init__(self, page, name, method, preset):
        super().__init__(page)
        self.page, self.name = page, name
        self.title(f"MDAT · {name.replace('_', ' ')}")
        self.transient(page)
        self.vars = {}
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=12, pady=10)
        ttk.Button(bottom, text="Run", style="Accent.TButton", command=self._run).pack(side="right", padx=4)
        ttk.Button(bottom, text="Cancel", command=self.destroy).pack(side="right")
        self.background = tk.BooleanVar(value=name in BACKGROUND)
        if name in BACKGROUND:
            ttk.Checkbutton(bottom, text="run as a job (log, progress, abort)",
                            variable=self.background).pack(side="left")
        form = ttk.Frame(self)
        form.pack(side="top", fill="both", expand=True, padx=12, pady=12)
        form.columnconfigure(1, weight=1)
        if method.get("doc"):
            ttk.Label(form, text=method["doc"], style="Muted.TLabel", wraplength=460,
                      justify="left").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        for row, arg in enumerate(method["args"], start=1):
            if arg["name"] in HIDDEN:
                continue
            self._field(form, row, arg, preset)
        self.geometry("560x%s" % max(200, 60 + 44 * len(method["args"])))
        self.bind("<Escape>", lambda e: self.destroy())

    def _field(self, form, row, arg, preset):
        label = arg["name"].replace("_", " ").capitalize()
        default = preset.get(arg["name"], arg.get("default"))
        is_path = any(hint in arg["name"] for hint in PATH_FIELDS)
        ttk.Label(form, text=label + ":").grid(row=row, column=0, sticky="w", pady=3)
        if arg["kind"] == "bool":
            var = tk.BooleanVar(value=bool(default))
            ttk.Checkbutton(form, variable=var).grid(row=row, column=1, sticky="w")
        elif arg["kind"] == "choice":
            var = ttk.Combobox(form, values=arg.get("options") or [], state="readonly")
            var.set(default if default in (arg.get("options") or [])
                    else (arg["options"][0] if arg.get("options") else ""))
            var.grid(row=row, column=1, sticky="w", pady=3)
        else:
            holder = ttk.Frame(form)
            holder.grid(row=row, column=1, sticky="ew", pady=3)
            entry = ttk.Entry(holder)
            entry.pack(side="left", fill="x", expand=True)
            if default is not None:
                entry.insert(0, str(default))
            if is_path:
                ttk.Button(holder, text="…", width=3,
                           command=lambda e=entry, a=arg: self._pick(e, a)).pack(side="left", padx=3)
            var = entry
        self.vars[arg["name"]] = (var, arg, is_path)

    def _pick(self, entry, arg):
        exts = (".json",) if "json" in arg["name"] else (".jsonl", ".csv", ".parquet") if "source" in arg["name"] else ()
        p = pick(self.page.app, self, "", exts, None, arg["name"].replace("_", " "))
        if p:
            entry.delete(0, "end")
            entry.insert(0, p)

    def _values(self):
        out = {}
        for name, (var, arg, _) in self.vars.items():
            value = var.get()
            if arg["kind"] == "int":
                value = int(value) if str(value).strip() != "" else None
            elif isinstance(value, str):
                value = value.strip() or None
            if value is not None:
                out[name] = value
        return out

    def _run(self):
        values = self._values()
        if self.background.get():
            container = values.pop("container", None) or self.page.path.get()
            self.page.server().call("mdat.start",
                                    {"container": container, "name": self.name,
                                     "commands": [{"cmd": self.name, "args": values}]},
                                    lambda r: r and not isinstance(r, tuple)
                                    and (self.page.app.toast(f"{self.name}: queued as {r['run']}"),
                                         self.page.app.toggle_dock(force=True), self.destroy()))
            return
        self.page.call(self.name, values)
        self.destroy()
