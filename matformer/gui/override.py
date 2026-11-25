# matformer/gui/override.py
"""The override dialog, shared by Training and the config editor: a two-pane window — current
override list on top, the full framework field catalog (`config.fields_catalog`: every registered
config dataclass: path, kind, default, options, docs) below with a filter. Pick a field, type the
value (YAML literals: 3e-4, true, [1,2], "str"), Add. Because the catalog is the whole registry
view, the same window doubles as a parameter explorer for any section of Matformer — no second
GUI concept for "what can I set here"."""
import json
import tkinter as tk
from tkinter import ttk

from matformer.gui.widgets import fit_dialog, TableList, popup


class OverrideDialog(tk.Toplevel):
    """Modal editor of ["section[.entry].field=value", ...]. .result -> list or None.

    The button row is packed before anything that expands: a dialog whose accept/cancel can be
    pushed off-screen by its own content is not a dialog."""

    def __init__(self, app, server, initial=(), title="Training overrides", accept="OK"):
        super().__init__(app)
        self.app, self.server, self.title_ = app, server, title
        self.result, self.catalog = None, None
        self.title(title)
        self.geometry("820x580")
        self.minsize(620, 420)
        self.transient(app)
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=8, pady=8)
        ttk.Button(bottom, text=accept, style="Accent.TButton", command=self._ok).pack(side="right", padx=4)
        ttk.Button(bottom, text="Cancel", command=self.destroy).pack(side="right")
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=8, pady=6)
        ttk.Label(top, text="Overrides (yaml literals: 3e-4, true, [1,2], \"text\"):",
                  style="Muted.TLabel").pack(anchor="w")
        self.listbox = tk.Listbox(top, height=5, font=("TkFixedFont", 9))
        self.listbox.pack(fill="x", pady=2)
        row = ttk.Frame(top)
        row.pack(fill="x")
        ttk.Button(row, text="Remove selected", command=self._remove).pack(side="left")
        ttk.Label(row, text="   field path:", style="Muted.TLabel").pack(side="left")
        self.field = ttk.Entry(row, width=28)
        self.field.pack(side="left", padx=2)
        ttk.Label(row, text="value:").pack(side="left")
        self.value = ttk.Entry(row, width=20)
        self.value.pack(side="left", padx=2)
        self.cb_options = ttk.Combobox(row, values=[], state="readonly", width=14)
        self.cb_options.pack(side="left")
        self.cb_options.bind("<<ComboboxSelected>>",
                             lambda e: (self.value.delete(0, "end"),
                                        self.value.insert(0, self.cb_options.get())))
        self.value.bind("<Return>", lambda e: self._add())
        ttk.Button(row, text="Add", command=self._add).pack(side="left", padx=6)

        mid = ttk.Frame(self)
        mid.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 4))
        f = ttk.Frame(mid)
        f.pack(fill="x")
        ttk.Label(f, text="Field catalog:").pack(side="left")
        self.filter = ttk.Entry(f)
        self.filter.pack(side="left", fill="x", expand=True, padx=6)
        self.filter.bind("<KeyRelease>", lambda e: self._show())
        self.table = TableList(app, mid, on_double=self._pick, height=10)
        self.table.headings(["path", "kind", "default", "doc"])
        self.table.pack(fill="both", expand=True)
        self._loaded_server = None
        for o in initial:
            self.listbox.insert("end", o)
        self.load_catalog()
        self.bind("<Escape>", lambda e: self.destroy())

        def right_menu(e):
            if not self.table.tree.identify_row(e.y):
                return
            m = tk.Menu(self, tearoff=0)
            m.add_command(label="Use this field", command=lambda: self._pick(self.table.selected()))
            m.add_command(label="Copy path", command=lambda: self.app.copy(self.table.selected()["path"]))
            popup(m, e.x_root, e.y_root)
        self.table.tree.bind("<Button-3>", right_menu)

    def load_catalog(self):
        if self._loaded_server is self.server:
            return
        self._loaded_server = self.server
        self.server.call("config.fields_catalog", None, self._catalog)

    def _catalog(self, rows):
        if rows and not isinstance(rows, tuple):
            self.catalog = rows
            self._show()

    def _show(self):
        if not self.catalog:
            return
        t = self.filter.get().lower()
        self.table.fill([r for r in self.catalog
                         if t in r["path"].lower() or t in (r.get("doc") or "").lower()],
                        "no matching field")

    def _pick(self, row):
        if not row:
            return
        self.field.delete(0, "end")
        self.field.insert(0, row["path"])
        opts = row.get("options") or []
        self.cb_options.configure(values=opts)
        self.value.delete(0, "end")
        if opts:
            self.cb_options.set(opts[0])
            self.value.insert(0, opts[0])
        else:
            self.cb_options.set("")
            default = row.get("default")
            self.value.insert(0, "" if default is None else
                              (json.dumps(default)
                               if isinstance(default, (list, dict, bool)) else default))
        self.value.focus_set()

    def _add(self):
        path, value = self.field.get().strip(), self.value.get().strip()
        if not path:
            return
        try:                                    # keep strings quoted, numbers/bools/lists as yaml
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        self.listbox.insert("end", f"{path}={json.dumps(parsed)}")
        self.field.delete(0, "end")
        self.value.delete(0, "end")

    def _remove(self):
        for i in reversed(list(self.listbox.curselection())):
            self.listbox.delete(i)

    def _ok(self):
        self.result = list(self.listbox.get(0, "end"))
        self.destroy()
