# matformer/gui/regpage.py
"""Registry: the framework's own catalog, graphically parsed. Categories on the left (filterable,
live from the local registry — same data the server has); selecting one expands its tree on the
right: variants → definition location → implementations with priority, requires, constraints and
availability-test status. This is where you check which kernel won and why, no terminal needed."""
import tkinter as tk
from tkinter import ttk

from matformer.gui.icons import icon
from matformer.gui.widgets import Page, TableList


class RegistryPage(Page):
    """Category browser over the registry, tree-rendered."""

    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Registry", "categories, variants, implementations")
        row = ttk.Frame(self.body)
        row.pack(fill="x")
        ttk.Button(row, text="Refresh", style="Accent.TButton", command=self.refresh).pack(side="left")
        self.filter = ttk.Entry(row)
        self.filter.pack(side="left", fill="x", expand=True, padx=8)
        self.filter.bind("<KeyRelease>", lambda e: self._apply_filter())
        pan = ttk.PanedWindow(self.body, orient="horizontal")
        pan.pack(fill="both", expand=True, pady=6)
        left = ttk.Frame(pan)
        pan.add(left, weight=2)
        self.cats = TableList(app, left, on_double=self._show, on_menu=self._cat_menu, height=16)
        self.cats.headings(["category", "default", "variants"])
        self.cats.fill([])
        self.cats.pack(fill="both", expand=True)
        right = ttk.Frame(pan)
        pan.add(right, weight=3)
        self.detail_title = ttk.Label(right, text="Category detail", style="Head.TLabel")
        self.detail_title.pack(fill="x")
        self.tree = ttk.Treeview(right)
        tbar = ttk.Scrollbar(right, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tbar.set)
        tbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.tree.heading("#0", text="entry", anchor="w")
        self.all_rows = []
        self.refresh()
        self.tree.bind("<<TreeviewOpen>>", lambda e: None)

    def activated(self):
        self.refresh()

    def refresh(self):
        try:
            from matformer.matformer_registry import registry
            self._info = registry.get_registry_info(relative_filepath=True)
        except Exception as problem:
            self.app.toast(f"local registry unavailable: {problem}", error=True)
            return
        self.all_rows = [{"category": name, "default": cat.get("default_variant"),
                          "variants": len(cat["variants"])} for name, cat in sorted(self._info.items())]
        self._apply_filter()

    def _apply_filter(self):
        t = self.filter.get().lower()
        self.cats.fill([r for r in self.all_rows
                        if t in r["category"].lower() or t in str(r["variants"]).lower()],
                       "Registry is empty.")

    def _cat_menu(self, m, row):
        if row:
            m.add_command(label="Copy category", command=lambda: self.app.copy(row["category"]))

    def _show(self, row):
        row = row or self.cats.selected()
        if not row:
            return
        cat = self._info.get(row["category"], {})
        self.detail_title.configure(text=row["category"])
        self.tree.delete(*self.tree.get_children())
        for variant, vnode in sorted(cat.get("variants", {}).items()):
            v = self.tree.insert("", "end", text=variant, open=len(cat.get("variants", {})) < 4,
                                 image=icon("dir"))
            if vnode.get("definition"):
                d = vnode["definition"]
                self.tree.insert(v, "end", text=f"definition · {d['pkgname']}",
                                 values=(d.get("filepath") or "",), image=icon("config"))
            for impl_name, impl in sorted(vnode.get("implementations", {}).items(),
                                          key=lambda kv: -kv[1].get("priority", 0)):
                test = impl.get("test") or ""
                status = {"passed": "● ok", "failed": "● unavailable"}.get(test, test)
                node = self.tree.insert(v, "end", text=f"{impl_name}   ·   prio {impl.get('priority', 0)}"
                                             + (f"   ·   {status}" if status and status != "not tested" else ""),
                                        image=icon("model"))
                self.tree.insert(node, "end", text=(impl.get("filepath") or "")
                                 + (f"   requires: {', '.join(impl['requires'])}" if impl.get("requires") else "")
                                 + (f"   constraints: {impl['constraints']}" if impl.get("constraints") else ""))
        self.tree.column("#0", width=520, anchor="w", stretch=True)

