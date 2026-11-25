# matformer/gui/wizardpage.py
"""The wizard browser: every TOML definition the server exposes, grouped by category.

A wizard is the primary way to start long work. Double-click one, or press Open. The page keeps
no state of its own: the definitions live with the framework, the work is queued by the server.
"""
import tkinter as tk
from tkinter import ttk

from matformer.gui.widgets import Page
from matformer.gui.wizard import WizardDialog


class WizardPage(Page):
    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Wizards",
                         "Guided configurations for training, data and annotation")
        bar = ttk.Frame(self.body)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Label(bar, text="Search:").pack(side="left")
        self.filter = ttk.Entry(bar)
        self.filter.pack(side="left", fill="x", expand=True, padx=4)
        self.filter.bind("<KeyRelease>", lambda e: self._fill())
        ttk.Button(bar, text="Open", style="Accent.TButton",
                   command=self.open_selected).pack(side="left", padx=4)
        ttk.Button(bar, text="Refresh", style="Ghost.TButton", command=self.reload).pack(side="left")
        self.tree = ttk.Treeview(self.body, show="tree headings", selectmode="browse",
                                 columns=("description", "steps"))
        self.tree.heading("#0", text="Wizard")
        self.tree.heading("description", text="What it does")
        self.tree.heading("steps", text="Steps")
        self.tree.column("#0", width=250)
        self.tree.column("description", width=460)
        self.tree.column("steps", width=50, anchor="e")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self.open_selected())
        self.rows = []
        self.groups = {}
        self.set_subtitle("reading the definitions…")
        self.reload()

    @property
    def server(self):
        return self.app.server

    def activated(self):
        if not self.rows:
            self.reload()

    def server_changed(self):
        self.rows = []
        self.reload()

    def reload(self):
        self.server.call("wizard.list", None, self._loaded)

    def _loaded(self, payload):
        if not payload or isinstance(payload, tuple):
            self.set_subtitle("no answer from the server")
            return
        self.rows = payload
        self.set_subtitle(f"{len(self.rows)} wizards")
        self._fill()

    def _fill(self):
        text = self.filter.get().strip().lower()
        self.tree.delete(*self.tree.get_children())
        self.groups = {}
        shown = 0
        for row in self.rows:
            haystack = " ".join((row.get("title", ""), row.get("description", ""),
                                 row.get("category", ""), row.get("id", ""))).lower()
            if text and text not in haystack:
                continue
            category = row.get("category", "General")
            if category not in self.groups:
                self.groups[category] = self.tree.insert("", "end", text=category, open=True)
            self.tree.insert(self.groups[category], "end", iid=row["id"], text=row["title"],
                             values=(row.get("description", ""), row.get("steps", "")))
            shown += 1
        if not shown:
            self.set_subtitle("no wizard matches the filter" if text else "no wizards on this server")

    def selected(self):
        selection = self.tree.selection()
        return selection[0] if selection and selection[0] in {row["id"] for row in self.rows} else None

    def open_selected(self):
        wizard_id = self.selected()
        if not wizard_id:
            self.app.toast("select a wizard first", error=True)
            return
        self.launch(wizard_id)

    def launch(self, wizard_id):
        WizardDialog(self.app, wizard_id, on_done=lambda answer: self._done(wizard_id, answer))

    def _done(self, wizard_id, answer):
        self.app.log("WIZARD", f"{wizard_id}: {answer}")
        self.set_subtitle(f"last queued: {answer.get('kind')} {answer.get('run') or answer.get('config') or ''}")
        if answer.get("kind") in ("train", "mdat", "annotate"):
            self.app.toggle_dock(force=True)


def open_wizard(app, wizard_id, initial=None):
    """Entry point used by menus and other pages: opens one wizard with optional prefilled answers."""
    WizardDialog(app, wizard_id, initial=initial)
