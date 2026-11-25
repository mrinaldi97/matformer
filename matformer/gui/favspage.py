# matformer/gui/favspage.py
"""Favourites: the short lists a engineer keeps at hand — models (local checkpoints or hf:// ids)
and datasets. Populated from the file manager's right-click menu; consumed here: one click to load
a model into the Task pages, open a dataset in MDAT, or reveal a path."""
from tkinter import simpledialog, ttk

from matformer.gui.explorer import pick
from matformer.gui.widgets import Page, TableList


class FavPage(Page):
    """Models & datasets favourites, stored server-side (one small JSON file)."""

    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Favourites", "")
        pan = ttk.PanedWindow(self.body, orient="horizontal")
        pan.pack(fill="both", expand=True)
        self.tables = {}
        for kind, title in (("models", "Favourite models"), ("datasets", "Favourite datasets")):
            frame = ttk.Frame(pan)
            pan.add(frame, weight=1)
            ttk.Label(frame, text=title, style="Head.TLabel").pack(fill="x")
            table = TableList(app, frame, height=10)
            table.headings(["item"])
            table.fill([])
            table.pack(fill="both", expand=True)
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=4)
            ttk.Button(row, text="Add…", command=lambda k=kind, t=table: self.add(k, t)
                       ).pack(side="left")
            ttk.Button(row, text="Remove",
                       command=lambda k=kind, t=table: self.remove(k, t)).pack(side="left", padx=4)
            if kind == "models":
                ttk.Button(row, text="Load in Tasks", style="Accent.TButton",
                           command=lambda t=table: self._load_model(t)).pack(side="left")
            else:
                ttk.Button(row, text="Open in MDAT", style="Accent.TButton",
                           command=lambda t=table: self._open_dataset(t)).pack(side="left")
            self.tables[kind] = table

    def activated(self):
        self.refresh()

    def refresh(self):
        self.app.server.call("fav.list", None, self._fill)

    def _fill(self, favs):
        if not favs or isinstance(favs, tuple):
            return
        self.tables["models"].fill([{"item": i} for i in favs.get("models", [])],
                                   "No favourite models — add them from the file manager.")
        self.tables["datasets"].fill([{"item": i} for i in favs.get("datasets", [])],
                                     "No favourite datasets yet.")

    def add(self, kind, table):
        p = pick(self.app, self, "", (), None, "Choose a " + kind[:-1])
        if not p:
            p = simpledialog.askstring("Favourite", "Path or hf://org/model:", parent=self)
        if p:
            self.app.server.call("fav.add", {"kind": kind, "item": p.strip()}, lambda r: self.refresh())

    def remove(self, kind, table):
        row = table.selected()
        if row:
            self.app.server.call("fav.delete", {"kind": kind, "item": row["item"]},
                                 lambda r: self.refresh())

    def _load_model(self, table):
        row = table.selected()
        if row:
            self.app.open_model(row["item"])

    def _open_dataset(self, table):
        row = table.selected()
        if row and "://" not in row["item"]:
            self.app.open_dataset(row["item"])
