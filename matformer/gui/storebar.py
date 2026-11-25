# matformer/gui/storebar.py
"""The file/backend/databank header shared by the activation and vector views.

Both pages look at the same physical activation stores and need the same three answers; one
widget keeps them consistent (and the focus path seeded from the server's assets)."""
import tkinter as tk
from tkinter import ttk

from matformer.gui.explorer import pick
from matformer.gui.widgets import SearchableCombobox


class StoreBar(ttk.Frame):
    def __init__(self, parent, app, on_open, on_refresh, databank="vectors"):
        super().__init__(parent)
        self.app, self.on_open, self.on_refresh = app, on_open, on_refresh
        ttk.Label(self, text="Store").pack(side="left")
        self.path = ttk.Entry(self)
        self.path.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(self, text="…", width=2, command=self._pick).pack(side="left")
        ttk.Label(self, text="Backend").pack(side="left", padx=(8, 0))
        self.backend = ttk.Combobox(self, values=["lmdb", "duckdb"], width=8, state="readonly")
        self.backend.set("lmdb")
        self.backend.pack(side="left", padx=3)
        ttk.Label(self, text="Databank").pack(side="left")
        self.databank = SearchableCombobox(self, width=18, state="normal", values=[databank])
        self.databank.set(databank)
        self.databank.pack(side="left", padx=3)
        ttk.Button(self, text="Open", style="Accent.TButton",
                   command=self.open).pack(side="left", padx=3)
        ttk.Button(self, text="Refresh", style="Ghost.TButton",
                   command=self.refresh).pack(side="left")
        last = getattr(app, "last_store", None)
        if last:
            self.path.insert(0, last.get("filename") or "")
            self.backend.set(last.get("backend") or "lmdb")
            if last.get("databank"):
                self.databank.set(last["databank"])

    def _pick(self):
        path = pick(self.app, self, "", (".lmdb", ".duckdb", ".db"), None,
                    "Choose an activation store")
        if path:
            self.path.delete(0, "end")
            self.path.insert(0, path)

    def spec(self):
        return {"filename": self.path.get().strip(), "backend": self.backend.get() or "lmdb",
                "databank": self.databank.get().strip() or "vectors"}

    def set_spec(self, databank=None, backend=None):
        if databank:
            self.databank.set(databank)
        if backend:
            self.backend.set(backend)

    def set_databanks(self, names):
        names = [name for name in (names or []) if name]
        if names:
            self.databank.set_values(names)
        else:
            self.databank.set_values([self.databank.get() or "vectors"])

    def open(self):
        spec = self.spec()
        if not spec["filename"]:
            return self.app.toast("choose a store file first", error=True)
        self.app.last_store = spec
        self.on_open(spec)

    def refresh(self):
        spec = self.spec()
        if spec.get("filename"):
            self.app.last_store = spec
        self.on_refresh(spec)

    def seed_path(self):
        if self.path.get().strip():
            return

        def got(info):
            if not info or isinstance(info, tuple):
                return
            roots = info.get("roots") if isinstance(info, dict) else info
            for root in roots or []:
                if root.get("name") == "matformer_assets":
                    self.path.insert(0, root["path"].rstrip("/") + "/steering_vectors.lmdb")
                    return
        self.app.server.call("fs.roots", None, got)
