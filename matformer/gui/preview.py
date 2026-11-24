# matformer/gui/preview.py
"""Throwaway harness: renders the components built so far (theme, forms, console, table, file
manager over the REAL protocol against the embedded server). Run from the repo root:

    python -m matformer.gui.preview
"""
import os
import queue
import sys
import tkinter as tk
from tkinter import ttk

os.environ.setdefault("MATFORMER_SERVER_PORT", "8799")

from matformer.gui import theme
from matformer.gui.explorer import Explorer, pick
from matformer.gui.forms import Form
from matformer.gui.widgets import Console, TableList
from matformer.remote.net import Server

PALETTES = {
    "classic": dict(bg="#f0efe9", panel="#e4e2d9", field="#ffffff", fg="#1a1a1a", muted="#6f6f6f",
                    sel="#d8e6f7", accent="#0a246a", border="#b0aca0", on_accent="#ffffff",
                    ok="#2a7f2a", err="#b03030", grid="#dcd9d0"),
    "dark": dict(bg="#2b2b2b", panel="#333333", field="#1e1e1e", fg="#e6e6e6", muted="#9aa0a6",
                 sel="#3f5a7d", accent="#4f83cc", border="#4a4a4a", on_accent="#ffffff",
                 ok="#4ec94e", err="#ef6b6b", grid="#3a3a3a"),
}

FIELDS = [
    {"name": "temperature", "kind": "float", "default": 0.8, "min": 0, "max": 2, "step": 0.05,
     "doc": "Sampling temperature"},
    {"name": "max_new_tokens", "kind": "int", "default": 200, "min": 1, "max": 8192},
    {"name": "do_sample", "kind": "bool", "default": True},
    {"name": "presentation", "kind": "radio", "options": ["probs", "argmax", "full"], "default": "probs"},
    {"name": "decoding", "kind": "choice", "options": ["sample", "greedy", "beam"], "default": "sample"},
    {"name": "lr", "kind": "sci", "default": 0.0003},
    {"name": "topk_ids", "kind": "json", "default": [5, 9], "toggle": True},
    {"name": "model", "kind": "path", "hint": "checkpoint or hf://id"},
]

ROWS = [{"id": f"train-{i}", "project": "sae-sweep", "status": s, "step": st,
         "progress": p, "started": f"2026-11-1{i} 09:0{st}"}
        for i, (s, st, p) in enumerate([("running", 1420, "34%"), ("queued", 0, "0%"),
                                        ("done", 4000, "100%"), ("failed", 210, "12%")])]


class Preview:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Matformer — component preview")
        self.root.geometry("1360x820")
        import tkinter as _tk
        self.server_var = _tk.StringVar(value="local")
        self.fm_mode = "details"
        self.theme_name = "classic"
        self.paintables = []
        self.events = queue.Queue()
        self.servers = {"local": Server("local", "127.0.0.1:" + os.environ["MATFORMER_SERVER_PORT"],
                                        self.events)}
        self.server_names = ["local"]
        theme.apply(self.root, self.theme_name, PALETTES[self.theme_name])
        self._build()
        for s in self.servers.values():
            s.start()
        self._poll()

    def register_paintable(self, w):
        self.paintables.append(w)

    def palette(self):
        return PALETTES[self.theme_name]

    def server_of(self, name=None):
        return self.servers[name or self.server_var.get()]

    def status(self, msg):
        self.statusbar.configure(text=msg)

    def toast(self, msg, error=False):
        self.status(("ERROR " if error else "") + msg)

    def copy(self, p):
        self.root.clipboard_clear()
        self.root.clipboard_append(p)

    def open_file(self, path, srv): self.status(f"open_file {path}")
    def open_config(self, path, srv): self.status(f"open_config {path}")
    def open_model(self, path, srv): self.status(f"open_model {path}")
    def annotate_file(self, path, srv): self.status(f"annotate {path}")
    def start_training(self, path, srv): self.status(f"train {path}")

    def browse(self, root="", exts=()):
        return pick(self, self.root, root, tuple(exts))

    def toggle_theme(self):
        self.theme_name = "dark" if self.theme_name == "classic" else "classic"
        theme.apply(self.root, self.theme_name, PALETTES[self.theme_name])
        for w in self.paintables:
            w.repaint(PALETTES[self.theme_name])

    def _build(self):
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="Toggle theme", command=self.toggle_theme).pack(side="left")
        ttk.Label(bar, text="   Matformer preview: forms, console, table, file manager over NDJSON-RPC"
                  ).pack(side="left")
        self.statusbar = ttk.Label(self.root, text="ready", style="Muted.TLabel")
        self.statusbar.pack(side="bottom", fill="x", padx=8)

        grid = ttk.PanedWindow(self.root, orient="horizontal")
        grid.pack(fill="both", expand=True, padx=6, pady=4)

        left = ttk.Frame(grid)
        grid.add(left, weight=1)
        ttk.Label(left, text="File Manager (live fs.list)", style="Head.TLabel").pack(fill="x")
        self.explorer = Explorer(self, left)
        self.explorer.pack(fill="both", expand=True)

        mid = ttk.Frame(grid)
        grid.add(mid, weight=1)
        ttk.Label(mid, text="Transcript (Console)", style="Head.TLabel").pack(fill="x")
        self.console = Console(self, mid)
        self.console.pack(fill="both", expand=True, pady=(0, 6))
        self.console.put("you> La casa è\nmodel> ")
        for tok in ["grande", " e", " bella", "."]:
            self.console.put(tok)
        self.console.put("\n  7 in tok · 4 out · 61.8 ms\n")

        right = ttk.Frame(grid)
        grid.add(right, weight=1)
        ttk.Label(right, text="Jobs (TableList, sortable)", style="Head.TLabel").pack(fill="x")
        self.table = TableList(self, right)
        self.table.headings(["id", "project", "status", "step", "progress", "started"])
        self.table.fill(ROWS)
        self.table.pack(fill="both", expand=True, pady=(0, 6))
        ttk.Label(right, text="Task parameters (Form from schema)", style="Head.TLabel").pack(fill="x")
        holder = ttk.Frame(right)
        holder.pack(fill="x")
        self.form = Form(self, holder, FIELDS)
        self.form.pack(fill="x")
        ttk.Button(right, text="Print form values",
                   command=lambda: self.status(str(self.form.values()))).pack(anchor="w", pady=8)

    def _poll(self):
        try:
            while True:
                sname, name, data = self.events.get_nowait()
                if name == "__result__":
                    rid, result = data
                    cb = self.servers["local"].pending.pop(rid, None)
                    if cb:
                        cb(result)
                elif name == "__state__":
                    self.status(f"server {'up' if data else 'down'}")
                    if data:
                        self.explorer.refresh_roots()
                elif name == "error":
                    self.status(f"ERROR {data}")
        except queue.Empty:
            pass
        self.root.after(60, self._poll)


if __name__ == "__main__":
    from matformer.matformer_registry import registry      # noqa: F401 (boots the server thread)
    Preview().root.mainloop()
