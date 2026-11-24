# matformer/gui/syspage.py
"""System: the machine the (global) server runs on — read-only dashboard: OS, versions, RAM,
disk, GPUs, allowed roots. Right-click any row to copy its value."""
from tkinter import ttk

from matformer.gui.widgets import Page, TableList


class SysPage(Page):
    """Machine status of the active server."""

    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "System", "")
        row = ttk.Frame(self.body)
        row.pack(fill="x")
        ttk.Button(row, text="Refresh", style="Accent.TButton", command=self.refresh).pack(side="left")
        pan = ttk.PanedWindow(self.body, orient="horizontal")
        pan.pack(fill="both", expand=True, pady=6)
        left = ttk.Frame(pan)
        pan.add(left, weight=1)
        self.info = TableList(app, left, on_menu=_copy_menu, height=14)
        self.info.headings(["property", "value"])
        self.info.fill([])
        self.info.pack(fill="both", expand=True)
        right = ttk.Frame(pan)
        pan.add(right, weight=1)
        ttk.Label(right, text="GPUs", style="Head.TLabel").pack(fill="x")
        self.gpus = TableList(app, right, height=5)
        self.gpus.headings(["index", "name", "mem_used_mib", "mem_total_mib", "util", "driver"])
        self.gpus.fill([], "No GPU visible to nvidia-smi.")
        self.gpus.pack(fill="x")
        ttk.Label(right, text="Allowed roots", style="Head.TLabel").pack(fill="x", pady=(8, 0))
        self.roots = TableList(app, right, height=5)
        self.roots.headings(["root"])
        self.roots.fill([])
        self.roots.pack(fill="both", expand=True)

    def activated(self):
        self.refresh()

    def refresh(self):
        def got(info):
            if not info or isinstance(info, tuple):
                return
            gpus, roots = info.pop("gpus", []), info.pop("roots", [])
            self.info.fill([{"property": k, "value": _humanize(k, v)}
                            for k, v in sorted(info.items()) if v not in (None, "", [])],
                           "Server offline.")
            self.gpus.fill(gpus or [], "No GPU visible to nvidia-smi.")
            self.roots.fill([{"root": r} for r in roots], "—")
            self.set_subtitle(f"{info.get('name', '')}   ·   {info.get('host', '')}")
        self.app.server.call("sys.info", None, got)


def _copy_menu(m, row):
    if not row:
        return
    def do_copy():
        top = m.winfo_toplevel()
        top.clipboard_clear()
        top.clipboard_append(str(row.get("value", "")))
    m.add_command(label="Copy value", command=do_copy)


def _humanize(key, value):
    if key.startswith(("ram", "disk")):
        return f"{value / 2**30:.1f} GiB" if isinstance(value, (int, float)) else "-"
    return str(value)
