# matformer/gui/explorer.py
"""The file manager. Two synchronized views over the server filesystem (fs.*), jailed to the
roots declared in server.toml:

  • Details (default) — a hierarchical tree-table: sortable Name column, icon, Size and Modified
    right-aligned, folders lazy-load when opened (placeholder pump: expansion never depends on
    focus, refresh races are TclError-guarded).
  • Icons — a 64 px grid (Qt-style, two-line wrapped labels, selection highlight, hover tooltip)
    for spatial browsing.

One address bar: a breadcrumb of clickable segments; Ctrl+L swaps it for an editable entry that
also accepts hf:// URIs. The roots dropdown is one click from anywhere. Context menus are built
per detected type: YAML -> Edit / Train, weights -> Load as model, data -> Annotate, anything ->
favourites / rename / delete. `pick()` (the modal every path field in the suite uses) is this same
widget with an extension filter."""
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from matformer.gui.icons import icon, icon_for
from matformer.gui.widgets import fmt_time, placeholder, popup, tooltip, dismiss

WEIGHTS = (".ckpt", ".pt", ".safetensors", ".bin")
DATA = (".jsonl", ".json", ".csv")
CELL_W, CELL_H = 112, 108


def join(base, name):
    return base.rstrip("/") + "/" + name


def human(n):
    for unit in ("B", "K", "M", "G", "T", "P"):
        if abs(n) < 1024 or unit == "P":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


class Explorer(ttk.Frame):
    def __init__(self, app, parent, root="", exts=(), server=None, win=True):
        super().__init__(parent)
        self.app, self.exts, self.win = app, tuple(exts), win
        self.path, self.rows = "", []
        self._build_toolbar()
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=4)
        self.tree = ttk.Treeview(body, columns=("size", "mtime"))
        self.tree.column("#0", width=340, anchor="w")
        self.tree.column("size", width=90, anchor="e")
        self.tree.column("mtime", width=120, anchor="e")
        self.tree.heading("#0", text="Name", anchor="w", command=lambda: self._sort("name"))
        self.tree.heading("size", text="Size", anchor="e", command=lambda: self._sort("size"))
        self.tree.heading("mtime", text="Modified", anchor="e", command=lambda: self._sort("mtime"))
        self.tbar = ttk.Scrollbar(body, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.tbar.set)
        self.grid_view = FileGrid(app, body, self._open_row, self._row_menu, self.blank_menu, exts)
        self.tree.bind("<<TreeviewOpen>>", self._pump)
        self.tree.bind("<Double-1>", self._tree_double)
        self.tree.bind("<Button-3>", self._tree_menu)
        self.info = ttk.Label(self, style="Muted.TLabel")
        self.info.pack(fill="x", padx=8, pady=(2, 4))
        self.set_mode(getattr(app, "fm_mode", "details"))
        if root:
            self.after(60, lambda: self.navigate(root))
        else:
            self.refresh_roots()

    # ── toolbar: server is global now, only navigation lives here ─────────────
    def _build_toolbar(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=4, pady=3)
        ttk.Button(top, text="←", width=3, style="Ghost.TButton",
                   command=lambda: self.navigate(self.path.rsplit("/", 1)[0] or "/")).pack(side="left")
        ttk.Button(top, text="⟳", width=3, style="Ghost.TButton",
                   command=lambda: self.navigate(self.path)).pack(side="left")
        self.roots = ttk.Combobox(top, values=[], width=22, state="readonly")
        self.roots.pack(side="left", padx=4)
        self.roots.bind("<<ComboboxSelected>>", lambda e: self.navigate(self.roots.get()))
        self.crumb = ttk.Frame(top)
        self.crumb.pack(side="left", fill="x", expand=True)
        self.entry = ttk.Entry(top)
        self.entry.pack_forget()
        self.entry.bind("<Return>", lambda e: (self._commit_entry(), self._show_crumb()))
        self.entry.bind("<Escape>", lambda e: self._show_crumb())
        ttk.Button(top, text="New", width=4, style="Ghost.TButton", command=self.mkdir).pack(side="left", padx=2)
        self.mode_btn = ttk.Button(top, text="Icons", width=6, style="Ghost.TButton",
                                   command=lambda: self.set_mode(
                                       "icons" if self._mode == "details" else "details"))
        self.mode_btn.pack(side="left")
        self.bind_all("<Control-l>", self._edit_path)
        self._crumb_mode = True

    @property
    def server(self):
        return self.app.server_of(self.app.server_var.get())

    # ── breadcrumb ⇄ entry ────────────────────────────────────────────────────
    def _show_crumb(self):
        self._crumb_mode = True
        self.entry.pack_forget()
        self.crumb.pack(side="left", fill="x", expand=True)
        for w in self.crumb.winfo_children():
            w.destroy()
        parts = [seg for seg in self.path.split("/") if seg]
        acc = ""
        for seg in parts:
            acc += "/" + seg
            ttk.Button(self.crumb, text=seg, width=min(max(len(seg), 3), 16), style="Ghost.TButton",
                       command=lambda target=acc: self.navigate(target)).pack(side="left")
            ttk.Label(self.crumb, text="›", style="Faint.TLabel").pack(side="left")

    def _edit_path(self, _e=None):
        self._crumb_mode = False
        self.crumb.pack_forget()
        self.entry.pack(side="left", fill="x", expand=True, padx=2)
        self.entry.delete(0, "end")
        self.entry.insert(0, self.path)
        self.entry.select_range(0, "end")
        self.entry.icursor("end")
        self.entry.focus_set()

    def _commit_entry(self):
        t = self.entry.get().strip()
        if t:
            self.navigate(t)

    # ── modes ─────────────────────────────────────────────────────────────────
    def set_mode(self, mode):
        self._mode = mode
        self.mode_btn.configure(text="Icons" if mode == "details" else "Details")
        self.tree.pack_forget()
        self.grid_view.pack_forget()
        if mode == "details":
            self.tbar.pack(side="right", fill="y")
            self.tree.pack(side="left", fill="both", expand=True)
            self._fill_tree_root()
        else:
            self.tbar.pack_forget()
            self.grid_view.pack(side="left", fill="both", expand=True)
        self.app.fm_mode = mode

    def activated(self):
        self.navigate(self.path) if self.path else self.refresh_roots()

    # ── data ──────────────────────────────────────────────────────────────────
    def refresh_roots(self):
        def done(rs):
            if not rs or isinstance(rs, tuple):
                return
            self.roots.configure(values=[r["path"] for r in rs])
            if not self.path:
                self.navigate(rs[0]["path"])
        self.server.call("fs.roots", None, done)

    def navigate(self, path):
        path = (path or "").strip()
        if "://" in path:
            return self.app.open_model(path, self.app.server_var.get())
        self.path = path or "/"
        self._show_crumb()
        if not self.server.up:
            return self.info.configure(text=f"{self.server.name} is offline")
        self.server.call("fs.list", {"path": self.path}, self._show)

    def _show(self, rows):
        self.rows = rows or []
        self.grid_view.set_rows(self.rows, self.path)
        self.info.configure(text=f"{len(self.rows)} items   ·   right-click: actions   ·   "
                                 f"double-click: open   ·   Ctrl+L: edit path")
        if self._mode == "details":
            self._fill_tree_root()
        else:
            self._show_crumb()

    def _sort(self, column):
        keyf = {"name": lambda r: r["name"].lower(), "size": lambda r: r["size"],
                "mtime": lambda r: r["mtime"]}[column]
        self.rows.sort(key=lambda r: (r["kind"] != "dir", keyf(r)),
                       reverse=getattr(self, "_sort_dir", ("", False))[1] == column)
        self.grid_view.set_rows(self.rows, self.path)

    # ── details mode ──────────────────────────────────────────────────────────
    def _fill_tree_root(self):
        kids = self.tree.get_children()
        if kids and self.tree.item(kids[0], "tags") and self.tree.item(kids[0], "tags")[0] == self.path:
            self.tree.item(kids[0], open=True)
            self._pump()
            return
        self.tree.delete(*self.tree.get_children())
        node = self.tree.insert("", "end", text=self.path or "/", open=True, tags=(self.path,),
                                image=icon("dir"))
        self.tree.insert(node, "end", text="...")
        self._pump()

    def _walk(self, nodes):
        for n in nodes:
            yield n
            yield from self._walk(self.tree.get_children(n))

    def _pump(self, _e=None):
        for node in list(self._walk(self.tree.get_children(""))):
            try:
                if not self.tree.item(node, "open"):
                    continue
                kids = self.tree.get_children(node)
                tags = self.tree.item(node, "tags")
                path = tags[0] if tags else None
                if path and len(kids) == 1 and self.tree.item(kids[0], "text") == "...":
                    self.tree.delete(kids[0])
                    self.server.call("fs.list", {"path": path},
                                     lambda rows, n=node, p=path: self._fill_tree(n, rows, p))
            except tk.TclError:
                return                                  # the tree was rebuilt mid-flight

    def _fill_tree(self, node, rows, path):
        try:
            if not self.tree.exists(node) or self.tree.item(node, "tags")[0] != path:
                return
            for r in rows or []:
                kid = self.tree.insert(node, "end", text=r["name"], image=icon(icon_for(r)),
                                       values=(human(r["size"]), fmt_time(r["mtime"])),
                                       tags=(join(path, r["name"]),))
                if r["kind"] == "dir":
                    self.tree.insert(kid, "end", text="...")
        except tk.TclError:
            return
        self._pump()

    def _tree_double(self, _e):
        node = self.tree.focus()
        if not node or not self.tree.item(node, "tags"):
            return
        path = self.tree.item(node, "tags")[0]
        if any(self.tree.item(k, "text") == "..." for k in self.tree.get_children(node)):
            self.tree.item(node, open=True)
            self._pump()
        elif self.tree.get_children(node):
            self.tree.item(node, open=not self.tree.item(node, "open"))
        else:
            self.app.open_file(path, self.app.server_var.get())

    def _tree_menu(self, e):
        node = self.tree.identify_row(e.y)
        if not node:
            return
        self.tree.selection_set(node)
        path = self.tree.item(node, "tags")[0]
        is_dir = bool(self.tree.get_children(node))
        self._menu_for(path, is_dir, e)

    # ── actions ───────────────────────────────────────────────────────────────
    def _open_row(self, row):
        path = join(self.path, row["name"])
        if row["kind"] == "dir":
            return self.navigate(path)
        if self.win:
            self.app.open_file(path, self.app.server_var.get())
        else:
            self.chosen(path)

    def chosen(self, path):
        self.app.open_file(path, self.app.server_var.get())

    def _row_menu(self, menu, row):
        self._menu_for_menu(menu, join(self.path, row["name"]), row["kind"] == "dir")

    def blank_menu(self, menu):
        menu.add_command(label="New folder…", command=self.mkdir)
        menu.add_command(label="Refresh", command=lambda: self.navigate(self.path))
        menu.add_command(label="Edit path (Ctrl+L)", command=self._edit_path)

    def _menu_for(self, path, is_dir, e):
        m = tk.Menu(self, tearoff=0)
        self._menu_for_menu(m, path, is_dir)
        if m.index("end") is not None:
            popup(m, e.x_root, e.y_root)

    def _menu_for_menu(self, m, path, is_dir):
        name, app = path.rsplit("/", 1)[-1], self.app
        low = name.lower()
        if is_dir:
            m.add_command(label="Open", command=lambda: self.navigate(path))
        if not is_dir and low.endswith((".yaml", ".yml")):
            m.add_command(label="Edit config", command=lambda: app.open_config(path))
            m.add_command(label="Train with this config", command=lambda: app.start_training(path))
        if not is_dir and low.endswith(WEIGHTS):
            m.add_command(label="Load as model in Tasks", command=lambda: app.open_model(path))
        if not is_dir and low.endswith(DATA):
            m.add_command(label="Annotate with current task", command=lambda: app.annotate_file(path))
        m.add_separator()
        m.add_command(label="Add to favourite models", command=lambda: self.fav(path, "models"))
        m.add_command(label="Add to favourite datasets", command=lambda: self.fav(path, "datasets"))
        m.add_command(label="Copy path", command=lambda: app.copy(path))
        m.add_separator()
        m.add_command(label="Rename…", command=lambda: self.rename(path))
        m.add_command(label="Delete…", command=lambda: self.delete(path))

    def mkdir(self):
        name = simpledialog.askstring("New folder", "Name:", parent=self)
        if name:
            self.server.call("fs.mkdir", {"path": join(self.path, name)},
                             lambda r: self.navigate(self.path))

    def rename(self, path):
        name = path.rsplit("/", 1)[-1]
        new = simpledialog.askstring("Rename", "New name:", initialvalue=name, parent=self)
        if new and new != name:
            self.server.call("fs.rename", {"path": path, "to": join(path.rsplit("/", 1)[0], new)},
                             lambda r: self.navigate(self.path or "/"))

    def delete(self, path):
        if messagebox.askyesno("Delete", f"Delete {path.rsplit('/', 1)[-1]} from the server?", parent=self):
            self.server.call("fs.delete", {"path": path}, lambda r: self.navigate(self.path or "/"))

    def fav(self, path, kind):
        self.server.call("fav.add", {"kind": kind, "item": path},
                         lambda r: self.app.toast(f"added to {kind}: {path}"))

    def server_changed(self):
        self.refresh_roots()


# ── the icon grid ─────────────────────────────────────────────────────────────────────────────
class FileGrid(tk.Canvas):
    """64 px tiles, two-line wrapped labels, selection highlight, wheel scroll, double-click and
    right-click: the spatial view, in a hundred lines of canvas."""

    def __init__(self, app, parent, on_open, on_menu, blank_menu, exts=()):
        super().__init__(parent, bd=0, highlightthickness=0)
        self.app, self.on_open, self.on_menu = app, on_open, on_menu
        self.blank_menu_fn, self.exts = blank_menu, tuple(exts)
        self.items, self.selected, self.path = [], None, ""
        self.bind("<Configure>", lambda e: self._layout())
        self.bind("<Button-1>", lambda e: self._click(e))
        self.bind("<Double-Button-1>", lambda e: self._click(e, open_it=True))
        self.bind("<Button-3>", self._right)
        self.bind("<Motion>", self._hover)
        self.bind("<Leave>", lambda e: dismiss())
        self.bind("<Enter>", lambda e: self._wheel_bind(True))
        self.bind("<Leave>", lambda e: self._wheel_bind(False))
        app.register_paintable(self)

    def repaint(self, p):
        self.configure(bg=p["surface"], highlightthickness=0)
        self._draw()

    def _wheel(self, e):
        if not self.winfo_ismapped():
            return
        self.yview_scroll(-1 if getattr(e, "num", 0) == 4 or e.delta > 0 else 1, "units")

    def _wheel_bind(self, on):
        if on:
            self.bind_all("<MouseWheel>", self._wheel)
            self.bind_all("<Button-4>", self._wheel)
            self.bind_all("<Button-5>", self._wheel)
        else:
            self.unbind_all("<MouseWheel>")
            self.unbind_all("<Button-4>")
            self.unbind_all("<Button-5>")

    def set_rows(self, rows, path):
        self.path, self.selected = path, None
        self.items = [r for r in rows
                      if not (r["kind"] == "file" and self.exts
                              and "." + r["name"].rsplit(".", 1)[-1].lower() not in self.exts)]
        self._layout()

    def _layout(self):
        cols = max(1, self.winfo_width() // CELL_W)
        for i, it in enumerate(self.items):
            it["x"], it["y"] = (i % cols) * CELL_W + 8, (i // cols) * CELL_H + 8
        rows = (len(self.items) + cols - 1) // cols if cols else 1
        self.configure(scrollregion=(0, 0, max(self.winfo_width(), cols * CELL_W),
                                     max(self.winfo_height(), rows * CELL_H + 12)))
        self._draw()

    def _draw(self):
        self.delete("all")
        p = self.app.palette()
        for it in self.items:
            x, y, sel = it.get("x", 0), it.get("y", 0), it is self.selected
            if sel:
                self.create_rectangle(x - 2, y - 2, x + CELL_W - 10, y + CELL_H - 10,
                                      fill=p["accent_soft"], outline=p["accent"], width=1, tags="bg")
            self.create_image(x + (CELL_W - 20) // 2, y + 40, image=icon(icon_for(it), 48), tags="i")
            self._wrapped(it["name"], x + (CELL_W - 20) // 2, y + 82,
                          p["text"] if not sel else p["accent"])
        if not self.items:
            self.create_text(self.winfo_width() / 2, self.winfo_height() / 2,
                             text="(empty folder)", fill=p["faint"])

    def _wrapped(self, name, cx, cy, color):
        label = name if len(name) <= 17 else name[:14] + "…"
        self.create_text(cx, cy, text=label, fill=color, font=("TkDefaultFont", 8))

    def _at(self, e):
        x, y = self.canvasx(e.x), self.canvasy(e.y)
        for it in self.items:
            if it["x"] - 2 <= x < it["x"] + CELL_W - 10 and it["y"] - 2 <= y < it["y"] + CELL_H - 10:
                return it
        return None

    def _click(self, e, open_it=False):
        it = self._at(e)
        self.selected = it
        self._draw()
        if it and open_it:
            self.on_open(it)

    def _hover(self, e):
        it = self._at(e)
        if it and (len(it["name"]) > 16 or it["kind"] == "file"):
            tooltip(f"{it['name']}\n{human(it['size'])}   {fmt_time(it['mtime'])}", e.x_root, e.y_root)
        else:
            dismiss()

    def _right(self, e):
        it = self._at(e)
        self.selected = it
        self._draw()
        m = tk.Menu(self, tearoff=0)
        if it:
            self.on_menu(m, it)
        else:
            self.blank_menu_fn()(m)
        if m.index("end") is not None:
            popup(m, e.x_root, e.y_root)


def pick(app, parent, root="", exts=(), server=None, title="Browse files on the server"):
    """Modal picker -> path string or None. The whole GUI browses through this one dialog."""
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.geometry("900x600")
    dlg.transient(parent)
    result = {"v": None}
    ex = Explorer(app, dlg, exts=exts, server=server, win=False)
    ex.set_mode("icons")
    ex.pack(fill="both", expand=True)
    if root:
        ex.after(100, lambda: ex.navigate(root))
    ex.chosen = lambda p: (result.__setitem__("v", p), dlg.destroy())
    bb = ttk.Frame(dlg)
    bb.pack(fill="x", padx=8, pady=8)

    def ok():
        sel = ex.grid_view.selected if ex._mode == "icons" else None
        if sel:
            result["v"] = join(ex.path, sel["name"])
        else:
            node = ex.tree.focus() if ex._mode == "details" else ""
            result["v"] = ex.tree.item(node, "tags")[0] if node and ex.tree.item(node, "tags") else ex.path
        dlg.destroy()
    ttk.Button(bb, text="OK", style="Accent.TButton", command=ok).pack(side="right", padx=(6, 0))
    ttk.Button(bb, text="Cancel", command=dlg.destroy).pack(side="right")
    dlg.grab_set()
    parent.wait_window(dlg)
    return result["v"]
