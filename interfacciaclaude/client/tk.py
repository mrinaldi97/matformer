"""LLM-Lab — a small tkinter demo of the Qt shell.

Everything is fake: fake servers, fake filesystem, fake training, fake metrics.
It exists to judge the UI/UX, not to do anything useful.

    python llm_lab_demo.py
"""
import random
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# ────────────────────────────────────────────────────────────── themes ──
THEMES = {
    "Light": dict(bg="#f2f2f2", panel="#fafafa", field="#ffffff", fg="#1e1e1e",
                  muted="#767676", sel="#cfe3ff", accent="#3b7ddd", border="#c9c9c9",
                  on_accent="#ffffff"),
    "Dark": dict(bg="#2b2b2b", panel="#333333", field="#3c3f41", fg="#e8e8e8",
                 muted="#9aa0a6", sel="#3f5a7d", accent="#5b8dd6", border="#4a4a4a",
                 on_accent="#ffffff"),
}


def apply_theme(root, name):
    p = THEMES[name]
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=p["bg"], foreground=p["fg"], fieldbackground=p["field"],
                bordercolor=p["border"], focuscolor=p["accent"], font=("Segoe UI", 9))
    s.configure("TFrame", background=p["bg"])
    s.configure("Panel.TFrame", background=p["panel"])
    s.configure("TLabel", background=p["bg"], foreground=p["fg"])
    s.configure("Muted.TLabel", background=p["bg"], foreground=p["muted"])
    s.configure("Title.TLabel", background=p["bg"], foreground=p["fg"], font=("Segoe UI", 10, "bold"))
    s.configure("Head.TLabel", background=p["panel"], foreground=p["fg"], font=("Segoe UI", 9, "bold"))
    s.configure("TButton", padding=(8, 4))
    s.configure("TEntry", fieldbackground=p["field"], foreground=p["fg"], insertcolor=p["fg"])
    s.configure("TCombobox", fieldbackground=p["field"], foreground=p["fg"],
                background=p["field"], arrowcolor=p["fg"])
    s.configure("TSpinbox", fieldbackground=p["field"], foreground=p["fg"], arrowcolor=p["fg"])
    s.configure("TCheckbutton", background=p["bg"], foreground=p["fg"])
    s.configure("Treeview", background=p["field"], fieldbackground=p["field"],
                foreground=p["fg"], rowheight=22, bordercolor=p["border"])
    s.configure("Treeview.Heading", background=p["panel"], foreground=p["fg"], relief="flat")
    s.map("Treeview", background=[("selected", p["accent"])],
          foreground=[("selected", p["on_accent"])])
    s.configure("TNotebook", background=p["bg"], bordercolor=p["border"])
    s.configure("TNotebook.Tab", background=p["panel"], foreground=p["fg"], padding=(12, 5))
    s.map("TNotebook.Tab", background=[("selected", p["bg"])], foreground=[("selected", p["accent"])])
    s.configure("TPanedwindow", background=p["bg"])
    s.configure("Sash", sashthickness=6, gripcount=0, background=p["bg"])
    s.configure("Vertical.TScrollbar", background=p["panel"], troughcolor=p["bg"],
                bordercolor=p["border"], arrowcolor=p["fg"])
    s.configure("Horizontal.TScrollbar", background=p["panel"], troughcolor=p["bg"],
                bordercolor=p["border"], arrowcolor=p["fg"])
    s.configure("TProgressbar", background=p["accent"], troughcolor=p["field"])
    root.option_add("*TCombobox*Listbox.background", p["field"])
    root.option_add("*TCombobox*Listbox.foreground", p["fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", p["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", p["on_accent"])
    root.option_add("*Menu.background", p["panel"])
    root.option_add("*Menu.foreground", p["fg"])
    root.option_add("*Menu.activeBackground", p["accent"])
    root.option_add("*Menu.activeForeground", p["on_accent"])


# ──────────────────────────────────────────────────────── fake backend ──
def ext_of(name):
    return ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""


def _f(size):
    return {"kind": "file", "size": size}


def _d(children):
    return {"kind": "dir", "children": children}


def make_fs(seed):
    rnd = random.Random(seed)
    return _d({
        "datasets": _d({
            "fineweb-edu": _d({f"shard_{i:05d}.jsonl": _f(rnd.randint(80, 400) * 1_000_000)
                               for i in range(6)}),
            "alpaca-cleaned": _d({"data.jsonl": _f(22_000_000), "README.md": _f(3200)}),
            "openwebtext": _d({f"part_{i:03d}.jsonl": _f(rnd.randint(50, 200) * 1_000_000)
                               for i in range(4)}),
        }),
        "configs": _d({
            "qwen3-8b-lora.yaml": _f(2140),
            "qwen3-8b-sft.yaml": _f(1890),
            "llama3-1b-pretrain.yaml": _f(3320),
            "debug-smoke.yaml": _f(640),
        }),
        "models": _d({
            "qwen3-8b": _d({"model.safetensors": _f(16_400_000_000),
                            "config.json": _f(940), "tokenizer.json": _f(11_000_000)}),
            "llama3-1b": _d({"model.safetensors": _f(2_100_000_000), "config.json": _f(820)}),
            "lora-qwen3-adapter": _d({"adapter_model.safetensors": _f(180_000_000)}),
        }),
        "runs": _d({f"run-{1000 + i}": _d({"checkpoint.pt": _f(rnd.randint(200, 900) * 1_000_000),
                                           "metrics.jsonl": _f(120_000)}) for i in range(5)}),
    })


def list_dir(tree, path):
    parts = [p for p in path.split("/") if p]
    node = tree
    for p in parts:
        node = node.get("children", {}).get(p) if node.get("kind") == "dir" else None
        if node is None:
            return []
    if node.get("kind") != "dir":
        return []
    kids = node.get("children", {})
    ordered = sorted(kids.items(), key=lambda kv: (kv[1]["kind"] != "dir", kv[0].lower()))
    return [{"name": k, "kind": v["kind"], "size": v.get("size", 0)} for k, v in ordered]


CONFIG_FIELDS = [
    {"name": "model", "label": "Base model", "type": "path", "default": "hf://Qwen/Qwen3-8B"},
    {"name": "dataset", "label": "Dataset", "type": "path", "default": "/datasets/fineweb-edu"},
    {"name": "method", "type": "choice", "options": ["lora", "qlora", "full"], "default": "lora"},
    {"name": "lr", "label": "Learning rate", "type": "text", "default": "2e-4"},
    {"name": "steps", "type": "int", "default": 2000, "min": 1, "max": 1_000_000},
    {"name": "batch", "label": "Batch size", "type": "int", "default": 8, "min": 1, "max": 1024},
    {"name": "grad_ckpt", "label": "Gradient checkpointing", "type": "bool", "default": True},
    {"name": "output", "label": "Output dir", "type": "path", "default": "/runs/run-2001"},
]

PLUGINS = [
    ("model", "Model", [
        {"name": "repo", "label": "Model / path", "type": "path", "default": "hf://Qwen/Qwen3-8B"},
        {"name": "dtype", "type": "choice", "options": ["bfloat16", "float16", "float32"],
         "default": "bfloat16"},
        {"name": "device", "type": "choice", "options": ["auto", "cuda:0", "cuda:1", "cpu"],
         "default": "auto"},
    ]),
    ("sampling", "Sampling", [
        {"name": "temperature", "type": "float", "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
        {"name": "top_p", "type": "float", "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "max_tokens", "label": "Max tokens", "type": "int", "default": 512,
         "min": 1, "max": 32768},
        {"name": "seed", "type": "int", "default": -1, "min": -1, "max": 2 ** 31 - 1},
    ]),
    ("training", "Training", [
        {"name": "steps", "type": "int", "default": 2000, "min": 1, "max": 1_000_000},
        {"name": "lr", "label": "Learning rate", "type": "text", "default": "2e-4"},
        {"name": "batch", "label": "Batch size", "type": "int", "default": 8, "min": 1, "max": 1024},
        {"name": "grad_ckpt", "label": "Gradient checkpointing", "type": "bool", "default": True},
    ]),
    ("display", "Display", [
        {"name": "highlight", "label": "Highlight uncertainty", "type": "bool", "default": True},
        {"name": "smooth", "label": "Curve smoothing", "type": "float", "default": 0.6,
         "min": 0.0, "max": 0.99, "step": 0.05},
    ]),
]


class FakeServer:
    def __init__(self, app, name, latency):
        self.app, self.name, self.up, self.latency = app, name, False, latency
        self.fs = make_fs(name)

    def start(self):
        self.app.after(random.randint(200, 1400), self._connect)

    def _connect(self):
        self.up = True
        self.app.set_led(self.name, True)
        self.app.on_connect(self.name)

    def call(self, method, params=None, cb=None):
        if not self.up:
            self.app.on_event(self.name, "error", f"{method}: not connected")
            return
        def run():
            if cb:
                cb(self._handle(method, params or {}))
        self.app.after(self.latency, run)

    def _handle(self, m, p):
        if m == "fs.list":
            return list_dir(self.fs, p.get("path", "/"))
        if m == "fs.delete":
            return {"msg": "removed"}
        if m == "fs.rename":
            return {"msg": "renamed"}
        if m == "config.read":
            vals = {f["name"]: f.get("default") for f in CONFIG_FIELDS}
            return {"fields": CONFIG_FIELDS, "values": vals}
        if m == "config.write":
            return {"msg": "config saved"}
        if m == "train.start":
            return {"run": f"run-{random.randint(2000, 9999)}"}
        if m == "train.abort":
            return {"msg": "aborting"}
        if m == "data.datasets":
            return [{"name": f"{self.name}/fineweb-edu", "shards": 6, "tokens": "1.2B", "size": "1.8 GB"},
                    {"name": f"{self.name}/alpaca-cleaned", "shards": 1, "tokens": "18M", "size": "22 MB"},
                    {"name": f"{self.name}/openwebtext", "shards": 4, "tokens": "740M", "size": "980 MB"}]
        if m == "data.models":
            return [{"name": f"{self.name}/qwen3-8b", "params": "8.2B", "format": "safetensors", "size": "16.4 GB"},
                    {"name": f"{self.name}/llama3-1b", "params": "1.2B", "format": "safetensors", "size": "2.1 GB"},
                    {"name": f"{self.name}/lora-qwen3-adapter", "params": "84M", "format": "safetensors", "size": "180 MB"}]
        return {}


# ──────────────────────────────────────────────────────────── widgets ──
class Form(ttk.Frame):
    """Schema dict -> a small grid of controls.  Types: text int float bool choice path."""

    def __init__(self, app, parent, fields, browse=None):
        super().__init__(parent)
        self.app, self.browse, self.entries = app, browse, {}
        self.columnconfigure(1, weight=1)
        for r, f in enumerate(fields):
            ttk.Label(self, text=f.get("label", f["name"].replace("_", " ").capitalize())).grid(
                row=r, column=0, sticky="w", padx=(0, 12), pady=4)
            self._add(r, f)

    def _add(self, r, f):
        t, name = f["type"], f["name"]
        if t == "bool":
            var = tk.BooleanVar(value=bool(f.get("default", False)))
            w = ttk.Checkbutton(self, variable=var)
            get, put = var.get, lambda v: var.set(bool(v))
        elif t == "choice":
            opts = f.get("options", [])
            var = tk.StringVar(value=str(f.get("default", opts[0] if opts else "")))
            w = ttk.Combobox(self, textvariable=var, values=opts, state="readonly")
            get, put = var.get, lambda v: var.set(str(v))
        elif t == "int":
            var = tk.StringVar(value=str(f.get("default", 0)))
            w = ttk.Spinbox(self, from_=f.get("min", -10 ** 9), to=f.get("max", 10 ** 9),
                            textvariable=var, increment=1)
            get, put = (lambda: int(float(var.get() or 0))), (lambda v: var.set(str(int(v))))
        elif t == "float":
            var = tk.StringVar(value=str(f.get("default", 0.0)))
            w = ttk.Spinbox(self, from_=f.get("min", -1e9), to=f.get("max", 1e9),
                            increment=f.get("step", 0.1), textvariable=var)
            get, put = (lambda: float(var.get() or 0)), (lambda v: var.set(str(float(v))))
        elif t == "path":
            var = tk.StringVar(value=str(f.get("default", "")))
            holder = ttk.Frame(self)
            e = ttk.Entry(holder, textvariable=var)
            e.pack(side="left", fill="x", expand=True)
            ttk.Button(holder, text="…", width=3,
                       command=lambda f=f, var=var: self._pick(f, var)).pack(side="left", padx=(4, 0))
            w, get, put = holder, var.get, lambda v: var.set(str(v))
        else:
            var = tk.StringVar(value=str(f.get("default", "")))
            w = ttk.Entry(self, textvariable=var)
            get, put = var.get, lambda v: var.set(str(v))
        w.grid(row=r, column=1, sticky="ew", pady=4)
        self.entries[name] = (w, get, put)

    def _pick(self, f, var):
        p = self.browse and self.browse(f)
        if p:
            var.set(p)

    def values(self):
        return {n: g() for n, (_, g, _) in self.entries.items()}

    def load(self, d):
        for n, (_, _, put) in self.entries.items():
            if n in d:
                try:
                    put(d[n])
                except Exception:
                    pass


ICONS = {".yaml": "📝", ".yml": "📝", ".json": "🧾", ".jsonl": "📊",
         ".pt": "🧠", ".safetensors": "🧠", ".csv": "📊", ".md": "📄"}
DIR_ICON = "📁"


class IconView(tk.Frame):
    """The Qt QListView-in-IconMode stand-in: a wrapping grid of clickable tiles."""

    CELL_W, CELL_H = 118, 92

    def __init__(self, app, parent, on_open=None, on_menu=None):
        super().__init__(parent, bd=0, highlightthickness=0)
        self.app, self.on_open, self.on_menu = app, on_open, on_menu
        self.items, self.selected, self._cols = [], None, 6
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.holder = tk.Frame(self.canvas, bd=0, highlightthickness=0)
        self._win = self.canvas.create_window((0, 0), window=self.holder, anchor="nw")
        self.holder.bind("<Configure>",
                         lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", lambda e: self.select(None))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind_all(seq, self._wheel, add="+")
        app.register_tk(self.canvas, "canvas")
        app.register_tk(self.holder, "canvas")

    def restyle(self):
        p = self.app.palette()
        self.canvas.configure(bg=p["field"])
        self.holder.configure(bg=p["field"])
        rows = [it["row"] for it in self.items]
        self.set_items(rows)

    def _wheel(self, e):
        if not self.winfo_ismapped():
            return
        if getattr(e, "num", None) == 4:
            d = -1
        elif getattr(e, "num", None) == 5:
            d = 1
        else:
            d = -1 if e.delta > 0 else 1
        self.canvas.yview_scroll(d, "units")

    def _on_resize(self, e):
        self.canvas.itemconfigure(self._win, width=e.width)
        cols = max(1, e.width // self.CELL_W)
        if cols != self._cols:
            self._cols = cols
            self._relayout()

    def set_items(self, rows):
        for w in self.holder.winfo_children():
            w.destroy()
        self.items, self.selected = [], None
        p = self.app.palette()
        for r in rows:
            tile = tk.Frame(self.holder, bg=p["panel"], bd=0,
                            highlightthickness=1, highlightbackground=p["border"])
            icon = DIR_ICON if r["kind"] == "dir" else ICONS.get(ext_of(r["name"]), "📄")
            il = tk.Label(tile, text=icon, bg=p["panel"], fg=p["fg"],
                          font=("Segoe UI Emoji", 20))
            il.pack(pady=(10, 0))
            nl = tk.Label(tile, text=r["name"], bg=p["panel"], fg=p["fg"],
                          font=("Segoe UI", 8), wraplength=self.CELL_W - 16, justify="center")
            nl.pack(padx=2, pady=(3, 10), fill="x")
            data = {"row": r, "tile": tile, "labels": (il, nl)}
            for w in (tile, il, nl):
                w.bind("<Button-1>", lambda e, d=data: self.select(d))
                w.bind("<Double-Button-1>",
                       lambda e, d=data: self.on_open and self.on_open(d["row"]))
                w.bind("<Button-3>", lambda e, d=data: self.on_menu and self.on_menu(e, d["row"]))
                w.bind("<MouseWheel>", self._wheel)
            self.items.append(data)
        self._relayout()

    def select(self, d):
        p = self.app.palette()
        for it in self.items:
            on = it is d
            bg = p["sel"] if on else p["panel"]
            it["tile"].configure(bg=bg,
                                 highlightbackground=p["accent"] if on else p["border"])
            for lbl in it["labels"]:
                lbl.configure(bg=bg)
        self.selected = d

    def _relayout(self):
        cols = max(1, self._cols)
        for c in range(cols):
            self.holder.columnconfigure(c, weight=1, minsize=self.CELL_W)
        for i, it in enumerate(self.items):
            it["tile"].grid(row=i // cols, column=i % cols, padx=3, pady=3, sticky="nsew")

    def clear(self):
        self.set_items([])


class Explorer(ttk.Frame):
    """Remote file browser.  Used as a page, embedded in ConfigPage, and as a modal picker."""

    def __init__(self, app, parent, root="/", exts=(), server=None, win=None):
        super().__init__(parent)
        self.app, self.root, self.exts = app, root, tuple(exts)
        self.path, self.win, self.on_chosen = root, win, None

        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="Server:").pack(side="left")
        self.srv = ttk.Combobox(top, values=app.server_names, state="readonly", width=10)
        self.srv.set(server or app.server_names[0])
        self.srv.pack(side="left", padx=(4, 8))
        self.srv.bind("<<ComboboxSelected>>", lambda e: self.go(self.root))
        ttk.Button(top, text="⬆ Up", width=6,
                   command=lambda: self.go(self.path.rsplit("/", 1)[0] or "/")).pack(side="left", padx=(0, 8))
        self.entry = ttk.Entry(top)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self.enter())

        self.view = IconView(app, self, on_open=self.open_item, on_menu=self.menu)
        self.view.pack(fill="both", expand=True, padx=6)

        self.info = ttk.Label(self, text="", style="Muted.TLabel")
        self.info.pack(fill="x", padx=8, pady=(0, 4))

    # ── navigation ──────────────────────────────────────────────────────
    @property
    def server(self):
        return self.app.servers[self.srv.get()]

    def join(self, name):
        return self.path.rstrip("/") + "/" + name

    def activated(self):
        self.go(self.path)

    def go(self, path):
        self.path = path or "/"
        self.entry.delete(0, "end")
        self.entry.insert(0, self.path)
        srv = self.server
        if not srv.up:
            self.view.clear()
            self.info.configure(text=f"{srv.name} is offline")
            return
        srv.call("fs.list", {"path": self.path}, self.fill)

    def fill(self, rows):
        out = [r for r in rows
               if not (r["kind"] == "file" and self.exts and ext_of(r["name"]) not in self.exts)]
        self.view.set_items(out)
        self.info.configure(text=f"{len(out)} items")

    def enter(self):
        t = self.entry.get().strip()
        if "://" in t:
            if self.on_chosen:
                self.on_chosen(t)
        else:
            self.go(t)

    def open_item(self, row):
        path = self.join(row["name"])
        if row["kind"] == "dir":
            if self.win and self.win.looks_like_dataset(path):
                return self.win.open_dataset(path, self.srv.get())
            return self.go(path)
        ext = ext_of(row["name"])
        if self.win:
            if ext in (".yaml", ".yml"):
                return self.win.open_config(path, self.srv.get())
            if ext in (".pt", ".safetensors"):
                return self.win.open_model(path, self.srv.get())
            if ext == ".jsonl":
                return self.win.open_dataset(path, self.srv.get())
        if self.on_chosen:
            self.on_chosen(path)

    # ── context menu ────────────────────────────────────────────────────
    def menu(self, e, row):
        path = self.join(row["name"])
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Rename…", command=lambda: self.rename(path))
        m.add_command(label="Delete…", command=lambda: self.delete(path))
        m.add_separator()
        m.add_command(label="Copy path", command=lambda: self.app.log(f"copied {path}"))
        if row["kind"] == "file" and ext_of(row["name"]) in (".yaml", ".yml") and self.win:
            m.add_separator()
            m.add_command(label="Edit in config editor",
                          command=lambda: self.win.open_config(path, self.srv.get()))
            m.add_command(label="Train with this config",
                          command=lambda: self.win.start_training(path, {}, self.srv.get()))
        try:
            m.tk_popup(e.x_root, e.y_root)
        finally:
            m.grab_release()

    def rename(self, path):
        name = path.rsplit("/", 1)[-1]
        new = simpledialog.askstring("Rename", "New name:", initialvalue=name, parent=self)
        if not new or new == name:
            return
        new_path = path.rsplit("/", 1)[0].rstrip("/") + "/" + new
        self.server.call("fs.rename", {"path": path, "to": new_path},
                         lambda r: self.app.log(f"renamed to {new}"))
        self.go(self.path)

    def delete(self, path):
        name = path.rsplit("/", 1)[-1]
        if not messagebox.askyesno("Delete", f"Delete {name}?", parent=self):
            return
        self.server.call("fs.delete", {"path": path}, lambda r: self.app.log(f"deleted {name}"))
        self.go(self.path)

    def current(self):
        sel = self.view.selected
        if sel:
            return self.join(sel["row"]["name"])
        t = self.entry.get()
        return t if "://" in t else self.path


def pick(app, parent, server, root="/", exts=()):
    """Modal file picker -> path/URI string or None."""
    dlg = tk.Toplevel(parent)
    dlg.title("Browse files on the server")
    dlg.geometry("780x500")
    dlg.transient(parent)
    result = {"v": None}

    bb = ttk.Frame(dlg)
    bb.pack(side="bottom", fill="x", padx=8, pady=8)
    ex = Explorer(app, dlg, root, exts, server)
    ex.pack(fill="both", expand=True)

    def ok():
        result["v"] = ex.current()
        dlg.destroy()

    ex.on_chosen = lambda p: (result.__setitem__("v", p), dlg.destroy())
    ttk.Button(bb, text="OK", command=ok).pack(side="right", padx=(6, 0))
    ttk.Button(bb, text="Cancel", command=dlg.destroy).pack(side="right")
    ex.after(60, ex.activated)
    dlg.grab_set()
    parent.wait_window(dlg)
    return result["v"]


# ─────────────────────────────────────────────────────────────── pages ──
class Page(ttk.Frame):
    title = "Page"

    def __init__(self, app, parent):
        super().__init__(parent)
        self.app = app

    def activated(self):
        pass


class ChatPage(Page):
    title = "Chat"

    def __init__(self, app, parent):
        super().__init__(app, parent)
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Run on server:").pack(side="left")
        self.target = ttk.Combobox(top, values=app.server_names, state="readonly", width=12)
        self.target.set(app.server_names[0])
        self.target.pack(side="left", padx=8)
        ttk.Label(top, text="(everything below is simulated)", style="Muted.TLabel").pack(side="left")

        self.view = tk.Text(self, wrap="word", bd=0, relief="flat", padx=10, pady=8,
                            highlightthickness=1, state="disabled")
        self.view.pack(fill="both", expand=True, padx=10)
        self.view.tag_configure("you", foreground="#3b7ddd", font=("Segoe UI", 9, "bold"))
        self.view.tag_configure("sys", foreground="#8a8a8a")
        app.register_tk(self.view, "text")

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=10)
        self.box = tk.Text(bottom, height=3, bd=0, relief="flat", padx=8, pady=6,
                           highlightthickness=1)
        self.box.pack(side="left", fill="both", expand=True)
        self.box.bind("<Control-Return>", lambda e: (self.send(), "break"))
        ttk.Button(bottom, text="Send", command=self.send).pack(side="left", padx=(8, 0), fill="y")
        app.register_tk(self.box, "text")

        self.busy = False
        self._put("Ready.  Type a prompt and press Ctrl+Enter.\n\n", "sys")

    def _put(self, text, tag=None):
        self.view.configure(state="normal")
        self.view.insert("end", text, tag or "")
        self.view.see("end")
        self.view.configure(state="disabled")

    def send(self):
        text = self.box.get("1.0", "end").strip()
        if not text or self.busy:
            return
        self.box.delete("1.0", "end")
        self._put(f"you> {text}\n", "you")
        self._put("model> ", "sys")
        self.busy = True
        self._stream(0)

    def stop(self):
        self.busy = False

    def _stream(self, n):
        if not self.busy:
            return
        if n > 45:
            self._put("\n\n■ stopped · 46 tokens · 61.8 tok/s\n\n", "sys")
            self.busy = False
            return
        word = random.choice([" The", " model", " emits", " tokens", " one", " at", " a", " time,",
                              " streaming", " them", " into", " the", " transcript."])
        self._put(word)
        self.after(random.randint(18, 55), lambda: self._stream(n + 1))


class FilesPage(Page):
    title = "Files"

    def __init__(self, app, parent):
        super().__init__(app, parent)
        self.ex = Explorer(app, self, "/", win=app)
        self.ex.pack(fill="both", expand=True)

    def activated(self):
        self.ex.activated()


class ConfigPage(Page):
    title = "Config editor"

    def __init__(self, app, parent):
        super().__init__(app, parent)
        pan = ttk.PanedWindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=6, pady=6)

        self.ex = Explorer(app, pan, "/configs", (".yaml", ".yml"))
        self.ex.on_chosen = self.open
        pan.add(self.ex, weight=1)

        right = ttk.Frame(pan)
        pan.add(right, weight=2)
        self.title = ttk.Label(right, text="Double-click a config file on the left.",
                               style="Title.TLabel")
        self.title.pack(anchor="w", padx=10, pady=10)
        self.body = ttk.Frame(right)
        self.body.pack(fill="both", expand=True, padx=10)

        btns = ttk.Frame(right)
        btns.pack(fill="x", padx=10, pady=10)
        self.save = ttk.Button(btns, text="Save to server", state="disabled", command=self.write)
        self.save.pack(side="left")
        self.train = ttk.Button(btns, text="Train this config", state="disabled", command=self._train)
        self.train.pack(side="left", padx=8)

        self.form, self.path = None, None
        self.after(120, lambda: pan.sashpos(0, 380))

    def activated(self):
        self.ex.activated()

    def open(self, path, server=None):
        if server:
            self.ex.srv.set(server)
        self.path = path
        self.ex.server.call("config.read", {"path": path}, self._edit)

    def _edit(self, r):
        for w in self.body.winfo_children():
            w.destroy()
        self.form = Form(self.app, self.body, r["fields"], browse=self.app.browse)
        self.form.pack(fill="x")
        self.form.load(r["values"])
        self.title.configure(text=f"{self.path}   ·   on {self.ex.srv.get()}")
        self.save.configure(state="normal")
        self.train.configure(state="normal")

    def write(self):
        if not self.form:
            return
        self.ex.server.call("config.write",
                            {"path": self.path, "values": self.form.values()},
                            lambda r: self.app.log(f"{self.path}: {r['msg']}"))

    def _train(self):
        if self.path:
            self.app.start_training(self.path,
                                    self.form.values() if self.form else {},
                                    self.ex.srv.get())


class TrainingPage(Page):
    title = "Training"

    def __init__(self, app, parent):
        super().__init__(app, parent)
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)
        self.head = ttk.Label(top, text="No run started from this page yet.", style="Title.TLabel")
        self.head.pack(side="left")
        self.plots_btn = ttk.Button(top, text="Open in Plots", state="disabled",
                                    command=lambda: self.app.show_page("plots"))
        self.plots_btn.pack(side="right")
        self.abort = ttk.Button(top, text="Abort run", state="disabled", command=self._abort)
        self.abort.pack(side="right", padx=8)

        self.log = tk.Text(self, wrap="none", bd=0, relief="flat", padx=10, pady=8,
                           highlightthickness=1, state="disabled",
                           font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        app.register_tk(self.log, "text")
        self.run = None
        self._say("Training page ready.\n"
                  "  • Double-click a YAML in the Files page.\n"
                  "  • Or press 'Train this config' in the Config editor.\n")

    def _say(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self, config_path, values=None, server=None):
        srv = server or self.app.server_names[0]
        self.run = None
        self.head.configure(text=f"Starting {config_path} on {srv}…")
        self.abort.configure(state="normal")
        self.plots_btn.configure(state="disabled")
        self._say(f"\n→ starting training with {config_path} on {srv}\n")
        self.app.servers[srv].call("train.start",
                                   {"config": config_path, "values": values or {}},
                                   self._started)
        self._step = 0
        self.after(900, self._progress)

    def _started(self, r):
        self.run = r.get("run")
        self.head.configure(text=f"Run {self.run}")

    def _progress(self):
        if not self.run or self.abort.instate(["disabled"]):
            return
        self._step += random.randint(5, 25)
        if self._step > 400:
            self._say(f"[run] {self.run}  finished  400/400\n")
            self.abort.configure(state="disabled")
            self.plots_btn.configure(state="normal")
            return
        loss = 2.6 * (0.985 ** (self._step / 4)) + random.random() * 0.03
        self._say(f"[run] {self.run}  step {self._step:>4}/400  loss={loss:.4f}  "
                  f"lr=2.0e-4  {self._step / 400:5.1%}\n")
        self.after(random.randint(250, 700), self._progress)

    def _abort(self):
        if self.run:
            self._say(f"→ requested abort of {self.run}\n")
        self.abort.configure(state="disabled")
        self.plots_btn.configure(state="normal")


class PlotsPage(Page):
    title = "Plots"

    def __init__(self, app, parent):
        super().__init__(app, parent)
        self.runs, self.ticks = {}, {}
        names = ["local/run-1000", "local/run-1001", "gpu-01/run-1002",
                 "gpu-01/run-1003", "gpu-02/run-1004"]
        rnd = random.Random(7)
        for i, n in enumerate(names):
            xs = list(range(0, 400, 4))
            base = 2.9 - 0.25 * i
            ys = [base * (0.985 ** (x / 4)) + 0.15 + rnd.random() * 0.05 for x in xs]
            self.runs[n] = (xs, ys)
            self.ticks[n] = i < 3

        pan = ttk.PanedWindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=6, pady=6)
        side = ttk.Frame(pan)
        pan.add(side, weight=0)
        right = ttk.Frame(pan)
        pan.add(right, weight=1)

        ttk.Label(side, text="Runs", style="Head.TLabel").pack(anchor="w", padx=8, pady=(8, 4))
        for n in self.runs:
            var = tk.BooleanVar(value=self.ticks[n])
            cb = ttk.Checkbutton(side, text=n, variable=var,
                                 command=lambda n=n, v=var: self._toggle(n, v.get()))
            cb.pack(anchor="w", padx=12)
        ttk.Separator(side).pack(fill="x", pady=8, padx=8)
        row = ttk.Frame(side)
        row.pack(fill="x", padx=8)
        ttk.Label(row, text="Smooth").pack(side="left")
        self.smooth = ttk.Scale(row, from_=0, to=0.95, value=0.5,
                                command=lambda v: self.draw())
        self.smooth.pack(side="left", fill="x", expand=True, padx=6)
        self.logy = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Log-Y", variable=self.logy,
                        command=self.draw).pack(anchor="w", padx=12, pady=(6, 0))
        ttk.Button(side, text="+ Add plot…",
                   command=lambda: self.app.log("(demo) add derived plot")).pack(
            fill="x", padx=8, pady=(12, 2))
        ttk.Button(side, text="Export all…",
                   command=lambda: self.app.log("(demo) exported")).pack(fill="x", padx=8, pady=2)

        self.canvas = tk.Canvas(right, bd=0, highlightthickness=1)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        app.register_tk(self.canvas, "canvas")
        self.after(120, lambda: pan.sashpos(0, 230))

    def _toggle(self, name, on):
        self.ticks[name] = on
        self.draw()

    def _ema(self, ys, a):
        out, s = [], None
        for y in ys:
            s = y if s is None else a * s + (1 - a) * y
            out.append(s)
        return out

    def draw(self):
        c, p = self.canvas, self.app.palette()
        c.delete("all")
        W, H = c.winfo_width(), c.winfo_height()
        if W < 20 or H < 20:
            return
        L, R, T, B = 60, 20, 30, 40
        c.create_rectangle(L, T, W - R, H - B, outline=p["border"])
        c.create_text(L, 16, text="loss", anchor="w", fill=p["fg"], font=("Segoe UI", 9, "bold"))
        for i in range(5):
            y = T + (H - T - B) * i / 4
            c.create_line(L, y, W - R, y, fill=p["border"])
            c.create_text(L - 8, y, text=f"{2.0 - 0.5 * i:.1f}", anchor="e",
                          fill=p["muted"], font=("Segoe UI", 8))
        colors = ["#3b7ddd", "#e06c3a", "#2aa02a", "#b04fd0", "#d0a02a"]
        a = float(self.smooth.get())
        k = 0
        for i, (name, (xs, ys)) in enumerate(self.runs.items()):
            if not self.ticks.get(name):
                continue
            yy = self._ema(ys, a)
            pts = []
            for x, y in zip(xs, yy):
                if self.logy.get():
                    import math
                    y = math.log10(max(y, 1e-6))
                px = L + (W - L - R) * (x / max(1, xs[-1]))
                py = T + (H - T - B) * (1 - min(1.0, max(0.0, (y - 0.1) / 2.9)))
                pts += [px, py]
            if len(pts) >= 4:
                c.create_line(*pts, fill=colors[k % len(colors)], width=2, smooth=True)
                c.create_text(W - R - 6, T + 12 + k * 16, text=name, anchor="e",
                              fill=colors[k % len(colors)], font=("Segoe UI", 8))
                k += 1
        if k == 0:
            c.create_text(W / 2, H / 2, text="No runs ticked", fill=p["muted"])


class TablePage(Page):
    def __init__(self, app, parent, title, columns, call):
        super().__init__(app, parent)
        self.columns, self.call = columns, call
        ttk.Label(self, text=title, style="Title.TLabel").pack(anchor="w", padx=12, pady=10)
        head = ["server"] + columns
        self.tree = ttk.Treeview(self, columns=head, show="headings")
        for c in head:
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=170, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def activated(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for s in self.app.servers.values():
            if s.up:
                s.call(self.call, {}, lambda rows, s=s: self.add(s.name, rows))

    def add(self, server, rows):
        for r in rows:
            self.tree.insert("", "end", values=[server] + [r.get(c, "") for c in self.columns])


class DatasetsPage(TablePage):
    title = "Datasets"

    def __init__(self, app, parent):
        super().__init__(app, parent, "Datasets", ["name", "shards", "tokens", "size"],
                         "data.datasets")


class ModelsPage(TablePage):
    title = "Models"

    def __init__(self, app, parent):
        super().__init__(app, parent, "Models", ["name", "params", "format", "size"],
                         "data.models")


class LogPage(Page):
    title = "Log"

    def __init__(self, app, parent):
        super().__init__(app, parent)
        self.text = tk.Text(self, wrap="none", bd=0, relief="flat", padx=10, pady=8,
                            highlightthickness=1, state="disabled", font=("Consolas", 9))
        self.text.pack(fill="both", expand=True, padx=10, pady=10)
        app.register_tk(self.text, "text")
        self.app.log_view = self.text

    def write(self, line):
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")


# ───────────────────────────────────────────────────────────────── app ──
PAGES = [
    ("Work", [("chat", ChatPage), ("files", FilesPage),
              ("config", ConfigPage), ("training", TrainingPage)]),
    ("Data", [("datasets", DatasetsPage), ("models", ModelsPage)]),
    ("Analysis", [("plots", PlotsPage), ("log", LogPage)]),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLM-Lab — tkinter demo")
        self.geometry("1400x860")
        self.theme_name = "Light"
        self._tk_widgets = []
        self._restylers = []
        apply_theme(self, self.theme_name)

        self.servers = {"local": FakeServer(self, "local", 120),
                        "gpu-01": FakeServer(self, "gpu-01", 260),
                        "gpu-02": FakeServer(self, "gpu-02", 460)}
        self.server_names = list(self.servers)

        self._build_ui()
        self.show_page("chat")
        for s in self.servers.values():
            s.start()
        self.after(1200, self._sim_jobs)

    # ── theme plumbing ──────────────────────────────────────────────────
    def palette(self):
        return THEMES[self.theme_name]

    def register_tk(self, w, kind):
        self._tk_widgets.append((w, kind))
        self._paint_tk(w, kind)

    def _paint_tk(self, w, kind):
        p = self.palette()
        try:
            if kind == "text":
                w.configure(bg=p["field"], fg=p["fg"], insertbackground=p["fg"],
                            highlightbackground=p["border"],
                            selectbackground=p["accent"], selectforeground=p["on_accent"])
            elif kind == "canvas":
                w.configure(bg=p["field"], highlightbackground=p["border"])
        except tk.TclError:
            pass

    def toggle_theme(self):
        self.theme_name = "Dark" if self.theme_name == "Light" else "Light"
        apply_theme(self, self.theme_name)
        for w, kind in self._tk_widgets:
            self._paint_tk(w, kind)
        for fn in self._restylers:
            try:
                fn()
            except Exception:
                pass
        self.log(f"theme: {self.theme_name}")

    # ── UI construction ─────────────────────────────────────────────────
    def _build_ui(self):
        menubar = tk.Menu(self)
        self.configure(menu=menubar)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Connect all", command=self.connect_all)
        filem.add_command(label="Save settings…", command=lambda: self.log("settings saved"))
        filem.add_command(label="Load settings…", command=lambda: self.log("settings loaded"))
        filem.add_separator()
        filem.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filem)

        runm = tk.Menu(menubar, tearoff=0)
        runm.add_command(label="Send prompt", command=lambda: self.pages["chat"].send())
        runm.add_command(label="Stop generation", command=lambda: self.pages["chat"].stop())
        runm.add_command(label="Clear transcript", command=self._clear_chat)
        menubar.add_cascade(label="Run", menu=runm)

        viewm = tk.Menu(menubar, tearoff=0)
        viewm.add_command(label="Toggle theme", command=self.toggle_theme)
        viewm.add_command(label="Refresh page", command=lambda: self.pages[self.current].activated())
        menubar.add_cascade(label="View", menu=viewm)

        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="About", command=lambda: messagebox.showinfo(
            "LLM-Lab", "LLM-Lab — proof of concept.\nGUI and compute are separate processes.\n\n"
                       "(tkinter demo — everything is simulated)"))
        menubar.add_cascade(label="Help", menu=helpm)

        # toolbar
        tb = ttk.Frame(self)
        tb.pack(fill="x", padx=6, pady=(6, 0))
        for text, cmd in (("Connect", self.connect_all),
                          ("Send", lambda: self.pages["chat"].send()),
                          ("Stop", lambda: self.pages["chat"].stop()),
                          ("Refresh", lambda: self.pages[self.current].activated()),
                          ("Theme", self.toggle_theme)):
            ttk.Button(tb, text=text, command=cmd).pack(side="left", padx=(0, 4))

        # main area
        outer = ttk.PanedWindow(self, orient="vertical")
        outer.pack(fill="both", expand=True, padx=6, pady=6)
        main = ttk.PanedWindow(outer, orient="horizontal")
        jobs_frame = ttk.Frame(outer)
        outer.add(main, weight=5)
        outer.add(jobs_frame, weight=1)

        # navigator
        nav = ttk.Frame(main)
        main.add(nav, weight=0)
        ttk.Label(nav, text="Navigator", style="Head.TLabel").pack(fill="x", padx=6, pady=4)
        self.nav = ttk.Treeview(nav, show="tree", selectmode="browse", height=20)
        self.nav.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.nav.column("#0", width=170)
        self.nav.bind("<<TreeviewSelect>>", self._on_nav)

        # centre stack
        centre = ttk.Frame(main)
        main.add(centre, weight=1)
        self.stack = ttk.Frame(centre)
        self.stack.pack(fill="both", expand=True)
        self.pages = {}
        for _, items in PAGES:
            for name, cls in items:
                w = cls(self, self.stack)
                w.place(relx=0, rely=0, relwidth=1, relheight=1)
                self.pages[name] = w
        for group, items in PAGES:
            g = self.nav.insert("", "end", text=group, open=True)
            for name, cls in items:
                self.nav.insert(g, "end", iid=name, text=cls.title)
        self.current = "chat"

        # right dock: plugin forms
        right = ttk.Frame(main)
        main.add(right, weight=0)
        ttk.Label(right, text="Parameters", style="Head.TLabel").pack(fill="x", padx=6, pady=4)
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.forms = {}
        for pid, title, fields in PLUGINS:
            holder = ttk.Frame(nb, padding=10)
            nb.add(holder, text=title)
            f = Form(self, holder, fields, browse=self.browse)
            f.pack(fill="x")
            self.forms[pid] = f

        # jobs panel
        head = ttk.Frame(jobs_frame)
        head.pack(fill="x", padx=6, pady=4)
        ttk.Label(head, text="Jobs — all servers", style="Head.TLabel").pack(side="left")
        self.jobs_tree = ttk.Treeview(jobs_frame,
                                      columns=("server", "id", "kind", "name", "status", "progress"),
                                      show="headings", height=6)
        for c, w in (("server", 90), ("id", 90), ("kind", 100), ("name", 220),
                     ("status", 100), ("progress", 100)):
            self.jobs_tree.heading(c, text=c.capitalize())
            self.jobs_tree.column(c, width=w, anchor="w")
        self.jobs_tree.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # status bar
        self.status = ttk.Frame(self)
        self.status.pack(fill="x", side="bottom")
        self.status_msg = ttk.Label(self.status, text="Ready", style="Muted.TLabel")
        self.status_msg.pack(side="left", padx=10)
        self.leds = {}
        for n in self.server_names:
            lbl = ttk.Label(self.status, text=f"● {n}")
            lbl.pack(side="right", padx=8)
            self.leds[n] = lbl
        self.set_led_all(False)

        self.after(150, lambda: main.sashpos(0, 190))
        self.after(160, lambda: main.sashpos(1, max(500, main.winfo_width() - 420)))
        self.after(170, lambda: outer.sashpos(0, max(400, outer.winfo_height() - 180)))

    # ── page handling ───────────────────────────────────────────────────
    def _on_nav(self, _=None):
        sel = self.nav.selection()
        if sel and sel[0] in self.pages:
            self.show_page(sel[0])

    def show_page(self, name):
        if name not in self.pages:
            return
        self.current = name
        self.pages[name].tkraise()
        self.pages[name].activated()
        if self.nav.selection() != (name,):
            self.nav.selection_set(name)

    # ── navigation helpers used by the explorer ─────────────────────────
    @staticmethod
    def looks_like_dataset(path):
        parent = path.rstrip("/").rsplit("/", 1)[0]
        return parent in ("/datasets", "/datasets/")

    def open_config(self, path, server=None):
        self.show_page("config")
        self.pages["config"].open(path, server)

    def open_model(self, path, server=None):
        self.show_page("models")
        self.log(f"opening model folder for {path}")

    def open_dataset(self, path, server=None):
        self.show_page("datasets")
        self.log(f"opening dataset {path}")

    def start_training(self, config_path, values=None, server=None):
        self.show_page("training")
        self.pages["training"].start(config_path, values, server)

    # ── misc ────────────────────────────────────────────────────────────
    def browse(self, field):
        srv = self.pages["chat"].target.get()
        return pick(self, self, srv, field.get("root", "/"), field.get("exts", ()))

    def connect_all(self):
        for s in self.servers.values():
            s.start()

    def set_led(self, name, up):
        color = "#2aa02a" if up else "#cc2222"
        self.leds[name].configure(text=f"● {name}", foreground=color)

    def set_led_all(self, up):
        for n in self.server_names:
            self.set_led(n, up)

    def on_connect(self, name):
        self.log(f"{name}: connected")
        self.pages[self.current].activated()

    def on_event(self, server, name, data):
        if name == "error":
            self.log(f"{server}: ERROR {data}")
            self.status_msg.configure(text=f"{server}: {data}")

    def log(self, msg):
        line = f"{msg}"
        self.status_msg.configure(text=line)
        lv = getattr(self, "log_view", None)
        if lv is not None:
            lv.configure(state="normal")
            lv.insert("end", line + "\n")
            lv.see("end")
            lv.configure(state="disabled")

    def _clear_chat(self):
        c = self.pages["chat"]
        c.view.configure(state="normal")
        c.view.delete("1.0", "end")
        c.view.configure(state="disabled")

    def _sim_jobs(self):
        up = [s for s in self.servers.values() if s.up]
        if up and random.random() < 0.45:
            srv = random.choice(up)
            jid = f"job-{random.randint(100, 999)}"
            iid = f"{srv.name}/{jid}"
            if self.jobs_tree.exists(iid):
                vals = list(self.jobs_tree.item(iid, "values"))
                done = random.random() < 0.25
                vals[4] = "done" if done else "running"
                pct = 100 if done else min(99, int(str(vals[5]).rstrip("%")) + random.randint(2, 12))
                vals[5] = f"{pct}%"
                self.jobs_tree.item(iid, values=vals)
            else:
                kind = random.choice(["training", "eval", "tokenize", "export"])
                self.jobs_tree.insert("", "end", iid=iid,
                                      values=[srv.name, jid, kind,
                                              f"{kind}-{random.randint(1, 99)}",
                                              "running", "0%"])
        self.after(1400, self._sim_jobs)


if __name__ == "__main__":
    App().mainloop()
