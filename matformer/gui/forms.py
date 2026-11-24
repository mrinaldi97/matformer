# matformer/gui/forms.py
"""Schema -> working form. The single place that draws fields: the plugin dock, the task page
(whose schema the server generates from the task's own dataclass via `task.fields`) and any future
editor all speak this one field language.

Kinds: text, int, bigint, float, sci, bool, choice, radio, json, path.
A field spec: {name, label?, kind/type, default?, doc?, options?, hint?, root?, exts?, toggle?, on?}
"""
import json
import tkinter as tk
from tkinter import ttk


def _set_entry(w, v):
    w.delete(0, "end")
    w.insert(0, "" if v is None else str(v))


def _text(app, parent, f):
    w = ttk.Entry(parent)
    if f.get("hint"):
        from matformer.gui.widgets import placeholder
        placeholder(w, f["hint"])
    return w, w.get, lambda v: _set_entry(w, v)


def _num(app, parent, f, integer=True):
    w = ttk.Spinbox(parent, from_=f.get("min", -10 ** 12), to=f.get("max", 10 ** 12),
                    increment=f.get("step", 1 if integer else 0.1))
    get = (lambda: int(float(w.get() or 0))) if integer else (lambda: float(w.get() or 0))
    return w, get, lambda v: _set_entry(w, v)


def _sci(app, parent, f):                                      # learning rates: 3e-4
    w, _, put = _text(app, parent, f)
    return w, (lambda: float(w.get() or 0)), \
        lambda v: put(f"{v:g}" if isinstance(v, float) else v)


def _bool(app, parent, f):
    v = tk.BooleanVar()
    w = ttk.Checkbutton(parent, variable=v)
    return w, v.get, lambda x: v.set(bool(x))


def _choice(app, parent, f):
    w = ttk.Combobox(parent, values=f.get("options", []), state="readonly")
    return w, w.get, lambda v: w.set(str(v))


def _radio(app, parent, f):
    w = ttk.Frame(parent)
    var = tk.StringVar()
    for o in f.get("options", []):
        ttk.Radiobutton(w, text=str(o), value=str(o), variable=var).pack(side="left", padx=(0, 10))
    return w, var.get, lambda v: var.set(str(v))


def _json(app, parent, f):                                     # lists/dicts/tuples, edited as JSON
    w, get, put = _text(app, parent, f)
    return w, (lambda: json.loads(get()) if get().strip() else None), \
        lambda v: put(json.dumps(v) if isinstance(v, (list, dict, tuple)) else v)


def _path(app, parent, f):
    holder = ttk.Frame(parent)
    w, get, put = _text(app, holder, f)
    w.pack(side="left", fill="x", expand=True)

    def pick():
        p = app.browse(f.get("root") or "", f.get("exts", ()))
        if p:
            put(p)
    ttk.Button(holder, text="...", width=3, command=pick).pack(side="left", padx=(4, 0))
    return holder, get, put


KINDS = {"text": _text, "int": lambda a, p, f: _num(a, p, f, True),
         "bigint": lambda a, p, f: _num(a, p, f, True), "float": lambda a, p, f: _num(a, p, f, False),
         "sci": _sci, "bool": _bool, "choice": _choice, "radio": _radio, "json": _json, "path": _path}


class Form(ttk.Frame):
    """Draws a list of field specs in two columns; values()/load() speak the field names.
    `run(method)` is handed to every command button the spec declares."""

    def __init__(self, app, parent, fields):
        super().__init__(parent)
        self.app = app
        self.entries, self.r = {}, 0
        self.columnconfigure(0, minsize=150)
        self.columnconfigure(2, weight=1)
        for f in fields:
            self._add(f)

    def _add(self, f):
        if f.get("section"):
            ttk.Label(self, text=f["section"].upper(), style="Head.TLabel").grid(
                row=self.r, column=0, columnspan=3, sticky="ew", pady=(8, 2))
            self.r += 1
        kind = f.get("kind") or f.get("type")
        if kind not in KINDS:
            return                                             # unknown kinds stay out of the form
        holder = ttk.Frame(self) if f.get("toggle") else None
        w, get, put = KINDS[kind](self.app, holder or self, f)
        tog = None
        if f.get("toggle"):                                    # unticked = parameter not sent
            tog = tk.BooleanVar(value=bool(f.get("on", False)))
            cb = ttk.Checkbutton(holder, variable=tog)
            w.configure(state="normal" if tog.get() else "disabled")
            tog.trace_add("write", lambda *_: w.configure(state="normal" if tog.get() else "disabled"))
            cb.pack(side="left")
            w.pack(side="left", fill="x", expand=True, padx=(4, 0))
            outer = holder
        else:
            outer = w
        ttk.Label(self, text=f.get("label", f["name"].replace("_", " ").capitalize()) + ":"
                  ).grid(row=self.r, column=0, sticky="w", padx=(0, 10), pady=2)
        outer.grid(row=self.r, column=1, columnspan=2, sticky="ew", pady=2)
        self.r += 1
        if f.get("doc"):
            ttk.Label(self, text=f["doc"], style="Muted.TLabel",
                      wraplength=320, justify="left").grid(row=self.r, column=1, columnspan=2, sticky="w")
            self.r += 1
        self.entries[f["name"]] = (get, put, tog)
        if "default" in f and f["default"] is not None:
            self.load({f["name"]: f["default"]})

    def values(self):
        out = {}
        for name, (get, _, tog) in self.entries.items():
            if tog is not None and not tog.get():
                continue
            try:
                value = get()
            except (ValueError, tk.TclError):
                continue
            if value is not None:
                out[name] = value
        return out

    def load(self, d):
        for name, (_, put, tog) in self.entries.items():
            if name in d:
                try:
                    put(d[name])
                    if tog is not None:
                        tog.set(True)
                except (ValueError, tk.TclError):
                    pass
