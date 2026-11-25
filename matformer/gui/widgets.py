# matformer/gui/widgets.py
"""Shared visual atoms, all following theme.py's design system:
flat surfaces, hairline separators, 26px rows, right-aligned numerics,
status signalled by a colored dot + word (never whole-row paint), text progress bars, tooltips on
truncated content, auto-hiding slim scrollbars, and a Page base class that gives every screen the
same header (title + subtitle + right-aligned actions).

    Console      monospace text well (read-only or editor), copy/export always available
    TableList    the table: zebra, sortable, upsert, empty-state, cell tooltips
    ScrollBox    scrollable holder for forms
    Page         header + body, the skeleton every screen shares
    popup()      one context menu at a time, closes on outside click
    tooltip()    hover captions (truncated cells, token detail)
    show_payload Table payload -> table, prose -> console — the same rule the TUI applies to
                 str(Table), so one command feeds both worlds.
"""
import tkinter as tk
from tkinter import ttk

from matformer.gui import theme
from matformer.gui.icons import icon, icon_for    # noqa: F401  (re-exported: one icons API)

RIGHT_COLUMNS = {"size", "docs", "step", "steps", "progress", "tokens", "chunks", "count",
                 "mem_used_mib", "mem_total_mib", "util", "version", "index"}
# Columns that are an identifier or a small number: they never need the room a sentence needs.
NARROW_COLUMNS = {"key", "id", "#", "n", "position", "row", "num", "number", "len", "length",
                  "epoch", "layer", "head", "index", "k", "rank", "step", "steps", "docs",
                  "tokens", "chunks", "count", "version", "exit", "p", "prob", "probability",
                  "util", "port", "at"}
TIME_COLUMNS = {"started", "mtime", "date", "time"}


def _width(column, sample_value=""):
    """A starting width for a column: short for ids and numbers, roomier for words."""
    name = (column or "").lower()
    if name in NARROW_COLUMNS:
        return 58
    if name in RIGHT_COLUMNS:
        return 82
    return max(90, min(420, 48 + 8 * max(len(column), len(str(sample_value)) // 2)))


def fmt_time(value):
    text = str(value or "")
    return text.replace("T", " ")[:16] if len(text) >= 16 else text


# ── context menus ────────────────────────────────────────────────────────────────────────────
_CURRENT = {}


def popup(menu, x, y):
    """Post a context menu; the previous one is always released first."""
    prev = _CURRENT.get("m")
    if prev is not None:
        try:
            prev.unpost()
            prev.grab_release()
        except tk.TclError:
            pass
        _CURRENT["m"] = None
    _CURRENT["m"] = menu
    menu.bind("<Unmap>", lambda e: _release(menu), add="+")
    menu.tk_popup(x, y)


def _release(menu):
    _CURRENT["m"] = None
    try:
        menu.grab_release()
    except tk.TclError:
        pass


# ── tooltip ──────────────────────────────────────────────────────────────────────────────────
_TIP = {"w": None, "job": None}


def tooltip(text, x, y):
    """A tooltip that cannot survive its owner: overrideredirect windows stay on screen even
    when the app loses focus, so every tooltip also self-destructs after a moment and on any
    click anywhere."""
    dismiss()
    if not text:
        return
    w = tk.Toplevel()
    w.wm_overrideredirect(True)
    w.wm_geometry(f"+{x + 12}+{y + 14}")
    tk.Label(w, text=text, relief="flat", background="#2b2b2b", foreground="#f0f0f0",
             font=("TkDefaultFont", 8), padx=6, pady=3, justify="left").pack()
    w.bind("<Enter>", lambda e: dismiss())
    _TIP["w"] = w
    _TIP["job"] = w.after(3000, dismiss)


def dismiss():
    if _TIP["job"] is not None:
        try:
            _TIP["job"].after_cancel(_TIP["job"])
        except Exception:
            pass
        _TIP["job"] = None
    if _TIP["w"] is not None:
        try:
            _TIP["w"].destroy()
        except tk.TclError:
            pass
        _TIP["w"] = None


# ── atoms ────────────────────────────────────────────────────────────────────────────────────
class Console(ttk.Frame):
    """A text well with tags and auto-scroll. Editable=True turns it into an editor.
    Content is always copyable (Ctrl+C); pages add export via context menu."""

    def __init__(self, app, parent, editable=False, wrap="word", height=0):
        super().__init__(parent)
        self.app = app
        self._editable = editable
        self.text = tk.Text(self, wrap=wrap, padx=12, pady=10, font=("TkFixedFont", 9),
                            state="normal" if editable else "disabled", spacing3=4,
                            **({"height": max(1, height)} if height else {}))
        self.bar = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=self._scroll)
        self.bar.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.autoscroll = True
        self.text.bind("<MouseWheel>", self._wheel, add="+")
        self.text.bind("<Button-4>", self._wheel, add="+")
        self.text.bind("<Button-5>", self._wheel, add="+")
        app.register_paintable(self)

    def _scroll(self, *args):
        self.bar.set(*args)

    def _wheel(self, e):
        self.text.yview_scroll(-1 if getattr(e, "num", 0) == 4 or e.delta > 0 else 1, "units")
        self.autoscroll = self.text.yview()[1] > 0.97

    def put(self, text, tag=None):
        stick = not self.text.winfo_ismapped() or self.text.yview()[1] > 0.97
        self.text.configure(state="normal")
        self.text.insert("end", text, tag or "")
        if stick:
            self.text.see("end")
        if not self._editable:
            self.text.configure(state="disabled")

    line = put

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        if not self._editable:
            self.text.configure(state="disabled")

    def repaint(self, p):
        theme.text_colors(self.text, p)
        self.text.tag_configure("meta", foreground=p["muted"])
        self.text.tag_configure("status", foreground=p["ok"])
        self.text.tag_configure("err", foreground=p["err"])
        self.text.tag_configure("you", foreground=p["accent"], font=("TkFixedFont", 9, "bold"))
        if not self._editable:
            self.text.configure(state="disabled")


class TableList(ttk.Frame):
    """Treeview over dicts: zebra rows, right-aligned numerics, human timestamps, sortable
    headers, upsert for live updates, empty-state caption, tooltips on truncated cells."""

    def __init__(self, app, parent, on_double=None, on_menu=None, status_col=None, height=14,
                 on_select=None):
        super().__init__(parent)
        self.app, self.columns, self.rows = app, [], []
        self.status_col, self.aligns = status_col, {}
        self.tree = ttk.Treeview(self, show="headings", height=height, selectmode="browse")
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.xbar = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.bar.set, xscrollcommand=self.xbar.set)
        self.bar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.xbar.pack(side="bottom", fill="x")
        self.empty = ttk.Label(self.tree, text="", style="Muted.TLabel", anchor="center")
        self.on_double, self.on_menu_cb = on_double, on_menu
        self.tree.bind("<Double-1>", lambda e: self.on_double and self.on_double(self.selected()))
        if on_select:
            self.tree.bind("<<TreeviewSelect>>", lambda e: on_select(self.selected()))
        if on_menu:
            self.tree.bind("<Button-3>", self._menu)
        self.tree.bind("<Motion>", self._hover)
        self.tree.bind("<Leave>", lambda e: dismiss())
        app.register_paintable(self)
        self._tags(app.palette())

    def headings(self, columns):
        self.columns = list(columns)
        self.tree.configure(columns=self.columns)
        for i, c in enumerate(self.columns):
            anchor = "e" if c in RIGHT_COLUMNS else "w"
            self.aligns[c] = anchor
            self.tree.heading(c, text=c.replace("_", " ").capitalize(),
                              command=lambda c=c: self.sort(c), anchor=anchor)
            self.tree.column(c, width=_width(c), minwidth=40, anchor=anchor, stretch=False)

    def fit(self):
        """Column widths from the content, not from the header length: an id column stays narrow,
        the prose column takes the room. Called after every fill."""
        for column in self.columns:
            longest = max([len(str(row.get(column, ""))) for row in self.rows[:60]] + [len(column)])
            self.tree.column(column, width=int(max(46, min(460, 22 + 7.4 * longest))),
                             minwidth=40, stretch=False)

    def fill(self, rows, empty="Nothing to show."):
        self.rows = list(rows)
        if not self.columns and rows:
            self.headings(list(rows[0]))
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.rows):
            self.tree.insert("", "end", iid=str(i), values=self._values(r), tags=self._tags_of(r, i))
        self.fit()
        self._empty_text(empty)

    def _values(self, r):
        out = []
        for c in self.columns:
            v = r.get(c, "")
            if c == "progress" and isinstance(v, (int, float)) and not isinstance(v, bool):
                frac = v if v <= 1 else v / 100
                k = int(round(frac * 8))
                v = f"{'▰' * k}{'▱' * (8 - k)} {int(frac * 100)}%"
            elif c in TIME_COLUMNS:
                v = fmt_time(v)
            elif isinstance(v, float):
                v = f"{v:.4g}"
            out.append("" if v is None else v)
        return out

    def _tags(self, p):
        self.tree.tag_configure("odd", background=p["surface"])
        self.tree.tag_configure("even", background=p["zebra"])
        self.tree.tag_configure("dim", foreground=p["faint"])
        for status, key in (("running", "accent"), ("starting", "accent"), ("done", "ok"),
                            ("failed", "err"), ("cancelled", "muted"), ("queued", "muted"),
                            ("expired", "muted")):
            self.tree.tag_configure(f"st_{status}", foreground=p[key])

    def _tags_of(self, r, i):
        t = ["even" if i % 2 else "odd"]
        if self.status_col and r.get(self.status_col) in ("running", "starting", "done", "failed",
                                                          "cancelled", "queued", "expired"):
            t.append(f"st_{r[self.status_col]}")
        return t

    def _empty_text(self, text):
        if not self.rows and text:
            self.empty.configure(text=text)
            self.empty.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self.empty.place_forget()

    def sort(self, column):
        reverse = getattr(self, "sort_reverse", False) and getattr(self, "sort_key", None) == column
        self.sort_key, self.sort_reverse = column, reverse

        def keyf(r):
            v = r.get(column)
            try:
                return (0, float(v))
            except (TypeError, ValueError):
                return (1, str(v or "").lower())
        self.rows.sort(key=keyf, reverse=reverse)
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.rows):
            self.tree.insert("", "end", iid=str(i), values=self._values(r), tags=self._tags_of(r, i))
        self.fit()

    def upsert(self, key, row):
        for i, r in enumerate(self.rows):
            if r.get(key) == row.get(key):
                self.rows[i] = row
                self.tree.item(str(i), values=self._values(row), tags=self._tags_of(row, i))
                return
        self.rows.append(row)
        self.tree.insert("", "end", iid=str(len(self.rows) - 1), values=self._values(row),
                         tags=self._tags_of(row, len(self.rows) - 1))
        self._empty_text("")

    def selected(self):
        sel = self.tree.selection()
        return self.rows[int(sel[0])] if sel else None

    def _hover(self, e):
        if self.tree.identify_region(e.x, e.y) != "cell":
            return dismiss()
        col, iid = self.tree.identify_column(e.x), self.tree.identify_row(e.y)
        if not iid or not col:
            return
        value = str(self.tree.set(iid, col))
        if len(value) * 7 > self.tree.column(col, "width"):
            tooltip(value, e.x_root, e.y_root)
        else:
            dismiss()

    def _menu(self, e):
        node = self.tree.identify_row(e.y)
        if not node:
            return
        self.tree.selection_set(node)
        m = tk.Menu(self, tearoff=0)
        self.on_menu_cb(m, self.selected())
        if m.index("end") is not None:
            popup(m, e.x_root, e.y_root)

    def repaint(self, p):
        self._tags(p)


class SearchableCombobox(ttk.Combobox):
    """A combobox whose list filters as you type. The dropdown stays open, so a long list of
    databanks is a couple of keystrokes away instead of a scroll."""

    def __init__(self, master, **kw):
        kw.setdefault("state", "normal")
        super().__init__(master, **kw)
        self._all = list(self["values"])
        self.bind("<KeyRelease>", self._filter, add="+")
        self.bind("<<ComboboxSelected>>", self._selected, add="+")

    def _filter(self, event):
        if event.keysym in ("Up", "Down", "Return", "Escape", "Tab", "Shift_L", "Shift_R",
                            "Control_L", "Control_R", "Alt_L", "Alt_R"):
            return
        typed = self.get().strip().lower()
        values = self._all if not typed else [v for v in self._all if typed in str(v).lower()]
        ttk.Combobox.configure(self, values=values)
        if values:
            try:
                self.tk.call("ttk::combobox::Post", self)
            except tk.TclError:
                pass

    def _selected(self, _event):
        ttk.Combobox.configure(self, values=self._all)

    def configure(self, **kw):
        if "values" in kw:
            self._all = list(kw["values"])
        return super().configure(**kw)

    def set_values(self, values):
        self._all = list(values or [])
        ttk.Combobox.configure(self, values=self._all)


class ScrollBox(ttk.Frame):
    """Scrollable holder for a Form: the scrollbar appears only when the content overflows."""

    def __init__(self, app, parent, width=300):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0, width=width)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._region)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win,
                                                                            width=max(e.width, 1)))
        self.canvas.configure(yscrollcommand=self._scroll)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bind("<Enter>", lambda e: self._wheel_on())
        self.bind("<Leave>", lambda e: self._wheel_off())
        app.register_paintable(self)

    def _region(self, _e=None):
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)

    def _scroll(self, first, last):
        self.bar.set(first, last)
        full = float(first) <= 0.0 and float(last) >= 0.999
        if full and self.bar.winfo_ismapped():
            self.bar.pack_forget()
        elif not full and not self.bar.winfo_ismapped():
            self.bar.pack(side="right", fill="y")

    def _wheel(self, e):
        if not self.winfo_ismapped():
            return
        d = -1 if getattr(e, "num", 0) == 4 or e.delta > 0 else 1
        self.canvas.yview_scroll(d, "units")

    def _wheel_on(self):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", self._wheel)
        self.canvas.bind_all("<Button-5>", self._wheel)

    def _wheel_off(self):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def repaint(self, p):
        self.canvas.configure(bg=p["surface"])


class DockPanel(ttk.Frame):
    """A panel with a header strip (title left, actions right)."""

    def __init__(self, app, parent, title):
        super().__init__(parent)
        self.strip = tk.Frame(self, height=22, bd=0, highlightthickness=0)
        self.strip.pack(fill="x")
        self.label = tk.Label(self.strip, text=title, bd=0, font=("TkDefaultFont", 9, "bold"))
        self.label.pack(side="left", padx=6)
        self.tools = ttk.Frame(self.strip)
        self.tools.pack(side="right")
        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True)
        app.register_paintable(self)

    def repaint(self, p):
        self.strip.configure(bg=p["sidebar"])
        self.label.configure(bg=p["sidebar"], fg=p["text"])


class Page(ttk.Frame):
    """Header (title, subtitle, right-aligned actions) + body — the skeleton every screen shares
    so the suite has one face. Subclasses pack content into self.body."""

    def __init__(self, app, parent, title, subtitle=""):
        super().__init__(parent)
        self.app = app
        head = ttk.Frame(self)
        head.pack(fill="x", padx=16, pady=(12, 4))
        box = ttk.Frame(head)
        box.pack(side="left")
        self.title_lbl = ttk.Label(box, text=title, style="Title.TLabel")
        self.title_lbl.pack(anchor="w")
        self.subtitle = ttk.Label(box, text=subtitle, style="Subtitle.TLabel")
        self.subtitle.pack(anchor="w")
        self.server_tag = ttk.Label(box, text="", style="Faint.TLabel")
        self.server_tag.pack(anchor="w")
        self.actions = ttk.Frame(head)
        self.actions.pack(side="right")
        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True, padx=16, pady=(6, 10))

    def set_subtitle(self, text):
        self.subtitle.configure(text=text)

    def set_server(self, name):
        self.server_tag.configure(text=f"server: {name}")


def fit_dialog(window, width=None, height=None, margin=18):
    """Size a Toplevel to its content, never taller or wider than the screen, and never below
    the requested minimum. Call after the last widget is packed, including async content."""
    window.update_idletasks()
    screen_w, screen_h = window.winfo_screenwidth(), window.winfo_screenheight()
    wanted_w = max(width or 0, window.winfo_reqwidth() + margin)
    wanted_h = max(height or 0, window.winfo_reqheight() + margin)
    wanted_w = min(wanted_w, screen_w - 40)
    wanted_h = min(wanted_h, screen_h - 70)
    parent = getattr(window, "master", None)
    try:
        if parent is not None and parent.winfo_exists() and parent.winfo_width() > 1:
            x = parent.winfo_rootx() + max(0, (parent.winfo_width() - wanted_w) // 2)
            y = parent.winfo_rooty() + max(0, (parent.winfo_height() - wanted_h) // 2)
        else:
            x = max(0, (screen_w - wanted_w) // 2)
            y = max(0, (screen_h - wanted_h) // 2)
    except tk.TclError:
        x = y = 20
    window.geometry(f"{wanted_w}x{wanted_h}+{max(0, x)}+{max(0, y)}")
    window.minsize(min(wanted_w, screen_w - 40), min(wanted_h, screen_h - 70))
    return wanted_w, wanted_h


def tip(widget, text):
    """Attach the existing tooltip to a widget on hover."""
    widget.bind("<Enter>", lambda e: tooltip(text, e.x_root, e.y_root), add="+")
    widget.bind("<Leave>", lambda e: dismiss(), add="+")


def placeholder(entry, text):
    """ttk.Entry has no placeholder; emulate it (restored on focus-out when left empty)."""
    entry._ph = text
    entry.insert(0, text)
    entry.configure(foreground="#808080")

    def focus_in(_):
        if entry.get() == entry._ph:
            entry.delete(0, "end")
            entry.configure(foreground="")

    def focus_out(_):
        if not entry.get().strip():
            entry.delete(0, "end")
            entry.insert(0, entry._ph)
            entry.configure(foreground="#808080")
    entry.bind("<FocusIn>", focus_in, add="+")
    entry.bind("<FocusOut>", focus_out, add="+")


def show_payload(app, parent, payload, empty="(empty)"):
    """Render a server answer: Table payload -> TableList, anything else -> Console."""
    for w in parent.winfo_children():
        w.destroy()
    holder = ttk.Frame(parent)
    holder.pack(fill="both", expand=True)
    if isinstance(payload, dict) and payload.get("__table__"):
        t = TableList(app, holder, on_menu=_table_menu)
        t.headings(payload["columns"])
        t.fill(payload["rows"], empty)
        t.pack(fill="both", expand=True)
        return t
    c = Console(app, holder)
    text = payload if isinstance(payload, str) else _dump(payload)
    c.put(text.strip() or empty)
    c.pack(fill="both", expand=True)
    return c


def _table_menu(m, row):
    if not row:
        return
    def copy_all():
        top = m.winfo_toplevel()
        top.clipboard_clear()
        top.clipboard_append(chr(10).join(str(r) for r in row.values()))
    m.add_command(label="Copy row", command=copy_all)


def _dump(obj):
    import json
    return json.dumps(obj, indent=1, default=str) if not isinstance(obj, str) else obj
