# matformer/gui/theme.py
"""The design system. Everything visual flows from here: token palettes in app.toml, three button
tiers (Accent / Default / Ghost / Danger), flat surfaces raised by tone not bevel (lightcolor ==
darkcolor == background), hairlines instead of grooves, focus rings on inputs, slim arrow-less
scrollbars, custom rounded check/radio indicators, and a central THEMED registry so a theme switch
re-paints literally everything (tags, dots, menus, popdowns included — dark mode is fully applied
or it is a lie).

Keys: pages use the semantic tokens (bg sidebar surface hover zebra border border_strong text
muted faint accent accent_soft accent_fg ok warn err err_soft). app.toml [themes.*] may define any
subset; aliases (panel->sidebar, field->surface, fg->text, sel->accent, on_accent->accent_fg)
keep older widgets working."""
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

FAMILIES = ("Inter", "Noto Sans", "Segoe UI", "DejaVu Sans", "TkDefaultFont")
MONOS = ("DejaVu Sans Mono", "Liberation Mono", "Consolas", "TkFixedFont")
ALIASES = {"panel": "sidebar", "field": "surface", "fg": "text", "sel": "accent",
           "on_accent": "accent_fg", "grid": "zebra"}

THEMED = []           # (widget, fn(widget, palette)) — re-applied on every theme switch


def themed(widget, fn):
    THEMED.append((widget, fn))
    return widget


def palette_of(cfg, name):
    """A theme table from app.toml, alias-expanded with sane fallbacks."""
    raw = dict(cfg["themes"].get(name) or {})
    p = {"name": name}
    p.update({k: raw.get(k) for k in
              ("bg sidebar surface sunken border border_strong hover zebra text muted faint"
               " accent accent_hover accent_soft accent_fg ok warn err err_soft").split()})
    for k, v in raw.items():
        p[ALIASES.get(k, k)] = v
    p = {k: v for k, v in p.items() if v is not None}          # setdefault would not replace None
    p.setdefault("surface", "#ffffff"); p.setdefault("bg", "#f4f4f2"); p.setdefault("sidebar", "#ecebe8")
    p.setdefault("text", "#1d1c1a"); p.setdefault("accent", "#2f5fd0"); p.setdefault("accent_fg", "#ffffff")
    p.setdefault("accent_soft", "#e4ebfb"); p.setdefault("border", "#d9d7d2")
    p.setdefault("border_strong", "#bdbab3"); p.setdefault("hover", "#e9e8e4"); p.setdefault("zebra", "#fafaf9")
    p.setdefault("muted", "#6a6760"); p.setdefault("faint", "#9b988f"); p.setdefault("ok", "#1f8a4c")
    p.setdefault("warn", "#b7791f"); p.setdefault("err", "#c0392b"); p.setdefault("err_soft", "#fbeceb")
    p.setdefault("sunken", p["surface"]); p.setdefault("accent_hover", p["accent"])
    for old, new in ALIASES.items():                       # keep old key names alive for widgets
        if old not in p or p.get(old) is None:
            p[old] = p.get(new)
    return p


def pick_family(root, override=None, mono=False):
    available = set(tkfont.families(root))
    if override and override in available:
        return override
    return next((f for f in (MONOS if mono else FAMILIES) if f in available),
                "TkFixedFont" if mono else "TkDefaultFont")


# ── drawn indicators (14px rounded boxes/circles, check/dot inside) ───────────────────────────
def _fill(im, painter, d, blank="#ffffff"):
    """Repaint an existing PhotoImage in place. Theme switches mutate the images ttk already
    references, so `blank` (the surrounding widget background) is always the current palette's."""
    im.put([[painter(x, y) or blank for x in range(d)] for y in range(d)])


def _inside_round(x, y, d, r=3):
    """True if (x, y) is inside the rounded-rect border (the corner is cut)."""
    if x < 0 or y < 0 or x >= d or y >= d:
        return False
    near = (x < r or y < r or x >= d - r or y >= d - r)
    if not near:
        return True
    cx, cy = min(x, d - 1 - x) - r, min(y, d - 1 - y) - r
    return cx * cx + cy * cy <= r * r


_ELEMENTS_MADE = set()


def build_images(p):
    """Check/radio indicators + status dots for one palette, stored on theme.IND. Images are
    created once per process and repainted on every theme switch."""
    d = 14
    specs = {
        "check": lambda x, y: p["border_strong"] if _inside_round(x, y, d) else p["surface"],
        "check_on": lambda x, y: _checked_paint(x, y, p),
        "radio": lambda x, y: _radio_paint(x, y, p, on=False),
        "radio_on": lambda x, y: _radio_paint(x, y, p, on=True),
        "dot_none": lambda x, y: None,
    }
    for name, color in (("ok", p["ok"]), ("err", p["err"]), ("warn", p["warn"]),
                        ("muted", p["muted"]), ("accent", p["accent"])):
        specs["dot_" + name] = (lambda c: (lambda x, y:
                                           c if (x - 5 + .5) ** 2 + (y - 5 + .5) ** 2 <= 16
                                           else None))(color)
    for key, painter in specs.items():
        if key not in IND:
            size = d if key.startswith(("check", "radio")) else 10
            IND[key] = tk.PhotoImage(name=f"mtind_{key}", width=size, height=size)
        blank = p["bg"] if key.startswith(("check", "radio")) else p["surface"]
        _fill(IND[key], painter, IND[key].width(), blank)
    return IND


def _checked_paint(x, y, p):
    if _inside_round(x, y, 14):
        return p["accent"]
    # checkmark strokes
    if y - x >= 3 and y - x <= 5 and x >= 6:
        return p["accent_fg"]
    if x + y >= 11 and x + y <= 13 and x <= 6 and y >= 4:
        return p["accent_fg"]
    return p["accent"]


def _radio_paint(x, y, p, on):
    dx, dy = x - 6.5, y - 6.5
    r2 = dx * dx + dy * dy
    if r2 > 49:
        return None
    if r2 > 36:
        return p["border_strong"]
    if on:
        return p["accent"] if r2 <= 16 else p["surface"]
    return p["surface"]


IND = {}


# ── the style pass ─────────────────────────────────────────────────────────────────────────────
def apply(root, name, p, fonts=None):
    fonts = fonts or {}
    fam, mfam = pick_family(root, fonts.get("family") or None), pick_family(root, None, mono=True)
    ui, mono = int(fonts.get("ui", 9)), int(fonts.get("mono", 9))
    for named in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkIconFont", "TkCaptionFont",
                  "TkSmallCaptionFont", "TkTooltipFont"):
        try:
            tkfont.nametofont(named).configure(family=fam, size=ui)
        except tk.TclError:
            pass
    tkfont.nametofont("TkHeadingFont").configure(family=fam, size=ui, weight="normal")
    tkfont.nametofont("TkFixedFont").configure(family=mfam, size=mono)
    root.configure(bg=p["bg"])
    build_images(p)

    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=p["bg"], foreground=p["text"], fieldbackground=p["surface"],
                bordercolor=p["border"], lightcolor=p["bg"], darkcolor=p["bg"],
                troughcolor=p["bg"], focuscolor=p["accent"], font=(fam, ui), relief="flat")

    for w in ("TFrame", "TLabel", "TPanedwindow", "TNotebook"):
        s.configure(w, background=p["bg"], relief="flat")
    s.configure("Surface.TFrame", background=p["surface"])

    # buttons: three tiers, all flat (elevation by tone)
    s.configure("TButton", background=p["surface"], foreground=p["text"], bordercolor=p["border"],
                lightcolor=p["surface"], darkcolor=p["surface"], borderwidth=1, relief="flat",
                padding=(12, 5), focusthickness=0, focuscolor=p["accent"])
    s.map("TButton", background=[("pressed", p["border"]), ("active", p["hover"]), ("disabled", p["bg"])],
          foreground=[("disabled", p["faint"])], bordercolor=[("focus", p["accent"]), ("active", p["border_strong"])])
    s.configure("Accent.TButton", background=p["accent"], foreground=p["accent_fg"],
                bordercolor=p["accent"], lightcolor=p["accent"], darkcolor=p["accent"])
    s.map("Accent.TButton", background=[("active", p["accent_hover"]), ("pressed", p["accent_hover"]),
                                        ("disabled", p["border"])], foreground=[("disabled", p["bg"])])
    s.configure("Ghost.TButton", background=p["bg"], bordercolor=p["bg"], lightcolor=p["bg"],
                darkcolor=p["bg"], padding=(8, 4))
    s.map("Ghost.TButton", background=[("active", p["hover"])])
    s.configure("Danger.TButton", background=p["surface"], foreground=p["err"],
                bordercolor=p["border"])
    s.map("Danger.TButton", background=[("active", p["err_soft"])])

    # inputs with focus ring
    for w in ("TEntry", "TSpinbox", "TCombobox"):
        s.configure(w, fieldbackground=p["surface"], foreground=p["text"], insertcolor=p["text"],
                    arrowcolor=p["muted"], background=p["surface"], bordercolor=p["border"],
                    lightcolor=p["surface"], darkcolor=p["surface"], borderwidth=1, relief="flat",
                    padding=(6, 4), arrowsize=14)
        s.map(w, bordercolor=[("focus", p["accent"]), ("disabled", p["border"])],
              lightcolor=[("focus", p["accent"])], darkcolor=[("focus", p["accent"])],
              fieldbackground=[("readonly", p["surface"]), ("disabled", p["bg"])],
              foreground=[("disabled", p["faint"])])
    root.option_add("*TCombobox*Listbox.background", p["surface"])
    root.option_add("*TCombobox*Listbox.foreground", p["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", p["accent_soft"])
    root.option_add("*TCombobox*Listbox.selectForeground", p["text"])
    root.option_add("*TCombobox*Listbox.border*", p["border"])

    # indicators: custom rounded images
    if "mt.check" not in _ELEMENTS_MADE:
        s.element_create("mt.check", "image", IND["check"], ("selected", IND["check_on"]),
                         border=2, sticky="w")
        s.element_create("mt.radio", "image", IND["radio"], ("selected", IND["radio_on"]),
                         border=2, sticky="w")
        _ELEMENTS_MADE.update({"mt.check", "mt.radio"})
    s.layout("TCheckbutton", [("mt.check", {"side": "left", "sticky": ""}),
                              ("Checkbutton.label", {"sticky": "nswe"})])
    s.layout("TRadiobutton", [("mt.radio", {"side": "left", "sticky": ""}),
                              ("Radiobutton.label", {"sticky": "nswe"})])
    s.configure("TCheckbutton", background=p["bg"], foreground=p["text"], padding=(0, 3))
    s.configure("TRadiobutton", background=p["bg"], foreground=p["text"], padding=(0, 3))

    # tabs: flat, underline-ish selection
    s.configure("TNotebook", tabmargins=(6, 3, 6, 0), borderwidth=0)
    s.configure("TNotebook.Tab", background=p["bg"], foreground=p["muted"], padding=(12, 6),
                borderwidth=0)
    s.map("TNotebook.Tab", background=[("selected", p["surface"])],
          foreground=[("selected", p["text"]), ("active", p["text"])])

    # tables: borderless, 26px rows, muted flat headers
    s.configure("Treeview", background=p["surface"], fieldbackground=p["surface"], foreground=p["text"],
                rowheight=26, borderwidth=0, relief="flat")
    s.configure("Treeview.Heading", background=p["bg"], foreground=p["muted"], relief="flat",
                font=(fam, ui), padding=(8, 5))
    s.map("Treeview.Heading", background=[("active", p["hover"])], foreground=[("active", p["text"])])
    s.map("Treeview", background=[("selected", p["accent_soft"])], foreground=[("selected", p["text"])])
    try:
        s.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
    except Exception:
        pass

    # slim scrollbars, no arrows
    for orient, name in (("vertical", "Vertical"), ("horizontal", "Horizontal")):
        s.layout(f"{name}.TScrollbar",
                 [(f"{name}.Scrollbar.trough", {"sticky": "ns" if orient == "vertical" else "ew",
                   "children": [(f"{name}.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})])
        s.configure(f"{name}.TScrollbar", width=10, arrowsize=0, borderwidth=0, relief="flat",
                    troughcolor=p["surface"], background=p["border_strong"],
                    lightcolor=p["border_strong"], darkcolor=p["border_strong"])
        s.map(f"{name}.TScrollbar", background=[("active", p["muted"])])

    # progress + scale
    s.configure("TProgressbar", background=p["accent"], troughcolor=p["hover"], borderwidth=0,
                lightcolor=p["accent"], darkcolor=p["accent"], thickness=6)
    s.configure("TScale", background=p["bg"], troughcolor=p["hover"], sliderrelief="flat",
                sliderlength=14, borderwidth=0)
    s.configure("Horizontal.TScale", background=p["bg"], gripcount=0)

    # labels
    s.configure("Head.TLabel", background=p["bg"], foreground=p["text"], font=(fam, 10, "bold"),
                padding=(0, 2))
    s.configure("Muted.TLabel", foreground=p["muted"], font=(fam, max(7, ui - 1)))
    s.configure("Faint.TLabel", foreground=p["faint"], font=(fam, max(7, ui - 1)))
    s.configure("Title.TLabel", font=(fam, 14, "bold"), background=p["bg"], foreground=p["text"])
    s.configure("Subtitle.TLabel", font=(fam, ui), background=p["bg"], foreground=p["muted"])

    # sashes: flat hairlines
    for sp in ("Horizontal.TSash", "Vertical.TSash"):
        s.configure(sp, background=p["bg"], sashwidth=5, sashpad=0, relief="flat", gripcount=0,
                    lightcolor=p["bg"], darkcolor=p["bg"])

    THEMED[:] = [(w, fn) for (w, fn) in THEMED if w.winfo_exists()]


def menu_colors(p):
    return dict(bg=p["surface"], fg=p["text"], activebackground=p["accent"],
                activeforeground=p["accent_fg"], borderwidth=1, relief="flat",
                selectcolor=p["surface"])


def text_colors(widget, p, mono=True):
    """Repaint a plain tk.Text / tk.Canvas / tk.Listbox content well."""
    widget.configure(bg=p["surface"], fg=p["text"], insertbackground=p["text"], relief="flat",
                     borderwidth=0, highlightthickness=0, selectbackground=p["accent_soft"],
                     selectforeground=p["text"],
                     font=(pick_family(widget, None, True), 9) if mono else widget.cget("font"))
