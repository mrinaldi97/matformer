# matformer/gui/configpage.py
"""Config editor & wizard, framework-driven. Opening (or creating) a YAML asks the server to parse
it through MatformerConfigs (`config.read`): includes resolved, references linked, every section
paired with its registered dataclass — one tab per section entry ("model", "training",
"data.train", "task.test"), each in a scrolling well, with inline documentation and browse buttons
on path fields. The last tab is the file itself, editable: structured editing is a convenience, not
a cage.

Saving goes back through the framework (override strings, includes preserved); 'Train…' opens the
override dialog first, then queues the job."""
import json
import os
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from matformer.gui.explorer import pick
from matformer.gui.forms import Form
from matformer.gui.icons import icon
from matformer.gui.widgets import Console, Page, ScrollBox, fit_dialog


class ConfigPage(Page):
    """Schema-driven editor + wizard over config.read / config.save / config.new."""

    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Config Editor", "no config open")
        bar = ttk.Frame(self.body)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Button(bar, image=icon("open", 16), text="Open…", compound="left", command=self._pick_open).pack(side="left")
        ttk.Button(bar, text="New…", command=self.new_wizard).pack(side="left", padx=3)
        self.path_cb = ttk.Combobox(bar, values=[], width=34)
        self.path_cb.pack(side="left", fill="x", expand=True, padx=4)
        self.path_cb.bind("<Return>", lambda e: self.open(self.path_cb.get()))
        ttk.Button(bar, image=icon("save", 16), text="Save", compound="left", style="Accent.TButton",
                   command=self.save).pack(side="left")
        ttk.Button(bar, text="Save as…", command=self.save_as).pack(side="left", padx=3)
        structure = ttk.Menubutton(bar, text="Structure", style="Ghost.TButton")
        structure_menu = tk.Menu(structure, tearoff=0)
        structure_menu.add_command(label="Add entry…", command=self.add_entry)
        structure_menu.add_command(label="Remove section", command=self.remove_section)
        structure["menu"] = structure_menu
        structure.pack(side="left", padx=3)
        self.notebook = ttk.Notebook(self.body)
        self.notebook.pack(fill="both", expand=True)
        self.blank = ttk.Frame(self.notebook)
        ttk.Label(self.blank, justify="center", style="Muted.TLabel",
                  text="no config open").pack(expand=True)
        self.notebook.add(self.blank, text="welcome")
        self.path, self.sections, self.forms = None, [], {}
        self.raw = None
        ttk.Button(self.actions, text="Train…", style="Accent.TButton",
                   command=self.train).pack(side="left")

    # ── open / render ──────────────────────────────────────────────────────────
    def _pick_open(self):
        p = pick(self.app, self, "", (".yaml", ".yml"), None, "Open a config")
        if p:
            self.open(p)

    @property
    def server(self):
        return self.app.server

    def new_wizard(self):
        NewConfigDialog(self)

    def open(self, path, server_name=None):
        if not path:
            return
        self.path = path
        self.set_subtitle(f"reading {path}…")
        self.server.call("config.read", {"path": path}, self._render)

    def activated(self):
        pass

    def _render(self, result):
        if not result or isinstance(result, tuple):
            self.set_subtitle("config.read failed")
            return
        for tab in list(self.notebook.tabs()):
            self.notebook.forget(tab)
        self.forms, self.sections = {}, result["sections"]
        for sec in self.sections:
            label = _tab_label(sec)
            tab = ttk.Frame(self.notebook)
            scroll = ScrollBox(self.app, tab, width=10)
            scroll.pack(fill="both", expand=True)
            form = Form(self.app, scroll.inner, sec["fields"])
            form.load({f["name"]: f["default"] for f in sec["fields"] if f["default"] is not None})
            form.pack(fill="x", anchor="n", padx=8, pady=8)
            self.notebook.add(tab, text=label)
            self.forms[label] = (form, sec)
        self._add_raw_tab()
        self.set_subtitle(f"{self.path}   ·   {len(self.sections)} sections"
                          + (f"   ·   model: {self._variant_of('model')}"
                             if self._variant_of("model") else ""))
        values = list(self.path_cb.cget("values"))
        if self.path not in values:
            self.path_cb.configure(values=[self.path] + values[:12])
        self.path_cb.set(self.path)

    def _add_raw_tab(self):
        tab = ttk.Frame(self.notebook)
        self.raw = Console(self.app, tab, editable=True)
        self.raw.pack(fill="both", expand=True)
        self.notebook.add(tab, text="file")
        self.server.call("fs.read", {"path": self.path},
                         lambda r: r and not isinstance(r, tuple)
                         and (self.raw.clear(), self.raw.put(r["text"])))

    def _variant_of(self, section):
        return next((s.get("variant") for s in self.sections if s["section"] == section), None)

    # ── overrides / save ───────────────────────────────────────────────────────
    def overrides(self):
        """Diff current form values against the loaded ones: framework override strings."""
        out = []
        for label, (form, sec) in self.forms.items():
            prefix = f"{sec['section']}.{sec['name']}" if sec.get("name") else sec["section"]
            for f in sec["fields"]:
                try:
                    current = form.entries[f["name"]][0]()
                except (KeyError, ValueError, tk.TclError):
                    continue
                if _same(current, f["default"]):
                    continue
                out.append(f"{prefix}.{f['name']}={json.dumps(current)}")
        return out

    def save(self):
        if self.notebook.tab(self.notebook.select(), "text") == "file":
            return self.save_raw()
        changes = self.overrides()
        if not changes:
            return self.app.toast("nothing changed to save")
        self.server.call("config.save", {"source": self.path, "target": self.path,
                                         "overrides": changes},
                         lambda r: (self.app.toast(str(r.get("msg", r))), self.open(self.path)))

    def save_raw(self):
        text = self.raw.text.get("1.0", "end-1c")
        self.server.call("config.write", {"path": self.path, "text": text},
                         lambda r: (self.app.toast(str(r.get("msg", r))), self.open(self.path)))

    def save_as(self):
        target = simpledialog.askstring("Save config as", "New path:",
                                        initialvalue=self.path.replace(".yaml", "-edit.yaml"),
                                        parent=self)
        if not target:
            return
        inherit = messagebox.askyesno("Save as", "Keep the current file as an inherited base?\n\n"
                                      "yes → {includes: [current]} + only your changes\n"
                                      "no  → fully flattened copy")
        cmd = "config.derive" if inherit else "config.save"
        self.server.call(cmd, {"source": self.path, "target": target, "overrides": self.overrides()},
                         lambda r: (self.app.toast(str(r.get("msg", r))), self.open(target)))

    def train(self):
        self.app.start_training(self.path, overrides=self.overrides())

    def add_entry(self):
        AddEntryDialog(self)

    def remove_section(self):
        tab = self.notebook.tab(self.notebook.select(), "text")
        if tab in ("welcome", "file") or not self.path:
            return
        section, _, name = tab.partition(".")
        self.app.server.call("config.delete_section",
                             {"path": self.path, "section": section, "name": name or None},
                             lambda r: (self.app.toast(str(r.get("msg", r))), self.open(self.path)))


def _tab_label(section):
    """A tab name a user recognises: the section, and its entry name when there is more than one.
    Framework-internal suffixes (an unnamed entry the resolver called default_None) stay out."""
    name = (section.get("name") or "").replace("_None", "")
    if not name or name == "default":
        return section["section"]
    return f"{section['section']}.{name}"


class NewConfigDialog(tk.Toplevel):
    """New config: a name in the configs folder, the sections to declare, and optionally a base to
    inherit from — the child keeps {includes: [base]} so only what changes is ever written."""

    def __init__(self, page):
        super().__init__(page)
        self.page, self.app = page, page.app
        self.title("New config")
        self.geometry("600x420")
        self.transient(page)
        self.home = ""
        form = ttk.Frame(self)
        form.pack(fill="both", expand=True, padx=14, pady=12)
        form.columnconfigure(2, weight=1)
        ttk.Label(form, text="Folder:").grid(row=0, column=0, sticky="w", pady=4)
        self.folder = ttk.Entry(form)
        self.folder.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(form, text="…", width=3, command=self._pick_dir).grid(row=0, column=2, padx=4)
        ttk.Label(form, text="Name:").grid(row=1, column=0, sticky="w", pady=4)
        self.name = ttk.Entry(form)
        self.name.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Label(form, text="Inherit from:").grid(row=2, column=0, sticky="w", pady=4)
        self.base = ttk.Entry(form)
        self.base.grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(form, text="…", width=3, command=self._pick_base).grid(row=2, column=2, padx=4)
        box = ttk.LabelFrame(form, text=" Sections ")
        box.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=8)
        self.checks = {}
        self.sections = []
        self.grid_in = box
        ttk.Label(form, style="Muted.TLabel",
                  text="Checked sections are written with registry defaults; nothing checked means an empty file."
                  ).grid(row=4, column=1, columnspan=2, sticky="w")
        row = ttk.Frame(self)
        row.pack(fill="x", padx=14, pady=(0, 12))
        ttk.Button(row, text="Create", style="Accent.TButton", command=self._create).pack(side="right")
        ttk.Button(row, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        self.bind("<Escape>", lambda e: self.destroy())
        self.app.server.call("config.home", None, self._home)
        fit_dialog(self, 600, 0)

    def _home(self, result):
        if result and not isinstance(result, tuple):
            self.home = result["path"]
            self.folder.insert(0, self.home)
            self._load_sections()

    def _load_sections(self):
        def got(categories):
            if not categories or isinstance(categories, tuple):
                return
            known = [c["category"] for c in categories if c["category"] != "plugin"]
            core = [s for s in ("model", "tokenizer", "data", "training", "task") if s in known]
            useful = {"validation", "masking", "checkpoint", "saver", "weights"}
            rest = [c for c in known if c not in core and c in useful]
            for section in core + rest:
                var = tk.BooleanVar(value=section in core)
                ttk.Checkbutton(self.grid_in, text=section, variable=var).pack(anchor="w", padx=6)
                self.checks[section] = var
            self.sections = core + rest
            fit_dialog(self, 600, 0)
        self.app.server.call("config.sections", None, got)

    def _pick_dir(self):
        p = pick(self.app, self, self.home or "", (), None, "Folder for the new config")
        if p:
            self.folder.delete(0, "end")
            self.folder.insert(0, p if not p.endswith(".yaml") else os.path.dirname(p))

    def _pick_base(self):
        p = pick(self.app, self, self.home or "", (".yaml", ".yml"), None, "Config to inherit")
        if p:
            self.base.delete(0, "end")
            self.base.insert(0, p)

    def _create(self):
        name = self.name.get().strip()
        if not name:
            return self.app.toast("a name is needed", error=True)
        if not name.endswith((".yaml", ".yml")):
            name += ".yaml"
        folder = self.folder.get().strip() or self.home
        sections = [s for s, var in self.checks.items() if var.get()]
        def created(result):
            if isinstance(result, tuple):
                return self.app.toast(result[1], error=True)
            if result and result.get("path"):
                self.destroy()
                self.page.open(result["path"])
            else:
                self.app.toast("config.new returned no path", error=True)
        self.app.server.call("config.new", {"path": os.path.join(folder, name),
                                            "sections": sections,
                                            "base": self.base.get().strip() or None},
                             created)


class AddEntryDialog(tk.Toplevel):
    """Declare another entry of a listed section (a second task, a data split): the categories and
    their variants come from the live registry."""

    def __init__(self, page):
        super().__init__(page)
        self.page, self.app = page, page.app
        self.title("Add entry")
        self.geometry("460x300")
        self.transient(page)
        form = ttk.Frame(self)
        form.pack(fill="both", expand=True, padx=14, pady=14)
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Section:").grid(row=0, column=0, sticky="w", pady=4)
        self.section = ttk.Combobox(form, values=[], state="readonly")
        self.section.grid(row=0, column=1, sticky="ew", pady=4)
        self.section.bind("<<ComboboxSelected>>", lambda e: self._variants())
        ttk.Label(form, text="Variant:").grid(row=1, column=0, sticky="w", pady=4)
        self.variant = ttk.Combobox(form, values=[], state="readonly")
        self.variant.grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(form, text="Name:").grid(row=2, column=0, sticky="w", pady=4)
        self.name = ttk.Entry(form)
        self.name.grid(row=2, column=1, sticky="ew", pady=4)
        row = ttk.Frame(self)
        row.pack(fill="x", padx=14, pady=(0, 12))
        ttk.Button(row, text="Add", style="Accent.TButton", command=self._add).pack(side="right")
        ttk.Button(row, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        self.categories = []
        self.app.server.call("config.sections", None, self._loaded)
        fit_dialog(self, 460, 0)

    def _loaded(self, categories):
        if categories and not isinstance(categories, tuple):
            self.categories = categories
            listed = [c["category"] for c in categories
                      if c["category"] in ("data", "task")]
            self.section.configure(values=listed)
            if listed:
                self.section.set(listed[0])
                self._variants()

    def _variants(self):
        cat = next((c for c in self.categories if c["category"] == self.section.get()), None)
        if cat:
            self.variant.configure(values=cat["variants"])
            self.variant.set(cat["default"] if cat["default"] in cat["variants"]
                             else (cat["variants"][0] if cat["variants"] else ""))

    def _add(self):
        if not self.name.get().strip() or not self.page.path:
            return self.app.toast("a name is needed", error=True)
        self.app.server.call("config.add_entry", {"path": self.page.path,
                                                  "section": self.section.get(),
                                                  "name": self.name.get().strip(),
                                                  "variant": self.variant.get() or None},
                             lambda r: (self.destroy(), self.app.toast(str(r.get("msg", r))),
                                        self.page.open(self.page.path)))


def _same(current, default):
    if isinstance(current, str) and current.strip() == "":
        return True
    if isinstance(default, float) and isinstance(current, (int, float)):
        return abs(current - default) < 1e-12
    if str(current) == str(default):
        return True
    try:
        return json.loads(current) == default
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
