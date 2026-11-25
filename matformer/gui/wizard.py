# matformer/gui/wizard.py
"""A renderer for the TOML wizard definitions.

The dialog knows nothing about training, MDAT or annotation: it draws the steps the server returns,
collects answers, and submits them. Every choice, path and list control is generic. Fields marked
advanced sit behind one expander; conditional fields follow the same `when` language the server
uses. The summary shows the generated config or command list before anything is queued.
"""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk

from matformer.gui.explorer import pick
from matformer.gui.widgets import Console, ScrollBox, fit_dialog


def _options(field):
    out = []
    for option in field.get("options") or []:
        if isinstance(option, dict):
            out.append((option.get("value"), option.get("label", str(option.get("value"))),
                        option.get("doc", "")))
        else:
            out.append((option, str(option), ""))
    return out


def _test(condition, values):
    if not condition:
        return True
    if isinstance(condition, list):
        return all(_test(one, values) for one in condition)
    if "any" in condition:
        return any(_test(one, values) for one in condition["any"])
    if "all" in condition:
        return all(_test(one, values) for one in condition["all"])
    if "not" in condition:
        return not _test(condition["not"], values)
    value = values.get(condition.get("field"))
    if "is" in condition:
        return value == condition["is"]
    if "equals" in condition:
        return value == condition["equals"]
    if "in" in condition:
        return value in condition["in"]
    if "not_in" in condition:
        return value not in condition["not_in"]
    if condition.get("truthy"):
        return bool(value)
    if "exists" in condition:
        return (value is not None) == bool(condition["exists"])
    return True


class WizardDialog(tk.Toplevel):
    def __init__(self, app, wizard_id, on_done=None, initial=None):
        super().__init__(app)
        self.app, self.wizard_id, self.on_done = app, wizard_id, on_done
        self.answers = dict(initial or {})
        self.schema, self.steps = {}, []
        self.context = {}
        self.index, self.summary, self._busy = 0, False, False
        self._generation, self._advanced = 0, False
        self._widgets = []
        self.title("Wizard")
        self.geometry("920x680")
        self.minsize(760, 520)
        self.transient(app)
        self._build()
        self._load()
        fit_dialog(self, 920, 0)

    @property
    def server(self):
        return self.app.server

    def _build(self):
        self.head = ttk.Frame(self)
        self.head.pack(fill="x", padx=14, pady=(12, 4))
        self.title_lbl = ttk.Label(self.head, text="reading the wizard…", style="Title.TLabel")
        self.title_lbl.pack(anchor="w")
        self.description = ttk.Label(self.head, style="Subtitle.TLabel", wraplength=840,
                                     justify="left")
        self.description.pack(anchor="w")
        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=6)
        left = ttk.Frame(body)
        body.add(left, weight=0)
        palette = self.app.palette()
        self.step_list = tk.Listbox(left, width=40, font=("TkDefaultFont", 9),
                                    activestyle="none", exportselection=False,
                                    bg=palette["surface"], fg=palette["text"],
                                    selectbackground=palette["accent_soft"],
                                    selectforeground=palette["text"],
                                    highlightthickness=0, bd=0)
        self.step_list.pack(fill="both", expand=True)
        self.step_list.bind("<<ListboxSelect>>", self._jump)
        right = ttk.Frame(body)
        body.add(right, weight=1)
        self.step_title = ttk.Label(right, text="", style="Head.TLabel")
        self.step_title.pack(anchor="w", padx=4)
        self.step_doc = ttk.Label(right, text="", style="Muted.TLabel", wraplength=610,
                                  justify="left")
        self.step_doc.pack(anchor="w", padx=4, pady=(2, 6))
        self.scroll = ScrollBox(self.app, right, width=10)
        self.scroll.pack(fill="both", expand=True)
        self.form = ttk.Frame(self.scroll.inner)
        self.form.pack(fill="x", anchor="n", padx=8, pady=4)
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=14, pady=10)
        self.back_btn = ttk.Button(bottom, text="< Back", command=self.back)
        self.back_btn.pack(side="left")
        self.next_btn = ttk.Button(bottom, text="Next >", style="Accent.TButton",
                                   command=self.next)
        self.next_btn.pack(side="right", padx=4)
        ttk.Button(bottom, text="Cancel", style="Ghost.TButton",
                   command=self.destroy).pack(side="right")
        self.bind("<Escape>", lambda e: self.destroy())

    # ── server round-trips ────────────────────────────────────────────────────
    def _load(self):
        def got(context):
            if context and not isinstance(context, tuple):
                self.context = context
            self._fetch()
        self.server.call("wizard.context", None, got)

    def _fetch(self, keep=None):
        self._generation += 1
        generation = self._generation
        self._busy = True
        try:
            self.next_btn.configure(state="disabled")
            self.back_btn.configure(state="disabled")
        except tk.TclError:
            pass
        self.server.call("wizard.schema", {"id": self.wizard_id, "answers": self.answers},
                         lambda payload: self._schema(payload, generation, keep))

    def _schema(self, payload, generation, keep):
        if generation != self._generation or not payload or isinstance(payload, tuple):
            return
        self._busy = False
        self.schema = payload
        self.steps = payload.get("steps") or []
        for step in self.steps:
            for field in step.get("fields", []):
                if field.get("prefill") and self.answers.get(field["name"]) in (None, ""):
                    value = self.context.get(field["prefill"])
                    if value not in (None, ""):
                        self.answers[field["name"]] = value
        self.title_lbl.configure(text=payload.get("title", self.wizard_id))
        self.description.configure(text=payload.get("description", ""))
        self.title(f"{payload.get('title', self.wizard_id)} - wizard")
        for name, value in (payload.get("values") or {}).items():
            if name not in self.answers and value is not None:
                self.answers[name] = value
        if keep:
            self.index = next((i for i, step in enumerate(self.steps) if step["id"] == keep), 0)
        self.index = max(0, min(self.index, max(0, len(self.steps) - 1)))
        self._fill_steps()
        self._render()
        try:
            self.next_btn.configure(state="normal")
        except tk.TclError:
            pass

    def _fill_steps(self):
        self.step_list.delete(0, "end")
        for i, step in enumerate(self.steps, start=1):
            self.step_list.insert("end", f"{i}.  {step['title']}")
        if self.steps:
            self.step_list.selection_clear(0, "end")
            self.step_list.selection_set(self.index)
            self.step_list.see(self.index)

    def _jump(self, _event=None):
        selection = self.step_list.curselection()
        if not selection or self.summary:
            return
        self._collect()
        self.index = selection[0]
        self._fetch(keep=self.steps[self.index]["id"])

    # ── rendering ─────────────────────────────────────────────────────────────
    def _render(self):
        self.summary = False
        for widget in self.form.winfo_children():
            widget.destroy()
        self._widgets = []
        if not self.steps:
            ttk.Label(self.form, text="this wizard has no step for the current answers",
                      style="Muted.TLabel").pack(anchor="w")
            self.next_btn.configure(text="Next >", state="disabled")
            self.back_btn.configure(state="disabled")
            return
        step = self.steps[self.index]
        self.step_title.configure(text=step["title"])
        self.step_doc.configure(text=step.get("description", ""))
        self.back_btn.configure(state="normal" if self.index else "disabled")
        self.next_btn.configure(text="Next >" if self.index < len(self.steps) - 1 else "Summary >",
                                state="normal")
        self.step_list.selection_clear(0, "end")
        self.step_list.selection_set(self.index)
        visible = [field for field in step.get("fields", [])
                   if _test(field.get("when"), self.answers)]
        focused = [field for field in visible if not field.get("advanced")]
        advanced = [field for field in visible if field.get("advanced")]
        button = None
        if advanced:
            frame = ttk.Frame(self.form)
            frame.pack(fill="x", pady=(0, 4))
            count = len(advanced)
            button = ttk.Button(frame, text=f"Advanced options ({count})",
                                style="Ghost.TButton", command=self._toggle_advanced)
            button.pack(anchor="w")
        section = None
        for field in focused:
            section = self._field(field, section)
        if advanced:
            self.advanced_holder = ttk.Frame(self.form)
            self._advanced_button = button
            self._advanced_count = len(advanced)
            if self._advanced:
                self.advanced_holder.pack(fill="x", anchor="n")
            self._advanced_frame = self.advanced_holder
            section = None
            for field in advanced:
                section = self._field(field, section, parent=self.advanced_holder)
        if not visible:
            ttk.Label(self.form, text="no option in this step for the current answers",
                      style="Muted.TLabel").pack(anchor="w")

    def _toggle_advanced(self):
        self._advanced = not self._advanced
        if self._advanced:
            self._advanced_frame.pack(fill="x", anchor="n")
            self._advanced_button.configure(text=f"Hide advanced options ({self._advanced_count})")
        else:
            self._advanced_frame.pack_forget()
            self._advanced_button.configure(
                text=f"Advanced options ({self._advanced_count})")

    def _field(self, field, section, parent=None):
        parent = parent or self.form
        if field.get("section") and field["section"] != section:
            ttk.Label(parent, text=field["section"].replace(".", " · ").upper(),
                      style="Head.TLabel").pack(anchor="w", pady=(10, 2))
            section = field["section"]
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        row.columnconfigure(1, weight=1)
        label = field.get("label", field["name"])
        ttk.Label(row, text=label + ":", width=30).grid(row=0, column=0, sticky="nw")
        value = self.answers.get(field["name"], field.get("default"))
        getter = self._widget(row, field, value)
        self._widgets.append((field, getter))
        if field.get("doc"):
            ttk.Label(row, text=field["doc"], style="Faint.TLabel", wraplength=430,
                      justify="left").grid(row=1, column=1, sticky="w", pady=(0, 2))
        return section

    def _widget(self, row, field, value):
        kind = field.get("kind", "text")
        holder = ttk.Frame(row)
        holder.grid(row=0, column=1, sticky="ew")
        holder.columnconfigure(0, weight=1)
        if kind == "bool":
            if value is None and field.get("default") is None:
                var = tk.StringVar(value="")
                box = ttk.Combobox(holder, values=["", "true", "false"], state="readonly")
                box.grid(row=0, column=0, sticky="ew")
                return lambda: None if not var.get() else var.get() == "true"
            var = tk.BooleanVar(value=bool(value))
            ttk.Checkbutton(holder, variable=var).grid(row=0, column=0, sticky="w")
            return var.get
        if kind in ("choice", "radio"):
            options = _options(field)
            if not options:
                options = [(value, str(value), "")] if value is not None else []
            labels = [label for _, label, _ in options]
            values = {label: option for option, label, _ in options}
            current = next((label for option, label, _ in options if option == value),
                           labels[0] if labels else "")
            if kind == "choice":
                var = tk.StringVar(value=current)
                box = ttk.Combobox(holder, values=labels, state="readonly", textvariable=var)
                box.grid(row=0, column=0, sticky="ew")
                box.bind("<<ComboboxSelected>>", lambda e: self._choice_changed())
                return lambda: values.get(var.get(), var.get())
            var = tk.StringVar(value=current)
            frame = ttk.Frame(holder)
            frame.grid(row=0, column=0, sticky="ew")
            for option, label, doc in options:
                button = ttk.Radiobutton(frame, text=label, value=label, variable=var,
                                         command=self._choice_changed)
                button.pack(anchor="w")
                if doc:
                    ttk.Label(frame, text=doc, style="Faint.TLabel").pack(anchor="w", padx=22)
            return lambda: values.get(var.get(), var.get())
        if kind == "multi":
            options = _options(field)
            selected = set(value or [])
            frame = ttk.Frame(holder)
            frame.grid(row=0, column=0, sticky="ew")
            variables = []
            for option, label, doc in options:
                var = tk.BooleanVar(value=option in selected)
                ttk.Checkbutton(frame, text=label, variable=var,
                                command=self._choice_changed).pack(anchor="w")
                if doc:
                    ttk.Label(frame, text=doc, style="Faint.TLabel").pack(anchor="w", padx=22)
                variables.append((option, var))
            return lambda: [option for option, var in variables if var.get()]
        if kind in ("int", "float", "sci"):
            text = "" if value is None else str(value)
            var = tk.StringVar(value=text)
            ttk.Entry(holder, textvariable=var).grid(row=0, column=0, sticky="ew")
            def number(cast):
                raw = var.get().strip()
                return None if not raw else cast(float(raw))
            if kind == "int":
                return lambda: number(int)
            return lambda: number(float)
        if kind == "json":
            text = json.dumps(value) if isinstance(value, (list, dict, tuple)) else ("" if value is None else str(value))
            var = tk.StringVar(value=text)
            ttk.Entry(holder, textvariable=var).grid(row=0, column=0, sticky="ew")
            def get_json():
                raw = var.get().strip()
                return json.loads(raw) if raw else None
            return get_json
        if kind == "path":
            var = tk.StringVar(value="" if value is None else str(value))
            entry = ttk.Entry(holder, textvariable=var)
            entry.grid(row=0, column=0, sticky="ew")
            ttk.Button(holder, text="…", width=3,
                       command=lambda: self._pick_into(var)).grid(row=0, column=1, padx=4)
            return var.get
        if kind == "paths":
            frame = ttk.Frame(holder)
            frame.grid(row=0, column=0, sticky="ew")
            box = tk.Listbox(frame, height=4, font=("TkFixedFont", 9), activestyle="none")
            box.grid(row=0, column=0, sticky="ew")
            frame.columnconfigure(0, weight=1)
            for one in value or []:
                box.insert("end", str(one))
            buttons = ttk.Frame(frame)
            buttons.grid(row=0, column=1, sticky="n")
            ttk.Button(buttons, text="Add…", width=7,
                       command=lambda: self._add_path(box)).pack(fill="x")
            ttk.Button(buttons, text="Remove", width=7, style="Ghost.TButton",
                       command=lambda: box.delete(tk.ACTIVE) if box.curselection() else None
                       ).pack(fill="x")
            return lambda: list(box.get(0, "end"))
        if kind == "tasks":
            return self._tasks_widget(holder, field, value)
        if kind == "mapping":
            frame = ttk.Frame(holder)
            frame.grid(row=0, column=0, sticky="ew")
            frame.columnconfigure(1, weight=1)
            keys = field.get("options") or list((value or {}).keys())
            entries = []
            for i, key in enumerate(keys):
                ttk.Label(frame, text=str(key)).grid(row=i, column=0, sticky="w", padx=(0, 8))
                current = (value or {}).get(key, field.get("default", "preserved"))
                var = tk.StringVar(value=str(current))
                ttk.Entry(frame, textvariable=var).grid(row=i, column=1, sticky="ew")
                entries.append((key, var))
            return lambda: {key: var.get() for key, var in entries}
        var = tk.StringVar(value="" if value is None else str(value))
        ttk.Entry(holder, textvariable=var).grid(row=0, column=0, sticky="ew")
        return var.get

    def _tasks_widget(self, holder, field, value):
        columns = field.get("columns") or [
            {"name": "type", "kind": "text", "label": "Task"},
            {"name": "name", "kind": "text", "label": "Name"},
            {"name": "data", "kind": "path", "label": "Data"},
        ]
        frame = ttk.Frame(holder)
        frame.grid(row=0, column=0, sticky="ew")
        rows = []

        def add_row(initial=None):
            initial = initial or {}
            row_frame = ttk.Frame(frame)
            row_frame.pack(fill="x", pady=1)
            ttk.Button(row_frame, text="✕", width=2, style="Ghost.TButton",
                       command=lambda f=row_frame: remove_row(f)).pack(side="right")
            widgets = {}
            for column in columns:
                name, ckind = column["name"], column.get("kind", "text")
                if ckind == "choice":
                    options = _options(column)
                    labels = [label for _value, label, _doc in options]
                    values = {label: option for option, label, _doc in options}
                    current = initial.get(name, column.get("default"))
                    label = next((l for o, l, _d in options if o == current),
                                 labels[0] if labels else "")
                    var = tk.StringVar(value=label)
                    box = ttk.Combobox(row_frame, values=labels, state="readonly",
                                       width=column.get("width", 14), textvariable=var)
                    box.pack(side="left", padx=2)
                    widgets[name] = lambda v=var, table=values: table.get(v.get(), v.get())
                elif ckind == "path":
                    box = ttk.Frame(row_frame)
                    box.pack(side="left", fill="x", expand=True, padx=2)
                    var = tk.StringVar(value=str(initial.get(name, column.get("default", "")) or ""))
                    entry = ttk.Entry(box, textvariable=var)
                    entry.pack(side="left", fill="x", expand=True)
                    ttk.Button(box, text="…", width=3,
                               command=lambda v=var: self._pick_into(v)).pack(side="left")
                    widgets[name] = var.get
                else:
                    var = tk.StringVar(value=str(initial.get(name, column.get("default", "")) or ""))
                    entry = ttk.Entry(row_frame, textvariable=var,
                                      width=column.get("width", 12))
                    entry.pack(side="left", padx=2)
                    widgets[name] = var.get
            rows.append((row_frame, widgets))
            return row_frame, widgets

        def remove_row(row_frame):
            for index, (widget, _w) in enumerate(rows):
                if widget is row_frame:
                    rows.pop(index)
                    break
            row_frame.destroy()

        for initial in value or [{}]:
            add_row(initial if isinstance(initial, dict) else {"type": initial})
        ttk.Button(frame, text="Add task", style="Ghost.TButton",
                   command=lambda: add_row({})).pack(anchor="w", pady=(3, 0))

        def values():
            out = []
            for _row_frame, widgets in rows:
                row = {}
                for name, getter in widgets.items():
                    try:
                        item = getter()
                    except (ValueError, tk.TclError):
                        item = None
                    if isinstance(item, str) and not item.strip():
                        item = ""
                    row[name] = item
                if row.get("type"):
                    out.append(row)
            return out
        return values

    def _pick_into(self, var):
        path = pick(self.app, self, "", (), None, "Choose a path")
        if path:
            var.set(path)

    def _add_path(self, box):
        path = pick(self.app, self, "", (".yaml", ".yml"), None, "Add a YAML config")
        if path:
            box.insert("end", path)

    def _choice_changed(self):
        self._collect()
        self._render()

    # ── values and navigation ─────────────────────────────────────────────────
    def _collect(self):
        for field, getter in self._widgets:
            try:
                value = getter()
            except (ValueError, tk.TclError):
                continue
            if isinstance(value, str) and not value.strip() and not field.get("required"):
                value = None
            self.answers[field["name"]] = value

    def _current_step(self):
        return self.steps[self.index] if self.steps and not self.summary else None

    def _step_errors(self):
        errors = []
        for field, getter in self._widgets:
            if not field.get("required"):
                continue
            if not _test(field.get("when"), self.answers):
                continue
            try:
                value = getter()
            except (ValueError, tk.TclError):
                value = None
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(field.get("label", field["name"]))
        return errors

    def back(self):
        if self.summary:
            self.summary = False
            self._render()
            return
        if self.index == 0:
            return
        self._collect()
        self.index -= 1
        self._fetch(keep=self.steps[self.index]["id"])

    def next(self):
        if self._busy:
            return
        self._collect()
        errors = self._step_errors()
        if errors:
            return self.app.toast("required: " + ", ".join(errors), error=True)
        if self.index < len(self.steps) - 1:
            self.index += 1
            self._fetch(keep=self.steps[self.index]["id"])
            return
        self.show_summary()

    def show_summary(self):
        self._collect()
        self.server.call("wizard.plan", {"id": self.wizard_id, "answers": self.answers},
                         self._planned)

    def _planned(self, plan):
        if not plan or isinstance(plan, tuple):
            return
        errors = plan.get("errors") or []
        if errors:
            return self.app.toast("; ".join(errors), error=True)
        self.summary = True
        self.step_title.configure(text="Summary")
        self.step_doc.configure(text="Review the work. Nothing has been queued yet.")
        for widget in self.form.winfo_children():
            widget.destroy()
        self._widgets = []
        head = ttk.Frame(self.form)
        head.pack(fill="x", pady=(0, 6))
        for line in plan.get("lines") or []:
            ttk.Label(head, text=line).pack(anchor="w")
        text = plan.get("yaml") or ""
        if not text and plan.get("commands"):
            text = "\n".join(f"mdat> {one['cmd']} {json.dumps(one.get('args') or {}, default=str)}"
                             for one in plan["commands"])
        if text:
            console = Console(self.app, self.form)
            console.pack(fill="both", expand=True)
            console.text.configure(height=18)
            console.put(text)
        self.step_list.selection_clear(0, "end")
        self.next_btn.configure(text="Queue job", style="Accent.TButton", command=self.submit)
        self.back_btn.configure(state="normal", text="< Back")
        self.back_btn.configure(command=self._leave_summary)

    def _leave_summary(self):
        self.summary = False
        self.next_btn.configure(text="Summary >", command=self.next)
        self.back_btn.configure(command=self.back)
        self._render()

    def submit(self):
        self.server.call("wizard.submit", {"id": self.wizard_id, "answers": self.answers},
                         self._submitted)

    def _submitted(self, answer):
        if not answer or isinstance(answer, tuple):
            return
        detail = answer.get("run") or answer.get("config") or ""
        self.app.toast(f"{answer.get('kind')}: {detail}")
        if answer.get("run"):
            self.app.toggle_dock(force=True)
        if self.on_done:
            self.on_done(answer)
        self.destroy()
