# matformer/gui/trainpage.py
"""Training control. A training usually needs more than one YAML (base + data + experiment): the
config list composes them, the override dialog (`override.py`, also reachable standalone) layers
parameters on top, and the queue dialog exposes the switches a run takes (GPUs, nodes, device,
checkpoint, compile…) — generated from `train.options`, so what the terminal can do the dialog can
do. Nothing starts until Queue is pressed: closing a dialog is not a command. Right-click a job:
export its log, open its curves in Plots, abort."""
import os
import tkinter as tk
from tkinter import ttk

from matformer.gui.explorer import pick
from matformer.gui.forms import Form
from matformer.gui.override import OverrideDialog
from matformer.gui.wizard import WizardDialog
from matformer.gui.widgets import Page, ScrollBox, TableList, fit_dialog, popup, show_payload


class TrainPage(Page):
    """Multi-config training launcher + live job queue."""

    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Training", "")
        self.overrides = []
        self.options = {}
        self.sweep = None          # which points of a sweep to run: None means every one of them
        self.sweep_runs = 0        # how many points the current configs declare, once asked
        self.project = "gui"
        self._build_actions()
        pan = ttk.PanedWindow(self.body, orient="horizontal")
        pan.pack(fill="both", expand=True)

        left = ttk.Frame(pan)
        pan.add(left, weight=2)
        ttk.Label(left, text="Configs (merged left to right)", style="Head.TLabel").pack(fill="x")
        self.cfgs = tk.Listbox(left, height=6, font=("TkFixedFont", 9),
                               selectmode=tk.EXTENDED, activestyle="none")
        self.cfgs.pack(fill="x")
        self.cfgs.bind("<Button-3>", self._cfg_menu)
        self.cfgs.bind("<Double-1>", lambda e: self.app.open_config(self.cfgs.get(self.cfgs.curselection()[0])
                                                                    if self.cfgs.curselection() else ""))
        row = ttk.Frame(left)
        row.pack(fill="x", pady=3)
        ttk.Button(row, text="Add…", width=6, command=self._add_cfg).pack(side="left")
        ttk.Button(row, text="Remove", width=7, style="Ghost.TButton", command=self._rm_cfg).pack(side="left", padx=3)
        ttk.Button(row, text="Clear", width=6, style="Ghost.TButton", command=lambda: self.cfgs.delete(0, "end")).pack(side="left")
        ttk.Label(left, text="Overrides", style="Head.TLabel").pack(fill="x", pady=(6, 0))
        self.ov_list = tk.Listbox(left, height=4, font=("TkFixedFont", 9), activestyle="none")
        self.ov_list.pack(fill="x")
        orow = ttk.Frame(left)
        orow.pack(fill="x", pady=3)
        ttk.Button(orow, text="Edit overrides…", command=self._edit_overrides).pack(side="left")

    def _build_actions(self):
        ttk.Button(self.actions, text="Wizard…", style="Accent.TButton",
                   command=lambda: WizardDialog(self.app, "train")).pack(side="left", padx=3)
        ttk.Button(self.actions, text="Queue configs", command=self.queue_dialog).pack(side="left")
        more = ttk.Menubutton(self.actions, text="More", style="Ghost.TButton")
        menu = tk.Menu(more, tearoff=0)
        menu.add_command(label="Sweep…", command=self.sweep_dialog)
        menu.add_command(label="Memory & speed probe…", command=self.probe_dialog)
        menu.add_command(label="Probe reports…", command=self.probe_reports)
        more["menu"] = menu
        more.pack(side="left", padx=3)

    @property
    def server(self):
        return self.app.server

    def _add_cfg(self):
        p = pick(self.app, self, "", (".yaml", ".yml"), None, "Add a config (base first)")
        if p and p not in self.cfgs.get(0, "end"):
            self.cfgs.insert("end", p)

    def _rm_cfg(self):
        for i in reversed(list(self.cfgs.curselection())):
            self.cfgs.delete(i)

    def _cfg_menu(self, e):
        index = self.cfgs.nearest(e.y)
        if index == "@1":
            return
        self.cfgs.selection_clear(0, "end")
        self.cfgs.selection_set(index)
        path = self.cfgs.get(index)
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Open in Config Editor", command=lambda: self.app.open_config(path))
        m.add_command(label="Move up", command=lambda: self._move(index, -1))
        m.add_command(label="Move down", command=lambda: self._move(index, 1))
        m.add_command(label="Remove", command=lambda: self.cfgs.delete(index))
        m.add_command(label="Copy path", command=lambda: self.app.copy(path))
        popup(m, e.x_root, e.y_root)

    def _move(self, index, delta):
        target = index + delta
        if target < 0 or target >= self.cfgs.size():
            return
        value = self.cfgs.get(index)
        self.cfgs.delete(index)
        self.cfgs.insert(target, value)
        self.cfgs.selection_set(target)

    def activated(self):
        self._subtitle()

    def server_changed(self):
        self._subtitle()

    def _edit_overrides(self):
        dlg = OverrideDialog(self.app, self.server, initial=self.overrides,
                             title="Overrides", accept="Keep")
        self.wait_window(dlg)
        if dlg.result is not None:
            self.set_overrides(dlg.result)

    def set_overrides(self, overrides):
        self.overrides = list(overrides)
        self.ov_list.delete(0, "end")
        for o in self.overrides:
            self.ov_list.insert("end", o)
        self._subtitle()

    def _subtitle(self):
        """What the page is about to queue, in one line: the configs, the overrides, and the sweep
        they expand into once the server has been asked (a sweep is only known after a read, so the
        count appears when it is known instead of claiming zero)."""
        parts = ([f"{self.cfgs.size()} configs"] if self.cfgs.size() else []) + \
            [f"{len(self.overrides)} overrides"]
        if self.sweep_runs > 1:
            parts.append(f"sweep: {self.sweep_runs} runs")
        if self.sweep:
            parts.append(f"selected: {self.sweep}")
        self.set_subtitle("   ·   ".join(parts))

    # ── sweeps ─────────────────────────────────────────────────────────────────
    def sweep_dialog(self):
        SweepDialog(self)

    def ask_sweep(self, then=None):
        """How many runs these configs declare, asked of the server: `then` gets the answer, and
        the page remembers it for the subtitle and the queue dialog."""
        def got(payload):
            if payload and not isinstance(payload, tuple):
                self.sweep_runs = payload.get("runs") or 0
                self._subtitle()
            if then:
                then(payload)
        self.server.call("train.sweep", self._sweep_params(), got)

    def _sweep_params(self):
        return {"config": list(self.cfgs.get(0, "end")), "overrides": list(self.overrides),
                "options": dict(self.options)}

    # ── queueing ───────────────────────────────────────────────────────────────
    def add_config(self, path):
        if path not in self.cfgs.get(0, "end"):
            self.cfgs.insert("end", path)
        self._subtitle()

    def name_default(self):
        """Suggest a run name from the last (most specific) config in the list."""
        paths = list(self.cfgs.get(0, "end"))
        return os.path.basename(paths[-1]).rsplit(".", 1)[0] if paths else "training"

    def queue_dialog(self):
        QueueDialog(self)

    def start(self, configs=None, overrides=None, options=None, name=None, project=None, probe=None,
              sweep=None, message="queued"):
        cfgs = configs if configs is not None else list(self.cfgs.get(0, "end"))
        if not cfgs:
            return self.app.toast("add at least one config file", error=True)
        params = {"config": cfgs, "overrides": overrides if overrides is not None else self.overrides,
                  "options": options if options is not None else self.options,
                  "sweep": sweep if sweep is not None else self.sweep,
                  "name": (name or "").strip() or None,
                  "project": (project or "").strip() or "gui"}
        params.update({k: v for k, v in (params["options"] or {}).items()
                       if k in ("device", "gpus", "save_model", "save_plugins", "sweep")})
        params["options"] = {k: v for k, v in (params["options"] or {}).items()
                             if k not in ("device", "gpus", "save_model", "save_plugins", "sweep")}
        if probe:
            params["probe"] = {k: v for k, v in probe.items() if v is not None}
        def queued(result):
            if not result or isinstance(result, tuple):
                return
            self.app.toast(f"{message}: {result.get('run')}")
            self.app.pages["jobs"].focus_job(result.get("run"), self.app.server_var.get())
        self.server.call("train.start", params, queued)

    # ── memory & speed probe ────────────────────────────────────────────────
    def probe_dialog(self):
        ProbeDialog(self)

    def probe_reports(self):
        """Every probe this server's roots know about; double-click one for its full answer."""
        self._payload_window("memory.reports", "Memory probes", on_open=self._probe_report)

    def _probe_report(self, row):
        if row:
            self._payload_window("memory.report", f"probe · {row.get('run', '')}",
                                 {"path": row.get("path")})

    def _payload_window(self, method, title, params=None, on_open=None):
        dlg = tk.Toplevel(self.app)
        dlg.title(title)
        dlg.geometry("900x460")
        dlg.transient(self.app)
        holder = ttk.Frame(dlg)
        holder.pack(fill="both", expand=True, padx=8, pady=8)
        placeholder = ttk.Label(holder, text="asking the server…", style="Muted.TLabel")
        placeholder.pack(anchor="w")
        bar = ttk.Frame(dlg)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Close", command=dlg.destroy).pack(side="right")
        if on_open is not None:
            ttk.Label(bar, text="double-click a row for its detail",
                      style="Faint.TLabel").pack(side="left")

        def got(payload):
            if not payload or isinstance(payload, tuple):
                show_payload(self.app, holder, str(payload or "no answer from the server"))
                return
            table = show_payload(self.app, holder, payload)
            if on_open is not None and hasattr(table, "tree"):
                table.tree.bind("<Double-1>", lambda e: on_open(table.selected()))
        self.server.call(method, params, got)
        dlg.bind("<Escape>", lambda e: dlg.destroy())


class QueueDialog(tk.Toplevel):
    """The last step before a training: name, project, and the switches the server says a run
    accepts. Cancel leaves the queue untouched."""

    def __init__(self, page):
        super().__init__(page)
        self.page, self.app = page, page.app
        self.title("Queue training")
        self.geometry("560x560")
        self.minsize(460, 400)
        self.transient(page)
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=12, pady=10)
        ttk.Button(bottom, text="Queue", style="Accent.TButton", command=self._queue).pack(side="right", padx=4)
        ttk.Button(bottom, text="Cancel", command=self.destroy).pack(side="right")
        head = ttk.Frame(self)
        head.pack(side="top", fill="x", padx=12, pady=(12, 4))
        head.columnconfigure(1, weight=1)
        for row, (label, value) in enumerate((("Run name", page.name_default()),
                                              ("Project", page.project))):
            ttk.Label(head, text=label + ":").grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(head)
            entry.insert(0, value)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            setattr(self, label.split()[0].lower(), entry)
        summary = " + ".join(self.page.cfgs.get(0, "end")) or "no config selected"
        ttk.Label(self, text=summary, style="Muted.TLabel", wraplength=520, justify="left").pack(
            side="top", anchor="w", padx=12, pady=(4, 0))
        swept = f"{len(self.page.overrides)} overrides"
        if self.page.sweep_runs or self.page.sweep:
            swept += f"   ·   sweep: {self.page.sweep_runs or '?'} runs" + (
                f", selected: {self.page.sweep}" if self.page.sweep else "")
        ttk.Label(self, text=swept, style="Muted.TLabel").pack(side="top", anchor="w", padx=12)
        scroll = ScrollBox(self.app, self, width=10)
        scroll.pack(side="top", fill="both", expand=True, padx=12, pady=8)
        self.form, self.fields = None, []
        scroll.inner.columnconfigure(0, weight=1)
        self.holder = ttk.Frame(scroll.inner)
        self.holder.pack(fill="x", anchor="n")
        self._cluster_fields_label = ttk.Label(self, text="", style="Faint.TLabel")
        self._cluster_fields_label.pack(side="top", anchor="w", padx=12)
        self.app.server.call("train.options", None, self._local_fields)
        self.app.server.call("slurm.fields", None, self._cluster_fields)
        self.bind("<Escape>", lambda e: self.destroy())
        fit_dialog(self, 560, 0)

    def _local_fields(self, payload):
        if not payload or isinstance(payload, tuple):
            return
        self.fields = list(payload["fields"])
        self._draw()
        self._cluster_fields_label.configure(text="reading cluster fields…")

    def _cluster_fields(self, payload):
        """The allocation switches arrive separately; a slow cluster probe must not keep the
        local queue dialog closed."""
        if not payload or isinstance(payload, tuple):
            self._cluster_fields_label.configure(text="")
            return
        extra = [{"name": "backend", "label": "Runs on", "kind": "choice",
                  "options": ["this box", "cluster"], "default": "this box",
                  "doc": "cluster hands the job to SLURM"}]
        extra += list(payload.get("fields") or [])
        names = {f["name"] for f in self.fields}
        self.fields = [f for f in extra if f["name"] not in names] + self.fields
        self._cluster_fields_label.configure(text="")
        self._draw()
        fit_dialog(self, 560, 0)

    def _draw(self):
        values = self.form.values() if self.form else {}
        for f in self.fields:
            if f["name"] in values:
                f["default"] = values[f["name"]]
            elif f["name"] in self.page.options:
                f["default"] = self.page.options[f["name"]]
        if self.form:
            self.form.destroy()
        self.form = Form(self.app, self.holder, self.fields)
        self.form.pack(fill="x")

    def _queue(self):
        if not self.page.cfgs.size():
            return self.app.toast("add at least one config file", error=True)
        options = self.form.values() if self.form else {}
        self.page.options = options
        self.page.start(name=self.Run.get(), project=self.Project.get(), options=options)
        self.destroy()


class SweepDialog(tk.Toplevel):
    """The sweep a set of configs declares, expanded before any of it runs: one row per run, and
    what that row changes. The answer comes from the server building the very configs a launch
    would build, so the preview cannot drift from the real thing -- and no model is trained to give
    it. Queue the whole product as one job, or point the page at a single combination (which is
    also what a cluster array job wants: one process per point, one shelf per point)."""

    def __init__(self, page):
        super().__init__(page)
        self.page, self.app = page, page.app
        self.title("Sweep of these configs")
        self.geometry("780x460")
        self.minsize(560, 300)
        self.transient(page)
        head = ttk.Frame(self)
        head.pack(side="top", fill="x", padx=10, pady=(10, 2))
        self.info = ttk.Label(head, text="reading the configs…", style="Muted.TLabel")
        self.info.pack(side="left")
        ttk.Button(head, text="Reload", width=7, style="Ghost.TButton",
                   command=self._ask).pack(side="right")
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=10, pady=8)
        ttk.Button(bottom, text="Queue every run", style="Accent.TButton",
                   command=self._queue_all).pack(side="right", padx=4)
        ttk.Button(bottom, text="Queue this run", command=self._queue_one).pack(side="right")
        ttk.Button(bottom, text="Close", style="Ghost.TButton", command=self.destroy).pack(side="left")
        self.table = TableList(self.app, self, on_double=lambda row: self._queue(row), height=12)
        self.table.headings(["#", "run", "what it changes"])
        self.table.fill([], "reading…")
        self.table.pack(fill="both", expand=True, padx=10, pady=4)
        self.bind("<Escape>", lambda e: self.destroy())
        self._ask()

    def _ask(self):
        self.server.call("train.sweep", self.page._sweep_params(), self._filled)

    @property
    def server(self):
        return self.page.server

    def _filled(self, payload):
        if not payload or isinstance(payload, tuple):
            return self.info.config(text="the server has no answer")
        runs = payload.get("runs") or 0
        self.page.sweep_runs = runs
        self.page._subtitle()
        self.info.config(text=f"{runs} runs" + (f" over {', '.join(payload['dimensions'])}"
                                                if runs > 1 else ": a plain single run"))
        self.table.fill([{"#": point["index"], "run": point["name"],
                          "what": point["describe"], "name": point["name"]}
                         for point in payload.get("points") or []],
                        "no sweep declared: these configs are one run")

    def _queue(self, row):
        if not row:
            return
        self.page.sweep = row["name"]
        self.destroy()
        self.page.queue_dialog()

    def _queue_one(self):
        row = self.table.selected()
        if not row:
            return self.app.toast("select a run first", error=True)
        self._queue(row)

    def _queue_all(self):
        self.page.sweep = None
        self.destroy()
        self.page.queue_dialog()


class ProbeDialog(tk.Toplevel):
    """What these GPUs can actually run: the probe's own knobs (how much memory to reach, how many
    steps to confirm it with) above the switches a training takes, because a probe *is* a training spec
    that stops after measuring. The answer arrives in the job's log; `apply` makes the same job go on
    and train at the size it found."""

    def __init__(self, page):
        super().__init__(page)
        self.page, self.app = page, page.app
        self.title("Memory & speed probe")
        self.geometry("580x600")
        self.minsize(460, 420)
        self.transient(page)
        self.fields, self.probe_names, self.form, self.waiting = [], set(), None, 2
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=12, pady=10)
        ttk.Button(bottom, text="Run probe", style="Accent.TButton", command=self._run).pack(side="right", padx=4)
        ttk.Button(bottom, text="Cancel", command=self.destroy).pack(side="right")
        summary = " + ".join(self.page.cfgs.get(0, "end")) or "no config selected"
        ttk.Label(self, text=summary, style="Muted.TLabel", wraplength=540, justify="left").pack(
            side="top", anchor="w", padx=12, pady=(12, 0))
        ttk.Label(self, text="mock tokens: no document is read, and nothing is saved",
                  style="Faint.TLabel").pack(side="top", anchor="w", padx=12, pady=(2, 0))
        scroll = ScrollBox(self.app, self, width=10)
        scroll.pack(side="top", fill="both", expand=True, padx=12, pady=8)
        scroll.inner.columnconfigure(0, weight=1)
        self.holder = ttk.Frame(scroll.inner)
        self.holder.pack(fill="x", anchor="n")
        self.app.server.call("train.probeOptions", None, lambda r: self._schema(r, probe=True))
        self.app.server.call("train.options", None, lambda r: self._schema(r))
        self.bind("<Escape>", lambda e: self.destroy())
        fit_dialog(self, 580, 0)

    def _schema(self, payload, probe=False):
        """Both schemas arrive on their own schedule: keep them, draw the form once the last is in."""
        if payload and not isinstance(payload, tuple):
            fields = list(payload.get("fields") or [])
            if probe:
                self.probe_names = {field["name"] for field in fields}
            self.fields += fields
        self.waiting -= 1
        if self.waiting:
            return
        for field in self.fields:
            if field["name"] in self.page.options:
                field["default"] = self.page.options[field["name"]]
        self.form = Form(self.app, self.holder, self.fields)
        self.form.pack(fill="x")
        fit_dialog(self, 560, 0)

    def _run(self):
        if not self.page.cfgs.size():
            return self.app.toast("add at least one config file", error=True)
        values = self.form.values() if self.form else {}
        probe = {name: values.pop(name) for name in list(values) if name in self.probe_names}
        self.page.start(options=values, name=f"probe-{self.page.name_default()}",
                        project=self.page.project, probe=probe, message="probe queued")
        self.destroy()
