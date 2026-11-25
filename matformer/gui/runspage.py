# matformer/gui/runspage.py
"""Runs & Plots: the workspace over the run database, built on `panels.py`.

Ticked runs are tailed live (`runs.watch` → `metric` events). Every picture is a *panel* — kind,
metric globs, split, aggregation, axis, smoothing — drawn into its own canvas inside a scrolling
grid, so line charts, bar comparisons and tables coexist and each one is exported, resized and
edited on its own. Hovering reads the exact value, dragging zooms the x axis, double-click undoes,
right-click gives the panel's menu. A whole view saves and loads as JSON.

This is also where interpretability will land: a direct-logit-attribution graph or an outlier-token
table is another panel kind with another source, and the grid, the menus, the export and the saved
views come with it.
"""
import csv
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matformer.gui import panels as P
from matformer.gui.forms import Form
from matformer.gui.widgets import Console, Page, ScrollBox, TableList, fit_dialog, placeholder, popup, tip
from matformer.utils.metrics import last
from matformer.gui.analysis import AnalysisView
from matformer.gui.tkplot import TkPlotCanvas

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def _field_text(entry):
    """The typed text, or '' when the entry still shows its placeholder."""
    text = entry.get()
    if text == getattr(entry, "_ph", None):
        return ""
    return text.strip().lower()


class RunsPage(Page):
    """Run browser + panel grid + wide comparison table."""

    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Runs & Plots", "")
        self.metric_names, self.keys_visible = [], []
        self.series, self.meta, self.configs = {}, {}, {}
        self.panels, self.ticked = [], set()
        self.focus, self.baseline = None, None
        self._wide_columns = None
        self._canvases = []
        self._dirty = False
        self._auto_ticked = False
        self._auto_done = False
        self._build_actions()
        pan = ttk.PanedWindow(self.body, orient="horizontal")
        pan.pack(fill="both", expand=True)
        self._build_left(pan)
        self._build_right(pan)
        app.register_paintable(self)

    def _build_actions(self):
        ttk.Button(self.actions, text="Recent", style="Ghost.TButton",
                   command=self.select_recent).pack(side="left", padx=3)
        ttk.Button(self.actions, text="Add panel…", style="Accent.TButton",
                   command=self.add_panel).pack(side="left", padx=3)
        ttk.Button(self.actions, text="View…", style="Ghost.TButton",
                   command=self._view_menu).pack(side="left")
        ttk.Button(self.actions, text="Gallery", style="Ghost.TButton",
                   command=self._gallery).pack(side="left", padx=3)
        ttk.Button(self.actions, text="Remove…", style="Ghost.TButton",
                   command=self._remove_selected).pack(side="left", padx=3)

    # ── left: what to look at ─────────────────────────────────────────────────
    def _build_left(self, pan):
        left = ttk.Frame(pan)
        pan.add(left, weight=2)
        top = ttk.Frame(left)
        top.pack(fill="x")
        ttk.Button(top, text="⟳", width=3, style="Ghost.TButton", command=self.refresh).pack(side="left")
        self.cat_all = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="All servers", variable=self.cat_all,
                        command=self.refresh).pack(side="left", padx=6)
        self.project = ttk.Combobox(left, values=["all projects"], state="readonly")
        self.project.pack(fill="x", pady=2)
        self.project.bind("<<ComboboxSelected>>", lambda e: self._show_runs())
        self.search = ttk.Entry(left)
        self.search.pack(fill="x", pady=2)
        placeholder(self.search, "filter runs by name or project")
        self.search.bind("<KeyRelease>", lambda e: self._show_runs())
        self.table = TableList(self.app, left, height=10)
        self.table.headings(["run", "project", "steps", "server"])
        self.table.tree.bind("<Button-1>", self._click_tick)
        self.table.tree.bind("<Button-3>", self._run_menu)
        self.table.pack(fill="both", expand=True, pady=(0, 4))
        row = ttk.Frame(left)
        row.pack(fill="x")
        for label, command in (("All", lambda: self._set_all(True)),
                               ("None", lambda: self._set_all(False)),
                               ("Focus", self._focus_selected),
                               ("Baseline", self._baseline_selected),
                               ("Clear", self._clear_marks)):
            ttk.Button(row, text=label, width=7, style="Ghost.TButton", command=command).pack(side="left", padx=2)
        self.marks = ttk.Label(left, text="no focus, no baseline", style="Faint.TLabel")
        self.marks.pack(anchor="w", pady=(2, 0))
        ttk.Label(left, text="Run config", style="Head.TLabel").pack(fill="x", pady=(6, 0))
        self.config_view = Console(self.app, left)
        self.config_view.text.configure(height=8)
        self.config_view.pack(fill="both", expand=True)

    # ── right: controls, panels, table ────────────────────────────────────────
    def _build_right(self, pan):
        right = ttk.Frame(pan)
        pan.add(right, weight=4)
        ctrl = ttk.Frame(right)
        ctrl.pack(fill="x")
        ttk.Label(ctrl, text="Smooth").pack(side="left")
        self.smooth = ttk.Scale(ctrl, from_=0, to=0.97, length=110)
        self.smooth.set(0.6)
        self.smooth.pack(side="left", padx=(4, 8))
        self.smooth.bind("<ButtonRelease-1>", lambda e: (self._apply_default_smooth(), self.redraw()))
        self.logy = tk.BooleanVar()
        ttk.Checkbutton(ctrl, text="Log-Y", variable=self.logy,
                        command=self._apply_globals).pack(side="left", padx=3)
        self.xlink = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="link x", variable=self.xlink).pack(side="left", padx=3)
        ttk.Label(ctrl, text="X").pack(side="left", padx=(8, 2))
        self.xaxis = ttk.Combobox(ctrl, values=list(P.XAXES), state="readonly", width=6)
        self.xaxis.set("step")
        self.xaxis.pack(side="left")
        self.xaxis.bind("<<ComboboxSelected>>", lambda e: self._apply_globals())
        ttk.Label(ctrl, text="cols").pack(side="left", padx=(8, 2))
        self.cols = ttk.Spinbox(ctrl, from_=1, to=4, width=3, command=self.redraw)
        self.cols.set(2)
        self.cols.pack(side="left")

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True, pady=3)
        self.grid_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.grid_tab, text="Panels")
        split = ttk.PanedWindow(self.grid_tab, orient="horizontal")
        split.pack(fill="both", expand=True)

        side = ttk.Frame(split)
        split.add(side, weight=0)
        ttk.Label(side, text="Panels", style="Head.TLabel").pack(fill="x")
        self.panel_filter = ttk.Entry(side, width=26)
        self.panel_filter.pack(fill="x", pady=2)
        self.panel_filter.bind("<KeyRelease>", lambda e: self._show_panels())
        panel_box = ttk.Frame(side)
        panel_box.pack(fill="x")
        self.panel_list = tk.Listbox(panel_box, font=("TkDefaultFont", 9), width=28, height=8,
                                     activestyle="none")
        panel_bar = ttk.Scrollbar(panel_box, orient="vertical", command=self.panel_list.yview)
        self.panel_list.configure(yscrollcommand=panel_bar.set)
        panel_bar.pack(side="right", fill="y")
        self.panel_list.pack(side="left", fill="x", expand=True)
        self.panel_list.bind("<<ListboxSelect>>", lambda e: self._toggle_panel())
        self.panel_list.bind("<Double-1>", lambda e: PanelDialog(self, self._panel_at()))
        controls = ttk.Frame(side)
        controls.pack(fill="x", pady=2)
        for label, command in (("↑", lambda: self._move_panel(-1)), ("↓", lambda: self._move_panel(1)),
                               ("edit…", lambda: PanelDialog(self, self._panel_at())),
                               ("dup", self._duplicate_panel), ("✕", self._delete_panel)):
            ttk.Button(controls, text=label, width=4, style="Ghost.TButton", command=command).pack(side="left", padx=1)
        ttk.Label(side, text="Available metrics", style="Head.TLabel").pack(fill="x", pady=(8, 2))
        self.metric_filter = ttk.Entry(side, width=26)
        self.metric_filter.pack(fill="x", pady=2)
        placeholder(self.metric_filter, "filter metrics")
        self.metric_filter.bind("<KeyRelease>", lambda e: self._show_metrics())
        key_box = ttk.Frame(side)
        key_box.pack(fill="both", expand=True)
        self.keys = tk.Listbox(key_box, selectmode="multiple", font=("TkDefaultFont", 9), width=28)
        key_bar = ttk.Scrollbar(key_box, orient="vertical", command=self.keys.yview)
        self.keys.configure(yscrollcommand=key_bar.set)
        key_bar.pack(side="right", fill="y")
        self.keys.pack(side="left", fill="both", expand=True)
        self.keys.bind("<Double-1>", lambda e: self._plot_selected_metrics())
        ttk.Label(side, text="double-click a metric to plot it", style="Faint.TLabel").pack(anchor="w")

        plots_side = ttk.Frame(split)
        split.add(plots_side, weight=1)
        self.grid_holder = ScrollBox(self.app, plots_side, width=10)
        self.grid_holder.pack(fill="both", expand=True)
        self.grid = tk.Frame(self.grid_holder.inner, bg=self.app.palette()["bg"])
        self.grid.pack(fill="both", expand=True)

        runs_tab = ttk.Frame(self.tabs)
        self.tabs.add(runs_tab, text="Runs")
        head = ttk.Frame(runs_tab)
        head.pack(fill="x")
        ttk.Button(head, text="Columns…", style="Ghost.TButton", width=9,
                   command=self._choose_columns).pack(side="right")
        ttk.Button(head, text="Export CSV…", style="Ghost.TButton",
                   command=self.export_csv).pack(side="right", padx=3)
        self.wide = TableList(self.app, runs_tab, height=12)
        self.wide.fill([])
        self.wide.tree.bind("<Button-3>", self._wide_menu)
        self.wide.pack(fill="both", expand=True)

    # ── data intake ───────────────────────────────────────────────────────────
    @property
    def server(self):
        return self.app.server

    def activated(self):
        self.refresh()

    def server_changed(self):
        self.refresh()

    def refresh(self):
        name = self.app.server_var.get()
        self.server.call("runs.list", None, lambda rows, n=name: self._filled(rows, n))

    def catalog_reload(self):
        self.refresh()

    def _filled(self, rows, server_name=None):
        if isinstance(rows, tuple) and rows and rows[0] == "error":
            self.meta = {}
            self.table.headings(["run", "mark", "project", "sweep", "steps", "server"])
            self.table.fill([], f"server error: {rows[1]}")
            self.wide.fill([], "server error")
            return
        rows = list(rows) if rows else []
        if server_name:
            for row in rows:
                row.setdefault("server", server_name)
        if self.cat_all.get():
            try:
                from matformer.utils.catalog import Catalog
                seen = {row["id"] for row in rows}
                for cached in Catalog().all():
                    if cached["id"] not in seen:
                        rows.append({"id": cached["id"], "run": cached["run"],
                                     "project": cached["project"], "version": cached["version"],
                                     "started": cached["started"], "steps": cached["steps"],
                                     "server": cached["server"], "offline": True})
            except Exception:
                pass
        self.meta = {row["id"]: row for row in rows}
        projects = ["all projects"] + sorted({row["project"] for row in rows if row.get("project")})
        self.project.configure(values=projects)
        if not self.project.get():
            self.project.set("all projects")
        self._show_runs()
        self._show_wide()

    def _show_runs(self):
        proj, text = self.project.get(), _field_text(self.search)
        rows = []
        for run in self.meta.values():
            if proj not in ("all projects", "", None) and run.get("project") != proj:
                continue
            # the sweep point is searchable too: typing its name leaves the whole product on screen
            if text and text not in run["run"].lower() \
                    and text not in (run.get("sweep") or "").lower():
                continue
            mark = ("★" if run["id"] == self.focus else "") + ("◆" if run["id"] == self.baseline else "")
            rows.append({"run": ("☑ " if run["id"] in self.ticked else "☐ ") + run["run"],
                         "project": run.get("project", ""), "sweep": run.get("sweep") or "",
                         "steps": run.get("steps") or 0,
                         "server": run.get("server", ""), "id": run["id"],
                         "mark": mark, "offline": run.get("offline")})
        self.table.headings(["run", "mark", "project", "sweep", "steps", "server"])
        self.table.fill(rows, "no runs")
        if rows and not self.ticked and not self._auto_ticked:
            self._auto_ticked = True
            wanted = [row for row in rows[:3] if not row.get("offline")]
            self.after(300, lambda rows=wanted: [self._toggle(row) for row in rows
                                                 if row.get("id") not in self.ticked])

    def select_recent(self):
        for row in list(self.meta.values())[:3]:
            if not row.get("offline") and row.get("id") not in self.ticked:
                self._toggle({"id": row["id"], "offline": False})

    def _click_tick(self, e):
        if self.table.tree.identify_column(e.x) != "#1":
            return
        iid = self.table.tree.identify_row(e.y)
        if iid:
            self.table.tree.selection_set(iid)
            self._toggle(self.table.selected())

    def _toggle(self, row):
        if not row:
            return
        run = row["id"]
        if row.get("offline"):
            return self.app.toast(f"{run}: server offline")
        if run in self.ticked:
            self.ticked.discard(run)
            self.server.call("runs.unwatch", {"run": run})
        else:
            self.ticked.add(run)
            self.server.call("runs.watch", {"run": run})
            self.server.call("runs.config", {"run": run}, lambda cfg, r=run: self._cfg(r, cfg))
        self._show_runs()
        self._show_wide()
        self.redraw()

    def _cfg(self, run, cfg):
        if cfg and not isinstance(cfg, tuple):
            self.configs[run] = cfg
            if not self.focus or self.focus == run:
                self._show_cfg(cfg)
            self._show_wide()

    def _show_cfg(self, cfg):
        self.config_view.clear()
        self.config_view.put("\n".join(f"{key}: {value}" for key, value in sorted(cfg.items())[:300]))

    def _set_all(self, on):
        for row in list(self.table.rows):
            if (row["id"] in self.ticked) != on:
                self._toggle(row)

    def focus_run(self, run_id):
        for rid, meta in self.meta.items():
            if run_id in rid or run_id in (meta.get("run") or ""):
                if rid not in self.ticked and not meta.get("offline"):
                    self.ticked.add(rid)
                    self.server.call("runs.watch", {"run": rid})
                self.focus = rid
        self._show_runs()
        self._paint_marks()
        self.redraw()

    def _focus_selected(self):
        row = self.table.selected()
        if row:
            self.focus = row["id"]
            self._paint_marks()
            self.redraw()

    def _baseline_selected(self):
        row = self.table.selected()
        if row:
            self.baseline = row["id"]
            self._paint_marks()
            self._show_wide()
            self.redraw()

    def _clear_marks(self):
        self.focus, self.baseline = None, None
        self._paint_marks()
        self._show_wide()
        self.redraw()

    def _paint_marks(self):
        self.marks.configure(text=(f"focus: {self.focus or '—'}   baseline: {self.baseline or '—'}"))

    def on_event(self, server, name, data):
        if name != "metric" or data.get("run") not in self.ticked:
            return
        for x, y in data["points"]:
            self.series.setdefault((data["run"], data["key"]), []).append((x, y))
        self._dirty = True
        if not getattr(self, "_redraw_job", None):
            self._redraw_job = self.after(200, self._throttled_redraw)

    def _throttled_redraw(self):
        self._redraw_job = None
        if self._dirty:
            self._dirty = False
            self._show_metrics()
            self._show_wide()
            self._auto_panels()
            self.redraw()

    def _auto_panels(self):
        if self.panels or self._auto_done:
            return
        keys = sorted({key for (_run, key) in self.series})
        if not keys:
            return
        self._auto_done = True
        for glob, title, span in (("*loss", "Loss", 2), ("lr", "Learning rate", 1),
                                  ("*grad*", "Gradient norm", 1), ("*accuracy", "Accuracy", 1)):
            if any(P.matches(key, glob) for key in keys):
                self.panels.append({**P.defaults(), "title": title, "metrics": glob,
                                    "span": span, "smooth": 0.4})
        self._show_panels()

    # ── panel management ──────────────────────────────────────────────────────
    def context(self):
        return P.Context(runs=lambda: sorted(self.ticked),
                         series=lambda run, key: tuple(zip(*self.series.get((run, key), [])))
                         if self.series.get((run, key)) else ([], []),
                         keys=lambda: sorted({key for (_run, key) in self.series}),
                         config=lambda run: self.configs.get(run, {}),
                         rpc=lambda method, params=None: self._rpc_table(method, params))

    def _rpc_table(self, method, params):
        """A panel whose data is a server command: asked once, answered with a Table dict. Kept
        synchronous on purpose — panels redraw on demand, and interpretability answers are fetched
        when asked, not streamed."""
        box = {}
        self.server.call(method, params or {}, lambda result: box.setdefault("value", result))
        for _ in range(60):
            if "value" in box:
                break
            self.update(20)
        value = box.get("value")
        return value if not isinstance(value, tuple) else None

    def _show_panels(self):
        filter_text = self.panel_filter.get().lower()
        self.keys_visible = []
        self.panel_list.delete(0, "end")
        for index, spec in enumerate(self.panels):
            title = spec.get("title") or spec.get("metrics", "panel")
            if filter_text and filter_text not in title.lower():
                continue
            self.keys_visible.append(index)
            self.panel_list.insert("end", ("☐ " if not spec.get("visible", True) else "") + title)

    def _panel_at(self):
        selection = self.panel_list.curselection()
        if not selection or not getattr(self, "keys_visible", None):
            return None
        try:
            return self.panels[self.keys_visible[selection[0]]]
        except IndexError:
            return None

    def _toggle_panel(self):
        spec = self._panel_at()
        if spec:
            spec["visible"] = not spec.get("visible", True)
            self._show_panels()
            self.redraw()

    def _move_panel(self, delta):
        selection = self.panel_list.curselection()
        if not selection:
            return
        i, j = selection[0], selection[0] + delta
        if 0 <= j < len(self.keys_visible):
            a, b = self.keys_visible[i], self.keys_visible[j]
            self.panels[a], self.panels[b] = self.panels[b], self.panels[a]
            self._show_panels()
            self.panel_list.selection_set(j)
            self.redraw()

    def _duplicate_panel(self):
        spec = self._panel_at()
        if spec:
            self.panels.append(dict(spec, title=(spec.get("title") or "panel") + " (copy)"))
            self._show_panels()
            self.redraw()

    def _delete_panel(self):
        spec = self._panel_at()
        if spec in self.panels:
            self.panels.remove(spec)
            self._show_panels()
            self.redraw()

    def _show_metrics(self):
        filter_text = _field_text(self.metric_filter)
        self.metric_names = [key for key in sorted({k for (_run, k) in self.series})
                             if not filter_text or filter_text in key.lower()]
        self.keys.delete(0, "end")
        for key in self.metric_names:
            self.keys.insert("end", key)

    def _plot_selected_metrics(self):
        chosen = [self.metric_names[i] for i in self.keys.curselection()] or []
        for key in chosen:
            self.panels.append(dict(P.defaults(), title=key, metrics=key))
        self._show_panels()
        self.redraw()

    def add_panel(self, preset=None):
        PanelDialog(self, preset)

    def new_panel(self, spec):
        for index, existing in enumerate(self.panels):
            if existing is spec:
                self.panels[index] = spec
                break
        else:
            self.panels.append(spec)
        self._show_panels()
        self.redraw()

    def _apply_globals(self):
        for spec in self.panels:
            spec["logy"] = self.logy.get()
            spec["xaxis"] = self.xaxis.get()
        self.redraw()

    def _apply_default_smooth(self):
        for spec in self.panels:
            spec["smooth"] = float(self.smooth.get())
        self.redraw()

    def _gallery(self):
        """Every kind a panel can take, with the source options: what this workspace is able to
        draw, without having to guess from memory."""
        dlg = tk.Toplevel(self.app)
        dlg.title("Panel kinds")
        dlg.geometry("640x420")
        dlg.transient(self.app)
        from matformer.gui.widgets import show_payload
        rows = [{"kind": kind,
                 "reads": "runs · metrics" if kind != "table" else "runs, or any command (Table)",
                 "notes": {"line": "series over the chosen axis", "scatter": "x against y",
                           "bar": "one bar per run/metric, delta vs baseline",
                           "histogram": "distribution of the values",
                           "table": "rows: final values, or an analysis result"}.get(kind, "")}
                for kind in P.KINDS]
        show_payload(self.app, dlg, {"__table__": True, "columns": ["kind", "reads", "notes"],
                                     "rows": rows})
        ttk.Button(dlg, text="Close", command=dlg.destroy).pack(anchor="e", padx=8, pady=6)

    def _view_menu(self):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Save view…", command=self._save_view)
        menu.add_command(label="Load view…", command=self._load_view)
        menu.add_command(label="Load view from file…", command=self._load_view_file)
        menu.add_command(label="Reset view", command=self._reset_view)
        menu.add_separator()
        menu.add_command(label="Export every panel (PNG)…", command=self.export_png)
        menu.add_command(label="Export data (CSV)…", command=self.export_csv)
        menu.add_command(label="Export data (JSON)…", command=self.export_json)
        menu.add_command(label="Export data (HTML)…", command=self.export_html)
        popup(menu, self.winfo_pointerx(), self.winfo_pointery())

    def view_dir(self):
        from pathlib import Path
        folder = Path.home() / ".matformer" / "views"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _state(self):
        return {"panels": self.panels, "smooth": float(self.smooth.get()),
                "logy": self.logy.get(), "xlink": self.xlink.get(), "cols": int(self.cols.get()),
                "xaxis": self.xaxis.get(), "columns": self._wide_columns,
                "project": self.project.get(), "focus": self.focus, "baseline": self.baseline}

    def _save_view(self):
        from tkinter.simpledialog import askstring
        name = askstring("Save view", "Name:", parent=self)
        if not name:
            return
        path = self.view_dir() / f"{name}.plotview.json"
        path.write_text(json.dumps(self._state(), indent=1), "utf-8")
        self.app.toast(f"view saved: {path}")

    def _load_view(self):
        options = sorted(p.stem for p in self.view_dir().glob("*.plotview.json"))
        if not options:
            return self.app.toast("no saved views yet")
        from tkinter.simpledialog import askstring
        choice = askstring("Load view", f"saved views: {', '.join(options)}", parent=self)
        if choice in options:
            self.apply_state(json.loads((self.view_dir() / f"{choice}.plotview.json").read_text("utf-8")))

    def _load_view_file(self):
        path = filedialog.askopenfilename(filetypes=[("Plot view", "*.json")])
        if path:
            try:
                self.apply_state(json.loads(open(path, encoding="utf-8").read()))
            except (OSError, json.JSONDecodeError) as problem:
                self.app.toast(f"{problem}", error=True)

    def apply_state(self, state):
        self.panels = state.get("panels") or []
        self.smooth.set(state.get("smooth", 0.6))
        self.logy.set(bool(state.get("logy")))
        self.xlink.set(bool(state.get("xlink", True)))
        self.cols.set(state.get("cols", 2))
        self.xaxis.set(state.get("xaxis", "step"))
        self._wide_columns = state.get("columns")
        self.focus, self.baseline = state.get("focus"), state.get("baseline")
        if state.get("project") in self.project.cget("values"):
            self.project.set(state["project"])
        self._show_runs()
        self._paint_marks()
        self._show_wide()
        self.redraw()

    def _reset_view(self):
        self.panels, self.focus, self.baseline, self._wide_columns = [], None, None, None
        self._auto_done = False
        self._show_panels()
        self._show_wide()
        self.redraw()

    # ── drawing ───────────────────────────────────────────────────────────────
    def repaint(self, palette):
        self.grid.configure(bg=palette["bg"])
        self._layout_key = None
        self.redraw()

    def redraw(self, force=False):
        if not hasattr(self, "grid"):
            return
        visible = [spec for spec in self.panels if spec.get("visible", True)]
        columns = max(1, int(self.cols.get() or 1))
        layout = (columns, tuple((id(spec), spec.get("span", 1), spec.get("kind", "line"),
                                  spec.get("title") or spec.get("metrics", ""))
                                 for spec in visible))
        if (not force and layout == getattr(self, "_layout_key", None) and self._canvases
                and all(spec.get("kind", "line") != "table" for spec in visible)):
            for spec, canvas, _panel in self._canvases:
                canvas.spec.update(spec)
                canvas.spec["focus"] = self.focus
                canvas.spec["baseline"] = self.baseline
                canvas.redraw()
            return
        self._layout_key = layout
        for widget in self.grid.winfo_children():
            widget.destroy()
        self._canvases = []
        if not visible:
            ttk.Label(self.grid, text="no panels - Add panel, or double-click a metric",
                      style="Muted.TLabel").grid(row=0, column=0, pady=30, columnspan=4)
            return
        context = self.context()
        palette = self.app.palette()
        cell = 0
        for spec in visible:
            panel = P.panel_for(dict(spec, focus=self.focus, baseline=self.baseline), context, self)
            frame = tk.Frame(self.grid, bg=palette["surface"], highlightbackground=palette["border"],
                             highlightthickness=1)
            span = min(columns, max(1, int(spec.get("span", 1) or 1)))
            if cell % columns + span > columns:
                cell += columns - (cell % columns)
            frame.grid(row=cell // columns, column=0, columnspan=span, sticky="nsew",
                       padx=3, pady=3)
            cell += span
            head = ttk.Frame(frame)
            head.pack(fill="x")
            title = ttk.Label(head, text=spec.get("title") or spec.get("metrics", ""),
                              style="Head.TLabel")
            title.pack(side="left", padx=6, pady=2)
            for glyph, action, doc in (("⤢", lambda s=spec: self._resize(s), "width"),
                                       ("✎", lambda s=spec: PanelDialog(self, s), "edit"),
                                       ("⧉", lambda s=spec: self.panels.append(dict(s)) or self.redraw(),
                                        "duplicate"),
                                       ("✕", lambda s=spec: self._remove(s), "remove")):
                button = ttk.Button(head, text=glyph, width=2, style="Ghost.TButton",
                                    command=action)
                button.pack(side="right")
                tip(button, doc)
            title.bind("<Button-3>", lambda e, s=spec, p=panel: self._panel_menu(e, s, p))
            body = tk.Frame(frame, bg=palette["surface"])
            body.pack(fill="both", expand=True)
            if getattr(panel, "needs_axes", True):
                canvas = TkPlotCanvas(self.app, body, dict(spec, focus=self.focus,
                                                           baseline=self.baseline), context, self)
                canvas.pack(fill="both", expand=True)
                self._canvases.append((spec, canvas, panel))
            else:
                widget = panel.widget(body, self.app)
                if isinstance(widget, tuple):
                    widget[0].pack(fill="both", expand=True)
                elif widget is not None:
                    widget.pack(fill="both", expand=True)
        for index in range(columns):
            self.grid.grid_columnconfigure(index, weight=1)

    def _panel_menu(self, e, spec, panel):
        menu = tk.Menu(self, tearoff=0)
        for kind in P.KINDS:
            menu.add_command(label=f"as {kind}",
                             command=lambda k=kind, s=spec: (s.update(kind=k), self.redraw()))
        menu.add_separator()
        menu.add_command(label="Edit…", command=lambda: PanelDialog(self, spec))
        menu.add_command(label="Width 1 / 2 / 3", command=lambda s=spec: self._resize(s))
        menu.add_command(label="Log Y", command=lambda s=spec: (s.update(logy=not s.get("logy")),
                                                                self.redraw()))
        menu.add_command(label="Reset zoom", command=lambda c=self._canvas_of(spec): c and c.reset())
        menu.add_separator()
        menu.add_command(label="Export PNG…", command=lambda s=spec: self._export_panel(s))
        menu.add_command(label="Export CSV…", command=lambda s=spec: self.export_csv(s))
        menu.add_command(label="Remove", command=lambda s=spec: self._remove(s))
        popup(menu, e.x_root, e.y_root)

    def _canvas_of(self, spec):
        return next((canvas for s, canvas, _panel in self._canvases if s is spec), None)

    def _resize(self, spec):
        spec["span"] = 1 + (int(spec.get("span", 1) or 1)) % 3
        self.redraw()

    def _remove(self, spec):
        if spec in self.panels:
            self.panels.remove(spec)
            self._show_panels()
            self.redraw()

    def _export_panel(self, spec):
        if not HAS_MPL:
            return self.app.toast("PNG export needs matplotlib", error=True)
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        palette = self.app.palette()
        panel = P.panel_for(dict(spec, focus=self.focus, baseline=self.baseline),
                            self.context(), self)
        if not getattr(panel, "needs_axes", True):
            return self.app.toast("this panel is a table", error=True)
        figure = Figure(figsize=(6.4, 3.8), dpi=120, facecolor=palette["surface"])
        axes = figure.add_subplot(111)
        panel.decorate(axes, palette)
        panel.draw(axes, palette)
        figure.tight_layout(pad=1.0)
        figure.savefig(path, dpi=170, facecolor=figure.get_facecolor())
        self.app.toast(f"saved {path}")

    def export_png(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path or not self._canvases:
            return
        from matplotlib.figure import Figure
        columns = max(1, int(self.cols.get() or 1))
        rows = -(-len(self._canvases) // columns)
        figure = Figure(figsize=(4.4 * columns, 2.6 * rows), dpi=96,
                        facecolor=self.app.palette()["surface"])
        context = self.context()
        for index, (spec, _canvas, panel) in enumerate(self._canvases, start=1):
            axes = figure.add_subplot(rows, columns, index)
            panel.decorate(axes, self.app.palette())
            panel.draw(axes, self.app.palette())
        figure.tight_layout(pad=1.2)
        figure.savefig(path, dpi=170, facecolor=figure.get_facecolor())
        self.app.toast(f"saved {path}")

    def export_json(self, spec=None):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("[")
            first = True
            for (run, key), points in sorted(self.series.items()):
                if spec and not P.matches(key, spec.get("metrics", "*")):
                    continue
                for x, y in points:
                    if not first:
                        handle.write(",")
                    first = False
                    json.dump({"run": run, "metric": key, "step": x, "value": y}, handle,
                              ensure_ascii=False, default=str)
            handle.write("]")
        self.app.toast(f"saved {path}")

    def export_html(self, spec=None):
        path = filedialog.asksaveasfilename(defaultextension=".html",
                                            filetypes=[("HTML", "*.html")])
        if not path:
            return
        import html as html_mod
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("<!doctype html><meta charset='utf-8'><title>Matformer metrics</title>"
                         "<style>body{font-family:system-ui;margin:2rem}"
                         "td,th{border:1px solid #ddd;padding:.25rem .5rem;font-size:.85rem}</style>"
                         "<table><thead><tr><th>run</th><th>metric</th><th>step</th><th>value</th>"
                         "</tr></thead><tbody>")
            for (run, key), points in sorted(self.series.items()):
                if spec and not P.matches(key, spec.get("metrics", "*")):
                    continue
                for x, y in points:
                    handle.write("<tr><td>" + html_mod.escape(str(run)) + "</td><td>"
                                 + html_mod.escape(str(key)) + "</td><td>" + str(x)
                                 + "</td><td>" + html_mod.escape(str(y)) + "</td></tr>")
            handle.write("</tbody></table>")
        self.app.toast(f"saved {path}")

    def export_csv(self, spec=None):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run", "metric", "step", "value"])
            for (run, key), points in sorted(self.series.items()):
                if spec and not P.matches(key, spec.get("metrics", "*")):
                    continue
                writer.writerows([run, key, x, y] for x, y in points)
        self.app.toast(f"saved {path}")

    # ── the wide comparison table ─────────────────────────────────────────────
    def _all_columns(self):
        metric_keys = [f"last {key}" for key in sorted({key for (_run, key) in self.series})]
        delta_keys = [f"Δ last {key}" for key in sorted({key for (_run, key) in self.series})]
        config_keys = sorted({field for cfg in self.configs.values() for field in cfg})
        return ["run", "project", "steps", "server", "started"] + metric_keys \
            + (delta_keys if self.baseline else []) + config_keys

    def _show_wide(self):
        available = self._all_columns()
        if self._wide_columns is None:
            chosen = available[:8]
        else:
            chosen = [c for c in self._wide_columns if c in available] or ["run"]
        self.wide.headings(chosen)
        keys = sorted({key for (_run, key) in self.series})
        base = {}
        if self.baseline:
            for key in keys:
                points = self.series.get((self.baseline, key))
                if points:
                    base[key] = last(self.series, (self.baseline, key))
        rows = []
        for rid, meta in self.meta.items():
            cfg = self.configs.get(rid) or {}
            values = {"run": ("★ " if rid == self.focus else "") + meta["run"],
                      "project": meta.get("project", ""), "steps": meta.get("steps") or 0,
                      "server": meta.get("server", ""), "started": meta.get("started", "")}
            for key in keys:
                value = last(self.series, (rid, key))
                if value is not None:
                    values[f"last {key}"] = value
                    if key in base:
                        values[f"Δ last {key}"] = round(value - base[key], 6)
            for field, value in cfg.items():
                values[field] = value
            rows.append(values)
        self.wide.fill(rows, "no runs")

    def _choose_columns(self):
        ColumnsDialog(self.app, self, self._all_columns(), self._wide_columns or [])

    def _wide_menu(self, e):
        row = self.wide.selected()
        if not row:
            return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Focus", command=lambda: (setattr(self, "focus", row.get("id") or row["run"]),
                                                         self.redraw()))
        menu.add_command(label="Copy run id", command=lambda: self.app.copy(row["run"]))
        popup(menu, e.x_root, e.y_root)

    def compare(self, run_a, run_b):
        """Two runs, field by field, differences highlighted."""
        self.server.call("runs.config", {"run": run_a}, lambda a: self._compare_with(run_a, run_b, a))

    def _compare_with(self, run_a, run_b, cfg_a):
        def got(cfg_b):
            DiffDialog(self.app, run_a, run_b,
                       cfg_a if isinstance(cfg_a, dict) else {},
                       cfg_b if isinstance(cfg_b, dict) else {})
        self.server.call("runs.config", {"run": run_b}, got)

    def _run_menu(self, e):
        row = self.table.selected()
        if not row:
            return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Tick / untick", command=lambda: self._toggle(row))
        menu.add_command(label="Focus", command=lambda: (setattr(self, "focus", row["id"]),
                                                         self._paint_marks(), self.redraw()))
        menu.add_command(label="Set as baseline", command=lambda: (setattr(self, "baseline", row["id"]),
                                                                   self._paint_marks(), self._show_wide(),
                                                                   self.redraw()))
        menu.add_command(label="Show config", command=lambda: self.server.call(
            "runs.config", {"run": row["id"]}, self._show_cfg))
        menu.add_command(label="Task results…", command=lambda: self._task_results(row["id"]))
        menu.add_command(label="Analyze metrics…", command=lambda: self._analyze_metrics(row["id"]))
        menu.add_command(label="Compare with…", command=lambda: self._pick_compare(row))
        menu.add_separator()
        menu.add_command(label="New panel from its metrics…",
                         command=lambda: self.add_panel(dict(P.defaults(), title=row["run"][:24])))
        menu.add_command(label="Copy id", command=lambda: self.app.copy(row["id"]))
        menu.add_separator()
        menu.add_command(label="Delete run…", command=lambda: self._delete_run(row["id"]))
        popup(menu, e.x_root, e.y_root)

    def _analyze_metrics(self, run_id):
        """The run's metric histories as a table/histogram/scatter: losses, norms, lr, task summaries
        all become rows, without a panel per key."""
        dlg = tk.Toplevel(self.app)
        dlg.title(f"Metrics · {run_id}")
        dlg.geometry("1000x620")
        dlg.transient(self.app)
        view = AnalysisView(self.app, dlg)
        view.pack(fill="both", expand=True, padx=8, pady=8)

        def got(metrics):
            if not metrics or isinstance(metrics, tuple):
                view.show_summary({}, title=run_id)
                return
            view.show_metrics(metrics, title=run_id)
        self.server.call("runs.metrics", {"run": run_id}, got)
        ttk.Button(dlg, text="Close", command=dlg.destroy).pack(anchor="e", padx=8, pady=(0, 8))
        dlg.bind("<Escape>", lambda e: dlg.destroy())

    def _task_results(self, run_id):
        dlg = tk.Toplevel(self.app)
        dlg.title(f"Task results · {run_id}")
        dlg.geometry("1000x620")
        dlg.transient(self.app)
        topbar = ttk.Frame(dlg)
        topbar.pack(fill="x", padx=8)
        latest = tk.BooleanVar(value=True)
        ttk.Checkbutton(topbar, text="latest execution per task", variable=latest,
                        command=lambda: rebuild()).pack(side="left")
        pan = ttk.PanedWindow(dlg, orient="vertical")
        pan.pack(fill="both", expand=True, padx=8, pady=8)
        table = TableList(self.app, pan, height=7)
        table.headings(["task", "phase", "step", "epoch", "records"])
        table.fill([], "reading…")
        pan.add(table, weight=1)
        holder = ttk.Frame(pan)
        view = AnalysisView(self.app, holder)
        view.pack(fill="both", expand=True)
        pan.add(holder, weight=4)

        def picked(row):
            if row:
                view.show_summary(row.get("summary") or {}, title=f"{row.get('task')} "
                                                                f"step {row.get('step')}")

        table.tree.bind("<<TreeviewSelect>>", lambda e: picked(table.selected()))

        state = {"rows": []}

        def display_rows():
            rows = state["rows"]
            if latest.get():
                seen = {}
                for row in rows:
                    seen[row.get("task")] = row          # later executions win
                rows = list(seen.values())
            return [{"task": r.get("task"), "phase": r.get("phase"), "step": r.get("step"),
                     "epoch": r.get("epoch"), "records": r.get("records"),
                     "summary": r.get("summary") or {}} for r in rows]

        def rebuild():
            table.fill(display_rows(), "no task executions")
            if table.rows:
                last = str(len(table.rows) - 1)
                table.tree.selection_set(last)
                table.tree.event_generate("<<TreeviewSelect>>")

        def got(rows):
            if not rows or isinstance(rows, tuple):
                state["rows"] = []
                rebuild()
                return
            state["rows"] = list(rows)
            rebuild()
        self.server.call("runs.tasks", {"run": run_id}, got)
        ttk.Button(dlg, text="Close", command=dlg.destroy).pack(anchor="e", padx=8, pady=(0, 8))
        dlg.bind("<Escape>", lambda e: dlg.destroy())

    def _remove_selected(self):
        row = self.table.selected()
        if not row:
            return self.app.toast("select a run first", error=True)
        self._delete_run(row["id"])

    def _delete_run(self, run_id):
        if not messagebox.askyesno("Delete run", f"Remove {run_id} from the run database?\n"
                                    "Files on disk are untouched.", parent=self):
            return
        self.server.call("runs.delete", {"run": run_id},
                         lambda r: (self.refresh(), self.app.toast(f"deleted {run_id}")))

    def _pick_compare(self, row):
        others = [r for r in self.table.rows if r["id"] != row["id"]]
        if not others:
            return self.app.toast("needs at least two runs")
        from tkinter.simpledialog import askstring
        names = [r["run"] for r in others]
        choice = askstring("Compare configs", f"{row['run']}  vs  ({', '.join(names[:8])})",
                           parent=self)
        if choice in names:
            other = next(r for r in others if r["run"] == choice)
            self.compare(row["id"], other["id"])


class PanelDialog(tk.Toplevel):
    """Every option a panel takes, generated from panels.SPEC_FIELDS: adding an option to the
    abstraction makes it editable here without touching this file."""

    def __init__(self, page, spec=None):
        super().__init__(page)
        self.page, self.editing = page, spec
        self.title("Edit panel" if spec else "Add panel")
        self.geometry("520x520")
        self.transient(page)
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=12, pady=10)
        ttk.Button(bottom, text="Save", style="Accent.TButton", command=self._ok).pack(side="right", padx=4)
        ttk.Button(bottom, text="Cancel", command=self.destroy).pack(side="right")
        scroll = ScrollBox(self.app, self, width=10)
        scroll.pack(fill="both", expand=True, padx=12, pady=12)
        fields = [dict(field, default=(spec or {}).get(field["name"], field.get("default")))
                  for field in P.SPEC_FIELDS]
        if not spec:
            fields[0]["default"] = _guess_title(page)
        self.form = Form(self.app, scroll.inner, fields)
        self.form.pack(fill="x")
        self.bind("<Escape>", lambda e: self.destroy())
        fit_dialog(self, 520, 0)

    @property
    def app(self):
        return self.page.app

    def _ok(self):
        spec = P.defaults()
        spec.update({key: value for key, value in self.form.values().items() if value not in (None, "")})
        spec["visible"] = (self.editing or {}).get("visible", True)
        spec["title"] = spec.get("title") or spec.get("metrics", "panel")
        self.page.new_panel(spec)
        self.destroy()


def _guess_title(page):
    keys = sorted({key for (_run, key) in page.series})
    if not keys:
        return "new panel"
    common = os.path.commonprefix(keys).strip("/") if len(keys) > 1 else keys[0]
    return common or "new panel"


class ColumnsDialog(tk.Toplevel):
    """Which columns the wide table shows: tick to show, and run is always there."""

    def __init__(self, app, page, available, current):
        super().__init__(page)
        self.page = page
        self.title("Columns")
        self.geometry("380x560")
        self.transient(page)
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=10, pady=8)
        ttk.Button(bottom, text="OK", style="Accent.TButton", command=self._ok).pack(side="right", padx=4)
        ttk.Button(bottom, text="Cancel", command=self.destroy).pack(side="right")
        quick = ttk.Frame(bottom)
        quick.pack(side="left")
        ttk.Button(quick, text="All", width=5, style="Ghost.TButton",
                   command=lambda: self._all(True)).pack(side="left")
        ttk.Button(quick, text="None", width=5, style="Ghost.TButton",
                   command=lambda: self._all(False)).pack(side="left", padx=3)
        self.vars = {}
        scroll = ScrollBox(app, self, width=10)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        for name in available:
            var = tk.BooleanVar(value=name in (current or []) or available.index(name) < 5)
            ttk.Checkbutton(scroll.inner, text=name, variable=var).pack(anchor="w")
            self.vars[name] = var
        self.bind("<Escape>", lambda e: self.destroy())

    def _all(self, on):
        for name, var in self.vars.items():
            if name != "run":
                var.set(on)

    def _ok(self):
        self.page._wide_columns = (["run"] + [name for name, var in self.vars.items()
                                              if var.get() and name != "run"])
        self.page._show_wide()
        self.destroy()


class DiffDialog(tk.Toplevel):
    """Two runs' configs side by side, changed fields highlighted."""

    def __init__(self, app, run_a, run_b, cfg_a, cfg_b):
        super().__init__(app)
        self.title(f"config diff — {run_a}  vs  {run_b}")
        self.geometry("860x560")
        self.transient(app)
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=8, pady=8)
        ttk.Button(bottom, text="Close", command=self.destroy).pack(side="right")
        keys = sorted(set(cfg_a) | set(cfg_b))
        changed = [key for key in keys if str(cfg_a.get(key)) != str(cfg_b.get(key))]
        ttk.Label(bottom, text=f"{len(changed)} of {len(keys)} fields differ",
                  style="Muted.TLabel").pack(side="left")
        only = tk.BooleanVar(value=True)
        ttk.Checkbutton(bottom, text="changed only", variable=only,
                        command=lambda: self._fill(keys, cfg_a, cfg_b, only.get())).pack(side="left", padx=8)
        self.table = TableList(app, self, height=18)
        self.table.headings(["field", run_a[:30], run_b[:30]])
        self.table.pack(fill="both", expand=True, padx=8, pady=8)
        self._fill(keys, cfg_a, cfg_b, True)
        self.bind("<Escape>", lambda e: self.destroy())

    def _fill(self, keys, cfg_a, cfg_b, only_changed):
        self.table.headings(["field", "value a", "value b"])
        self.table.fill([{"field": key, "value a": str(cfg_a.get(key, "")),
                          "value b": str(cfg_b.get(key, ""))}
                         for key in keys
                         if not only_changed or str(cfg_a.get(key)) != str(cfg_b.get(key))],
                        "no differences")

