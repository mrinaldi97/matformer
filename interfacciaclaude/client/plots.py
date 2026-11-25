"""Plots page.

Server events:  job {id, project, config, ...}   metric {run, project, key, points: [[step, value], ...]}

TOP       — Project scope, applied to both the sidebar list and the wide runs table.

LEFT      — name filter, checkbox list of runs, Select/Unselect all, Focus/Clear focus,
            Smoothing, Log-Y / Link-X, Plots panel (searchable, tick to show/hide, drag to reorder,
            Show/Hide all), + Add plot…, Export all…, Save view…, Load view….

RIGHT     — Plots tab: one plot per visible spec; drawn lines from ticked runs (or the single focused run).
                     Right-click a plot for Reset zoom / Export image / Export CSV (our menu,
                     not pyqtgraph's default one, which loses its item names in aggregate layouts).
            Runs tab:  wide, sortable, filterable table of every run in scope, with a customizable
                       column set (tick + drag). Select 2+ rows and right-click → Compare configs.
                       If a run is focused, this tab flips to a vertical field|value view + Notes box.

Scaling: only the first AUTO_TICK runs auto-tick; the plot engine only draws ticked runs.
"""
import fnmatch
import json
import statistics
from collections import defaultdict
from pathlib import Path

import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                            QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
                            QMessageBox, QPlainTextEdit, QPushButton, QSlider, QSplitter, QTableWidget,
                            QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from forms import Form

pg.setConfigOptions(background="w", foreground="k", antialias=True)

AGG = {"mean": statistics.fmean, "sum": sum, "min": min, "max": max, "std": statistics.pstdev}
LINES = {"per run": lambda r, k: r, "per metric": lambda r, k: k,
         "every run × metric": lambda r, k: f"{r} · {k}", "all in one": lambda r, k: "all"}
NEW_PLOT = {"fields": [
    {"name": "title", "type": "text", "hint": "(automatic)"},
    {"name": "metrics", "type": "text", "default": "norm/*", "hint": "glob: loss, norm/*, *grad*", "tip": "Which logged series to use"},
    {"name": "runs", "type": "text", "default": "*", "hint": "glob on server/run", "tip": "On top of the project / filter / ticks"},
    {"name": "lines", "label": "One line per", "type": "choice", "options": list(LINES), "tip": "How series are grouped into lines"},
    {"name": "agg", "label": "Combine with", "type": "choice", "options": list(AGG), "tip": "How series inside one line are merged, step by step"},
]}
DEFAULTS = [{"metrics": m, "runs": "*", "lines": "per run", "agg": "mean"} for m in ("loss", "lr", "grad_norm")] + [
    {"title": "norm/* — one line per component (mean over runs)", "metrics": "norm/*", "runs": "*", "lines": "per metric", "agg": "mean"},
    {"title": "norm/* — mean over components, one line per run", "metrics": "norm/*", "runs": "*", "lines": "per run", "agg": "mean"},
]


def ema(ys, a):
    out, s = [], None
    for y in ys:
        s = y if s is None else a * s + (1 - a) * y
        out.append(s)
    return out


def combine(series, agg):
    by = defaultdict(list)
    for xs, ys in series:
        for x, y in zip(xs, ys):
            by[x].append(y)
    xs = sorted(by)
    return xs, [AGG[agg](by[x]) for x in xs]


def _safe(name):
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)[:80]


class _Item(QTableWidgetItem):
    def __lt__(self, other):
        a, b = self.data(Qt.ItemDataRole.UserRole), other.data(Qt.ItemDataRole.UserRole)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a < b
        return super().__lt__(other)


class AddPlotDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Add derived plot")
        self.form = Form(NEW_PLOT, lambda *a: None)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.form); lay.addWidget(bb)
        self.resize(440, 260)


class ColumnsDialog(QDialog):
    """Pick which wide-table columns to show and in what order (drag to reorder)."""
    def __init__(self, parent, available, current):
        super().__init__(parent)
        self.setWindowTitle("Columns to show (tick + drag to reorder)")
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        chosen, seen = list(current or []), set()
        for c in chosen:
            if c in available and c not in seen:
                self._add(c, True); seen.add(c)
        for c in available:
            if c not in seen:
                self._add(c, False); seen.add(c)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        b_all = QPushButton("Select all");   b_all.clicked.connect(lambda: self._set_all(Qt.CheckState.Checked))
        b_no  = QPushButton("Unselect all"); b_no.clicked.connect(lambda: self._set_all(Qt.CheckState.Unchecked))
        row = QHBoxLayout(); row.addWidget(b_all); row.addWidget(b_no); row.addStretch()
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Tick to show; drag to reorder. 'run' is always shown."))
        lay.addWidget(self.list, 1); lay.addLayout(row); lay.addWidget(bb)
        self.resize(360, 500)

    def _add(self, name, checked):
        it = QListWidgetItem(name)
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.list.addItem(it)

    def _set_all(self, state):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.text() == "run":
                continue                                    # always on
            it.setCheckState(state)
        self.list.blockSignals(False)

    def selection(self):
        out = [self.list.item(i).text() for i in range(self.list.count())
               if self.list.item(i).text() == "run" or self.list.item(i).checkState() == Qt.CheckState.Checked]
        if "run" not in out:
            out.insert(0, "run")
        return out


class DiffDialog(QDialog):
    """Side-by-side config diff of two runs; changed keys highlighted."""
    def __init__(self, parent, a, b, cfg_a, cfg_b):
        super().__init__(parent)
        self.setWindowTitle(f"Config diff: {a}  vs  {b}")
        keys = sorted(set(cfg_a) | set(cfg_b))
        t = QTableWidget(len(keys), 3)
        t.setHorizontalHeaderLabels(["field", a, b])
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.horizontalHeader().setStretchLastSection(True)
        hl = QColor(255, 240, 200)
        for i, k in enumerate(keys):
            va, vb = cfg_a.get(k, ""), cfg_b.get(k, "")
            cells = [QTableWidgetItem(k), QTableWidgetItem(str(va)), QTableWidgetItem(str(vb))]
            if va != vb:
                for c in cells:
                    c.setBackground(hl)
            for j, c in enumerate(cells):
                t.setItem(i, j, c)
        t.setColumnWidth(0, 180)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject); bb.accepted.connect(self.accept)
        lay = QVBoxLayout(self)
        lay.addWidget(t); lay.addWidget(bb)
        self.resize(700, 520)


class PlotsPage(QWidget):
    COLS = 2
    ALL = "all projects"
    AUTO_TICK = 10

    def __init__(self, spec, win):
        super().__init__()
        self.data, self.meta = {}, {}
        self.ticked, self.order, self.notes = {}, [], {}
        self.focus = None
        self.specs, self.plots = [], []
        self.column_order = None
        self._table_struct = None

        # ── TOP: project scope ──────────────────────────────────────────────
        self.project = QComboBox(); self.project.addItem(self.ALL)
        self.project.currentTextChanged.connect(self.schedule)
        top = QHBoxLayout()
        top.addWidget(QLabel("Project scope:"))
        top.addWidget(self.project, 1)

        # ── LEFT sidebar ────────────────────────────────────────────────────
        self.search = QLineEdit(); self.search.setPlaceholderText("filter runs by name…")
        self.search.textChanged.connect(self.schedule)
        self.list = QListWidget()
        self.list.itemChanged.connect(self._on_list_toggled)
        self.list.itemDoubleClicked.connect(lambda it: self._focus_run(it.data(Qt.ItemDataRole.UserRole)))
        self.list.itemSelectionChanged.connect(self._on_list_selection)

        self.btn_all_runs = QPushButton("Select all");   self.btn_all_runs.clicked.connect(lambda: self._set_all_runs(True))
        self.btn_no_runs = QPushButton("Unselect all");  self.btn_no_runs.clicked.connect(lambda: self._set_all_runs(False))

        self.focus_label = QLabel("Focus: (none)")
        self.btn_focus = QPushButton("Focus selected run"); self.btn_focus.clicked.connect(self._focus_selected)
        self.btn_clear = QPushButton("Clear focus");         self.btn_clear.clicked.connect(self._clear_focus)
        self.btn_focus.setEnabled(False)

        # Smoothing + Log-Y / Link-X moved together here
        self.smooth = QSlider(Qt.Orientation.Horizontal); self.smooth.setRange(0, 99)
        self.smooth.valueChanged.connect(self.schedule)
        self.logy = QCheckBox("Log-Y");   self.logy.stateChanged.connect(self._relayout_plots)
        self.xlink = QCheckBox("Link X"); self.xlink.stateChanged.connect(self._relayout_plots)

        # Plots panel
        self.plot_search = QLineEdit(); self.plot_search.setPlaceholderText("filter plots…")
        self.plot_search.textChanged.connect(self._filter_plot_list)
        self.plot_list = QListWidget()
        self.plot_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.plot_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.plot_list.itemChanged.connect(self._on_plot_toggled)
        self.plot_list.model().rowsMoved.connect(self._on_plots_reordered)
        self.btn_all_plots = QPushButton("Show all"); self.btn_all_plots.clicked.connect(lambda: self._set_all_plots(True))
        self.btn_no_plots = QPushButton("Hide all");  self.btn_no_plots.clicked.connect(lambda: self._set_all_plots(False))

        self.btn_add_plot = QPushButton("+ Add plot…");   self.btn_add_plot.clicked.connect(self._open_add_plot)
        self.btn_export_all = QPushButton("Export all…"); self.btn_export_all.clicked.connect(self._open_export_menu)
        self.btn_save_view = QPushButton("Save view…");   self.btn_save_view.clicked.connect(self._save_view)
        self.btn_load_view = QPushButton("Load view…");   self.btn_load_view.clicked.connect(self._load_view)

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.addWidget(self.search)
        sl.addWidget(self.list, 1)
        row = QHBoxLayout(); row.addWidget(self.btn_all_runs); row.addWidget(self.btn_no_runs)
        sl.addLayout(row)
        row = QHBoxLayout(); row.addWidget(self.btn_focus); row.addWidget(self.btn_clear)
        sl.addLayout(row)
        sl.addWidget(self.focus_label)
        # ---- Smoothing sits right next to Log-Y / Link-X now ----
        row = QHBoxLayout(); row.addWidget(QLabel("Smooth")); row.addWidget(self.smooth, 1)
        sl.addLayout(row)
        row = QHBoxLayout(); row.addWidget(self.logy); row.addWidget(self.xlink); row.addStretch()
        sl.addLayout(row)
        sl.addWidget(QLabel("Plots")); sl.addWidget(self.plot_search)
        sl.addWidget(self.plot_list, 1)
        row = QHBoxLayout(); row.addWidget(self.btn_all_plots); row.addWidget(self.btn_no_plots)
        sl.addLayout(row)
        row = QHBoxLayout(); row.addWidget(self.btn_add_plot); row.addWidget(self.btn_export_all)
        sl.addLayout(row)
        row = QHBoxLayout(); row.addWidget(self.btn_save_view); row.addWidget(self.btn_load_view)
        sl.addLayout(row)

        # ── RIGHT: Plots tab ────────────────────────────────────────────────
        self.grid = pg.GraphicsLayoutWidget()

        # ── RIGHT: Runs tab ─────────────────────────────────────────────────
        self.runs_filter = QLineEdit(); self.runs_filter.setPlaceholderText("filter rows by any column…")
        self.runs_filter.textChanged.connect(self._apply_wide_visibility)
        self.btn_columns = QPushButton("Columns…"); self.btn_columns.clicked.connect(self._open_columns)
        self.runs_toolbar = QWidget()
        tl = QHBoxLayout(self.runs_toolbar); tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.runs_filter, 1); tl.addWidget(self.btn_columns)

        self.wide = QTableWidget(0, 2)
        self.wide.setHorizontalHeaderLabels(["✓", "run"])
        self.wide.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.wide.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.wide.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.wide.setSortingEnabled(True)
        self.wide.setAlternatingRowColors(True)
        self.wide.verticalHeader().hide()
        self.wide.horizontalHeader().setStretchLastSection(True)
        self.wide.itemChanged.connect(self._on_wide_toggled)
        self.wide.itemDoubleClicked.connect(self._focus_from_wide)
        self.wide.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.wide.customContextMenuRequested.connect(self._wide_context_menu)

        self.detail = QTableWidget(0, 2)
        self.detail.setHorizontalHeaderLabels(["field", "value"])
        self.detail.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.detail.setAlternatingRowColors(True)
        self.detail.verticalHeader().hide()
        self.detail.horizontalHeader().setStretchLastSection(True)
        self.detail.setColumnWidth(0, 220)
        self.detail.hide()

        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("notes for the focused run…")
        self.notes_edit.setFixedHeight(70)
        self.notes_edit.textChanged.connect(self._save_note)
        self.notes_edit.hide()

        self.runs_tab = QWidget()
        rl = QVBoxLayout(self.runs_tab); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.runs_toolbar)
        rl.addWidget(self.wide, 1)
        rl.addWidget(self.detail, 1)
        rl.addWidget(self.notes_edit)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.grid, "Plots")
        self.tabs.addTab(self.runs_tab, "Runs")
        self.tabs.currentChanged.connect(self.schedule)

        sp = QSplitter(); sp.addWidget(side); sp.addWidget(self.tabs); sp.setSizes([280, 1120])
        outer = QVBoxLayout(self); outer.setContentsMargins(6, 6, 6, 6)
        outer.addLayout(top); outer.addWidget(sp, 1)

        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(150)
        self.timer.timeout.connect(self.refresh)

        for s in DEFAULTS:
            self._add_plot_spec(s)

    # ── lifecycle / intake ──────────────────────────────────────────────────
    def activated(self):
        self.refresh()

    def schedule(self, *_):
        self.timer.start()

    def on_event(self, server, name, d):
        if name == "job" and d.get("project"):
            self.track(f"{server}/{d['id']}", d["project"], d.get("config", {}))
        elif name == "metric":
            run = f"{server}/{d['run']}"
            self.track(run, d.get("project", "?"))
            xs, ys = self.data.setdefault((run, d["key"]), ([], []))
            for x, y in d["points"]:
                xs.append(x); ys.append(y)
            self.schedule()

    def track(self, run, project, config=None):
        if run not in self.meta:
            self.meta[run] = {"project": project, "config": {}}
            self.ticked[run] = len(self.order) < self.AUTO_TICK
            self.order.append(run)
            it = QListWidgetItem(run)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if self.ticked[run] else Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, run)
            self.list.blockSignals(True); self.list.addItem(it); self.list.blockSignals(False)
            if self.project.findText(project) < 0:
                self.project.blockSignals(True)
                self.project.addItem(project)
                self.project.blockSignals(False)
        if config:
            self.meta[run]["config"] = config

    # ── plot specs / visibility / order ─────────────────────────────────────
    def _add_plot_spec(self, spec):
        title = spec.get("title") or f"{spec['metrics']} · {spec['lines']} · {spec['agg']}"
        spec = {**spec, "title": title, "visible": spec.get("visible", True)}
        self.specs.append(spec)
        it = QListWidgetItem(title)
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(Qt.CheckState.Checked if spec["visible"] else Qt.CheckState.Unchecked)
        it.setData(Qt.ItemDataRole.UserRole, spec)
        self.plot_list.blockSignals(True); self.plot_list.addItem(it); self.plot_list.blockSignals(False)
        self._relayout_plots()

    def _open_add_plot(self):
        d = AddPlotDialog(self)
        if d.exec():
            self._add_plot_spec(d.form.values())

    def _on_plot_toggled(self, item):
        spec = item.data(Qt.ItemDataRole.UserRole)
        if spec is None:
            return
        spec["visible"] = item.checkState() == Qt.CheckState.Checked
        self._relayout_plots()

    def _on_plots_reordered(self, *args):
        new = [self.plot_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.plot_list.count())]
        self.specs = [s for s in new if s is not None]
        self._relayout_plots()

    def _set_all_plots(self, on):
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        self.plot_list.blockSignals(True)
        for i in range(self.plot_list.count()):
            it = self.plot_list.item(i)
            if it.isHidden():
                continue
            spec = it.data(Qt.ItemDataRole.UserRole)
            if spec is None:
                continue
            it.setCheckState(state)
            spec["visible"] = on
        self.plot_list.blockSignals(False)
        self._relayout_plots()

    def _filter_plot_list(self, text):
        t = text.lower()
        for i in range(self.plot_list.count()):
            self.plot_list.item(i).setHidden(t not in self.plot_list.item(i).text().lower())

    # ── THE fix: use grid.ci.removeItem (GraphicsLayoutWidget does not proxy removeItem) ──
    def _relayout_plots(self, *_):
        # Detach every plot (do not destroy — a fresh PlotItem will be created below)
        for spec in self.specs:
            p = spec.pop("_plot", None)
            if p is None:
                continue
            try:
                p.setVisible(False)
            except Exception:
                pass
            try:
                self.grid.ci.removeItem(p)
            except Exception:
                pass
        self.plots = []
        n = 0
        for spec in self.specs:
            if not spec.get("visible", True):
                continue
            p = self.grid.addPlot(row=n // self.COLS, col=n % self.COLS, title=spec["title"])
            p.addLegend(offset=(5, 5)); p.showGrid(x=True, y=True)
            # Names shown in pyqtgraph's default context menu come from item.name(); shadow it so
            # aggregate layouts don't show "Plot"/"Viewbox".
            try:
                p.name = lambda t=spec["title"]: t
                p.vb.name = lambda t=spec["title"]: t
            except Exception:
                pass
            if self.logy.isChecked():
                p.setLogMode(y=True)
            self._install_plot_menu(spec, p)                # our own right-click menu
            spec["_plot"] = p
            self.plots.append(p)
            n += 1
        if self.xlink.isChecked() and len(self.plots) > 1:
            for p in self.plots[1:]:
                p.setXLink(self.plots[0])
        self.schedule()

    # ── our own plot context menu (Reset zoom / Export image / Export CSV) ──
    def _install_plot_menu(self, spec, plot):
        def show(ev):
            m = QMenu()
            m.addAction("Reset zoom", lambda: plot.getViewBox().autoRange())
            m.addSeparator()
            m.addAction("Export image (PNG)…", lambda: self._export_png(spec, plot))
            m.addAction("Export this plot's data (CSV)…", lambda: self._export_spec_csv(spec))
            m.exec(ev.screenPos().toPoint())
        try:
            plot.vb.raiseContextMenu = show
        except Exception:
            pass
        try:
            plot.setMenuEnabled(False)                      # kill the default ViewBox/PlotItem menu
        except Exception:
            pass

    # ── runs: sidebar list + wide table sync ────────────────────────────────
    def _on_list_selection(self):
        self.btn_focus.setEnabled(bool(self.list.selectedItems()))

    def _focus_selected(self):
        items = self.list.selectedItems()
        if items:
            self._focus_run(items[0].data(Qt.ItemDataRole.UserRole))

    def _focus_run(self, run):
        if not run:
            return
        self.focus = run
        self.focus_label.setText(f"Focus: <b>{run}</b>")
        self.schedule()

    def _clear_focus(self):
        self.focus = None
        self.focus_label.setText("Focus: (none)")
        self.schedule()

    def _focus_from_wide(self, it):
        run = self.wide.item(it.row(), self._wide_run_col()).text()
        self._focus_run(run)

    def _on_list_toggled(self, item):
        run = item.data(Qt.ItemDataRole.UserRole)
        if not run:
            return
        self.ticked[run] = item.checkState() == Qt.CheckState.Checked
        self._sync_wide_check(run)
        self.schedule()

    def _on_wide_toggled(self, item):
        if item.column() != 0:
            return
        run = item.data(Qt.ItemDataRole.UserRole)
        if not run:
            return
        self.ticked[run] = item.checkState() == Qt.CheckState.Checked
        self._sync_list_check(run)
        self.schedule()

    def _sync_wide_check(self, run):
        self.wide.blockSignals(True)
        for r in range(self.wide.rowCount()):
            c = self.wide.item(r, 0)
            if c and c.data(Qt.ItemDataRole.UserRole) == run:
                c.setCheckState(Qt.CheckState.Checked if self.ticked[run] else Qt.CheckState.Unchecked)
                break
        self.wide.blockSignals(False)

    def _sync_list_check(self, run):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == run:
                it.setCheckState(Qt.CheckState.Checked if self.ticked[run] else Qt.CheckState.Unchecked)
                break
        self.list.blockSignals(False)

    def _set_all_runs(self, on):
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        vis = self._visible_runs()
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            run = it.data(Qt.ItemDataRole.UserRole)
            if run in vis:
                it.setCheckState(state); self.ticked[run] = on
        self.list.blockSignals(False)
        self.wide.blockSignals(True)
        for r in range(self.wide.rowCount()):
            c = self.wide.item(r, 0)
            if c:
                run = c.data(Qt.ItemDataRole.UserRole)
                if run in vis:
                    c.setCheckState(state)
        self.wide.blockSignals(False)
        self.schedule()

    def _toggle(self, run):
        self.ticked[run] = not self.ticked.get(run, False)
        self._sync_wide_check(run); self._sync_list_check(run)
        self.schedule()

    # ── wide-table context menu (single: focus/tick/copy; multi: compare configs) ──
    def _wide_context_menu(self, pos):
        r = self.wide.rowAt(pos.y())
        if r < 0:
            return
        sel_rows = sorted({i.row() for i in self.wide.selectionModel().selectedRows()})
        run_col = self._wide_run_col()
        run = self.wide.item(r, run_col).text()
        m = QMenu(self)
        if len(sel_rows) >= 2:
            a = self.wide.item(sel_rows[0], run_col).text()
            b = self.wide.item(sel_rows[1], run_col).text()
            m.addAction(f"Compare configs: {a}  vs  {b}", lambda: self._compare_configs(a, b))
            m.addSeparator()
        m.addAction("Focus this run", lambda: self._focus_run(run))
        m.addAction("Tick / untick", lambda: self._toggle(run))
        m.addAction("Copy run name", lambda: QApplication.clipboard().setText(run))
        m.exec(self.wide.viewport().mapToGlobal(pos))

    def _compare_configs(self, a, b):
        DiffDialog(self, a, b, self.meta[a]["config"], self.meta[b]["config"]).exec()

    # ── filtering / plotting ────────────────────────────────────────────────
    def _visible_runs(self):
        proj, text = self.project.currentText(), self.search.text().lower()
        out = set()
        for r in self.meta:
            if proj != self.ALL and self.meta[r]["project"] != proj:
                continue
            if text and text not in r.lower():
                continue
            out.add(r)
        return out

    def _plot_runs(self):
        if self.focus:
            return {self.focus} if self.focus in self.meta else set()
        return {r for r in self._visible_runs() if self.ticked.get(r)}

    def lines(self, spec, runs):
        label, groups = LINES[spec["lines"]], defaultdict(list)
        for (run, key), s in self.data.items():
            if run not in runs:
                continue
            if not fnmatch.fnmatch(key, spec["metrics"] or "*"):
                continue
            if not self.focus and not fnmatch.fnmatch(run, spec["runs"] or "*"):
                continue
            groups[label(run, key)].append(s)
        return {n: combine(ss, spec["agg"]) for n, ss in sorted(groups.items())}

    # ── wide table ──────────────────────────────────────────────────────────
    def _config_keys(self):
        return sorted({k for m in self.meta.values() for k in m["config"]})

    def _metric_keys(self):
        return sorted({k for (_, k) in self.data})

    def _all_columns(self):
        return ["run", "project"] + self._config_keys() + [f"last {k}" for k in self._metric_keys()]

    def _effective_columns(self):
        cols = self.column_order if self.column_order else self._all_columns()
        cols = [c for c in cols if c in self._all_columns()]
        if "run" not in cols:
            cols = ["run"] + cols
        return cols

    def _wide_run_col(self):
        for c in range(self.wide.columnCount()):
            h = self.wide.horizontalHeaderItem(c)
            if h and h.text() == "run":
                return c
        return 1

    def _table_structure(self):
        return (tuple(self.order), tuple(self._config_keys()), tuple(self._metric_keys()),
                tuple(self.column_order or ()))

    def _maybe_rebuild_wide(self):
        struct = self._table_structure()
        if struct == self._table_struct:
            return
        self._table_struct = struct
        self._rebuild_wide()

    def _rebuild_wide(self):
        self.wide.setSortingEnabled(False)
        self.wide.blockSignals(True)
        self.wide.clearContents()
        cols = self._effective_columns()
        headers = ["✓"] + cols
        self.wide.setColumnCount(len(headers))
        self.wide.setHorizontalHeaderLabels(headers)
        self.wide.setRowCount(len(self.order))
        for i, run in enumerate(self.order):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk.setCheckState(Qt.CheckState.Checked if self.ticked.get(run) else Qt.CheckState.Unchecked)
            chk.setData(Qt.ItemDataRole.UserRole, run)
            self.wide.setItem(i, 0, chk)
            for j, col in enumerate(cols):
                it = _Item()
                if col == "run":
                    it.setText(run); it.setData(Qt.ItemDataRole.UserRole, run)
                elif col == "project":
                    it.setText(self.meta[run]["project"])
                elif col.startswith("last "):
                    ys = self.data.get((run, col[5:]), ([], []))[1]
                    if ys:
                        it.setText(f"{ys[-1]:.6g}"); it.setData(Qt.ItemDataRole.UserRole, float(ys[-1]))
                else:
                    v = self.meta[run]["config"].get(col, "")
                    it.setText(str(v)); it.setData(Qt.ItemDataRole.UserRole, v)
                self.wide.setItem(i, 1 + j, it)
        self.wide.blockSignals(False)
        self.wide.setSortingEnabled(True)
        self.wide.setColumnWidth(0, 24)
        self._apply_wide_visibility()

    def _update_wide_values(self):
        headers = [self.wide.horizontalHeaderItem(c).text() if self.wide.horizontalHeaderItem(c) else ""
                   for c in range(self.wide.columnCount())]
        run_col = headers.index("run") if "run" in headers else 1
        last_cols = [(c, h[5:]) for c, h in enumerate(headers) if h.startswith("last ")]
        for r in range(self.wide.rowCount()):
            run_item = self.wide.item(r, run_col)
            if not run_item:
                continue
            run = run_item.text()
            for c, k in last_cols:
                cell = self.wide.item(r, c)
                if cell is None:
                    continue
                ys = self.data.get((run, k), ([], []))[1]
                if ys:
                    cell.setText(f"{ys[-1]:.6g}")
                    cell.setData(Qt.ItemDataRole.UserRole, float(ys[-1]))

    def _apply_wide_visibility(self, *_):
        vis = self._visible_runs()
        t = self.runs_filter.text().strip().lower()
        for r in range(self.wide.rowCount()):
            c = self.wide.item(r, 0)
            run = c.data(Qt.ItemDataRole.UserRole) if c else None
            if run not in vis:
                self.wide.setRowHidden(r, True); continue
            if not t:
                self.wide.setRowHidden(r, False); continue
            hit = any((self.wide.item(r, c2) and t in self.wide.item(r, c2).text().lower())
                      for c2 in range(self.wide.columnCount()))
            self.wide.setRowHidden(r, not hit)

    def _open_columns(self):
        d = ColumnsDialog(self, self._all_columns(), self.column_order)
        if d.exec():
            self.column_order = d.selection()
            self._table_struct = None
            self.schedule()

    # ── detail + notes ──────────────────────────────────────────────────────
    def _save_note(self):
        if self.focus:
            self.notes[self.focus] = self.notes_edit.toPlainText()

    def _fill_detail(self):
        r = self.focus
        rows = [("run", r), ("project", self.meta[r]["project"])]
        for k, v in sorted(self.meta[r]["config"].items()):
            rows.append((k, v))
        for (rr, k), (xs, ys) in sorted(self.data.items()):
            if rr == r:
                rows.append((f"last {k}", f"{ys[-1]:.6g}" if ys else ""))
        self.detail.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.detail.setItem(i, 0, QTableWidgetItem(str(k)))
            self.detail.setItem(i, 1, QTableWidgetItem(str(v)))
        self.notes_edit.blockSignals(True)
        self.notes_edit.setPlainText(self.notes.get(r, ""))
        self.notes_edit.blockSignals(False)

    # ── exports ─────────────────────────────────────────────────────────────
    def _open_export_menu(self):
        m = QMenu(self)
        m.addAction("All series → long CSV…", self._export_all_series)
        m.addAction("All run configs → CSV…", self._export_all_configs)
        m.addAction("All visible plots → PNG folder…", self._export_all_pngs)
        m.exec(self.btn_export_all.mapToGlobal(self.btn_export_all.rect().bottomLeft()))

    def _export_all_series(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export all series", "all_series.csv", "CSV (*.csv)")
        if not path:
            return
        rows = ["run,project,metric,step,value"]
        for (run, key), (xs, ys) in self.data.items():
            proj = self.meta[run]["project"]
            for x, y in zip(xs, ys):
                rows.append(f'"{run}","{proj}","{key}",{x},{y}')
        Path(path).write_text("\n".join(rows), "utf-8")

    def _export_all_configs(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export all configs", "all_configs.csv", "CSV (*.csv)")
        if not path:
            return
        keys = self._config_keys()
        rows = ["run,project," + ",".join(f'"{k}"' for k in keys)]
        for run, m in self.meta.items():
            rows.append(",".join([f'"{run}"', f'"{m["project"]}"']
                                 + [f'"{m["config"].get(k, "")}"' for k in keys]))
        Path(path).write_text("\n".join(rows), "utf-8")

    def _export_all_pngs(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder for PNG exports")
        if not folder:
            return
        from pyqtgraph.exporters import ImageExporter
        n = 0
        for spec in self.specs:
            p = spec.get("_plot")
            if p is None:
                continue
            try:
                ex = ImageExporter(p)
                ex.parameters()["width"] = 1600
                ex.export(str(Path(folder) / f"{_safe(spec['title'])}.png"))
                n += 1
            except Exception:
                pass
        QMessageBox.information(self, "Export complete", f"Wrote {n} PNG file(s) to {folder}.")

    def _export_png(self, spec, plot):
        path, _ = QFileDialog.getSaveFileName(self, "Export image", f"{_safe(spec['title'])}.png", "PNG (*.png)")
        if not path:
            return
        try:
            from pyqtgraph.exporters import ImageExporter
            ex = ImageExporter(plot)
            ex.parameters()["width"] = 1600
            ex.export(path)
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))

    def _export_spec_csv(self, spec):
        path, _ = QFileDialog.getSaveFileName(self, "Export plot data", f"{_safe(spec['title'])}.csv", "CSV (*.csv)")
        if not path:
            return
        rows = ["series,step,value"]
        for name, (xs, ys) in self.lines(spec, self._plot_runs()).items():
            for x, y in zip(xs, ys):
                rows.append(f'"{name}",{x},{y}')
        Path(path).write_text("\n".join(rows), "utf-8")

    # ── bookmarkable view ───────────────────────────────────────────────────
    def _save_view(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save view", "view.json", "JSON (*.json)")
        if not path:
            return
        data = {
            "project": self.project.currentText(),
            "search": self.search.text(),
            "smooth": self.smooth.value(),
            "ticks": dict(self.ticked),
            "focus": self.focus,
            "specs": [{k: v for k, v in s.items() if k != "_plot"} for s in self.specs],
            "logy": self.logy.isChecked(),
            "xlink": self.xlink.isChecked(),
            "column_order": self.column_order,
            "notes": dict(self.notes),
        }
        Path(path).write_text(json.dumps(data, indent=1), "utf-8")

    def _load_view(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load view", "", "JSON (*.json)")
        if not path:
            return
        try:
            d = json.loads(Path(path).read_text("utf-8"))
        except Exception:
            return
        idx = self.project.findText(d.get("project", self.ALL))
        if idx >= 0:
            self.project.blockSignals(True); self.project.setCurrentIndex(idx); self.project.blockSignals(False)
        self.search.setText(d.get("search", ""))
        self.smooth.setValue(int(d.get("smooth", 0)))
        for r, v in (d.get("ticks") or {}).items():
            self.ticked[r] = bool(v)
            self._sync_list_check(r); self._sync_wide_check(r)
        self.notes = dict(d.get("notes") or {})
        self.logy.blockSignals(True);   self.logy.setChecked(bool(d.get("logy")));   self.logy.blockSignals(False)
        self.xlink.blockSignals(True);  self.xlink.setChecked(bool(d.get("xlink"))); self.xlink.blockSignals(False)
        self.column_order = d.get("column_order")
        self.specs = []
        self.plot_list.blockSignals(True); self.plot_list.clear(); self.plot_list.blockSignals(False)
        for s in (d.get("specs") or []):
            self._add_plot_spec(s)
        if d.get("focus"):
            self._focus_run(d["focus"])
        else:
            self._clear_focus()
        self._table_struct = None
        self.schedule()

    # ── refresh ─────────────────────────────────────────────────────────────
    def refresh(self):
        vis = self._visible_runs()
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(it.data(Qt.ItemDataRole.UserRole) not in vis)
        if self.tabs.currentIndex() == 0:
            a = self.smooth.value() / 100
            runs = self._plot_runs()
            visible_specs = [s for s in self.specs if s.get("visible", True)]
            for spec, p in zip(visible_specs, self.plots):
                if p.legend is not None:
                    p.legend.clear()                    # avoid stale entries after a redraw
                p.clear()
                for i, (name, (xs, ys)) in enumerate(self.lines(spec, runs).items()):
                    p.plot(xs, ema(ys, a), pen=pg.mkPen(pg.intColor(i, 9), width=2), name=name)
            return
        if self.focus and self.focus in self.meta:
            self.runs_toolbar.hide(); self.wide.hide()
            self.detail.show(); self.notes_edit.show()
            self._fill_detail()
        else:
            self.detail.hide(); self.notes_edit.hide()
            self.runs_toolbar.show(); self.wide.show()
            self._maybe_rebuild_wide()
            self._update_wide_values()
            self._apply_wide_visibility()
