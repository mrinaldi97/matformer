"""Remote file explorer (icon view), a modal picker built on it, the config editor page, and a
stub TrainingPage.

Explorer talks to any server through these methods:
  fs.list(path)    -> [{name, kind: dir|file, size}]
  fs.rename(path, to)
  fs.delete(path)
ConfigPage adds:
  config.read(path)   -> {fields, values}
  config.write(path, values)
TrainingPage adds:
  train.start(config, values, plugins) -> {run}
  train.abort(id)

The explorer is used in three ways:
  • as the "Files" page, where `win` is set: double-click routes YAMLs to the config editor,
    .pt/.safetensors to the models page, dataset folders/shards to the datasets page, and right-click
    offers Rename / Delete / Edit / Train.
  • inside the ConfigPage (no `win`): just emits `chosen` for YAML files.
  • as the modal `pick()` (no `win`): just emits `chosen` for whatever file/URI the user selects.
"""
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                            QListView, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit, QPushButton,
                            QScrollArea, QSplitter, QStyle, QToolButton, QVBoxLayout, QWidget)

from forms import Form

SP = QStyle.StandardPixmap
ICONS = {".yaml": SP.SP_FileDialogDetailedView, ".safetensors": SP.SP_DriveHDIcon, ".pt": SP.SP_DriveHDIcon,
         ".jsonl": SP.SP_FileDialogContentsView}


class Explorer(QWidget):
    chosen = Signal(str)      # double-clicked file, or an URI typed in the path bar (e.g. hf://qwen/Qwen3-8B)

    def __init__(self, servers, root="/", exts=(), server=None, win=None):
        super().__init__()
        self.servers, self.root, self.exts, self.path = servers, root, tuple(exts), root
        self.win = win                                         # if set, this is the Files page
        self.srv, self.up, self.edit = QComboBox(), QToolButton(), QLineEdit(root)
        self.srv.addItems(servers)
        self.srv.setCurrentText(server or self.srv.itemText(0))
        self.up.setText("⬆ Up")
        self.up.setShortcut("Backspace")
        self.edit.setPlaceholderText("/path   or   hf://org/model   (Enter to go)")
        self.view, self.info = QListWidget(), QLabel()
        self.view.setViewMode(QListView.ViewMode.IconMode)
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
        self.view.setMovement(QListView.Movement.Static)
        self.view.setIconSize(QSize(40, 40))
        self.view.setGridSize(QSize(120, 86))
        self.view.setWordWrap(True)
        top = QHBoxLayout()
        for w in (QLabel("Server:"), self.srv, self.up):
            top.addWidget(w)
        top.addWidget(self.edit, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.info)
        self.srv.currentTextChanged.connect(lambda _: self.go(self.root))
        self.up.clicked.connect(lambda: self.go(self.path.rsplit("/", 1)[0] or "/"))
        self.edit.returnPressed.connect(self.enter)
        self.view.itemDoubleClicked.connect(self.open)
        if win is not None:                                    # context menu only in the Files page
            self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.view.customContextMenuRequested.connect(self._menu)

    @property
    def server(self):
        return self.servers[self.srv.currentText()]

    @property
    def current(self):
        """What 'OK' means: selected item, else a typed URI, else the folder being viewed."""
        sel = self.view.selectedItems()
        return self.join(sel[0].text()) if sel else self.edit.text() if "://" in self.edit.text() else self.path

    def join(self, name):
        return self.path.rstrip("/") + "/" + name

    def activated(self):
        self.go(self.path)

    def go(self, path):
        self.path = path or "/"
        self.edit.setText(self.path)
        if not self.server.up:
            return self.info.setText(f"{self.server.name} is offline")
        self.server.call("fs.list", {"path": self.path}, lambda rows, p=self.path: p == self.path and self.fill(rows))

    def fill(self, rows):
        self.view.clear()
        for r in rows:
            ext = "." + r["name"].rsplit(".", 1)[-1]
            if r["kind"] == "file" and self.exts and ext not in self.exts:
                continue
            icon = self.style().standardIcon(SP.SP_DirIcon if r["kind"] == "dir" else ICONS.get(ext, SP.SP_FileIcon))
            item = QListWidgetItem(icon, r["name"])
            item.setData(Qt.ItemDataRole.UserRole, r["kind"])
            item.setToolTip(r["name"] if r["kind"] == "dir" else f"{r['name']}\n{r['size']:,} bytes")
            self.view.addItem(item)
        self.info.setText(f"{self.view.count()} items")

    # ── double-click dispatch ────────────────────────────────────────────────
    def open(self, item):
        kind = item.data(Qt.ItemDataRole.UserRole)
        path = self.join(item.text())
        if kind == "dir":
            if self.win is not None and self._looks_like_dataset(path):
                return self.win.open_dataset(path, self.srv.currentText())
            return self.go(path)
        ext = "." + item.text().rsplit(".", 1)[-1].lower() if "." in item.text() else ""
        if self.win is not None:
            if ext in (".yaml", ".yml"):
                return self.win.open_config(path, self.srv.currentText())
            if ext in (".pt", ".safetensors"):
                return self.win.open_model(path, self.srv.currentText())
            if ext == ".jsonl":
                return self.win.open_dataset(path, self.srv.currentText())
        self.chosen.emit(path)

    @staticmethod
    def _looks_like_dataset(path):
        """A folder directly under /datasets/ is treated as a dataset."""
        parent = path.rstrip("/").rsplit("/", 1)[0]
        return parent in ("/datasets", "/datasets/")

    def enter(self):
        t = self.edit.text().strip()
        self.chosen.emit(t) if "://" in t else self.go(t)

    # ── context menu (files page only) ───────────────────────────────────────
    def _menu(self, pos):
        item = self.view.itemAt(pos)
        if item is None:
            return
        path = self.join(item.text())
        kind = item.data(Qt.ItemDataRole.UserRole)
        m = QMenu(self)
        m.addAction("Rename…", lambda: self._rename(path))
        m.addAction("Delete…", lambda: self._delete(path))
        m.addSeparator()
        m.addAction("Copy path", lambda: self.win.statusBar().showMessage(path, 4000))
        if kind == "file" and "." in item.text():
            ext = "." + item.text().rsplit(".", 1)[-1].lower()
            if ext in (".yaml", ".yml"):
                m.addSeparator()
                m.addAction("Edit in config editor", lambda: self.win.open_config(path, self.srv.currentText()))
                m.addAction("Train with this config", lambda: self.win.start_training(path, None, self.srv.currentText()))
        m.exec(self.view.mapToGlobal(pos))

    def _rename(self, path):
        name = path.rsplit("/", 1)[-1]
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", text=name)
        new_name = (new_name or "").strip()
        if not ok or not new_name or new_name == name:
            return
        parent = path.rsplit("/", 1)[0]
        new_path = parent.rstrip("/") + "/" + new_name
        self.server.call("fs.rename", {"path": path, "to": new_path},
                         lambda r: self._after_change(r, f"renamed to {new_name}"))

    def _delete(self, path):
        name = path.rsplit("/", 1)[-1]
        if QMessageBox.question(self, "Delete", f"Delete <b>{name}</b>?") != QMessageBox.StandardButton.Yes:
            return
        self.server.call("fs.delete", {"path": path}, lambda r: self._after_change(r, f"deleted {name}"))

    def _after_change(self, r, verb):
        msg = (r or {}).get("msg", "done")
        if self.win is not None:
            self.win.statusBar().showMessage(f"{verb}: {msg}", 4000)
        self.go(self.path)


def pick(parent, servers, server, root="/", exts=()):
    """Modal picker -> path/URI string, or None. Double-click a file, select + OK, or type an hf:// id."""
    d, ex = QDialog(parent), Explorer(servers, root, exts, server)
    d.setWindowTitle("Browse files on the server")
    d.resize(760, 480)
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    bb.accepted.connect(d.accept)
    bb.rejected.connect(d.reject)
    ex.chosen.connect(lambda _: d.accept())
    lay = QVBoxLayout(d)
    lay.addWidget(ex)
    lay.addWidget(bb)
    ex.go(root)
    return ex.current if d.exec() else None


class ConfigPage(QWidget):
    """Explorer on the left; double-click a config and the server's schema becomes an editable form.
    A 'Train this config' button hands the current path off to the TrainingPage."""
    def __init__(self, spec, win):
        super().__init__()
        self.win, self.form, self.path = win, None, None
        self.ex = Explorer(win.servers, spec.get("root", "/"), spec.get("exts", [".yaml"]))
        self.ex.chosen.connect(self.open)
        self.title = QLabel("Double-click a config file on the left.")
        self.save = QPushButton("Save to server")
        self.train = QPushButton("Train this config")
        self.area = QScrollArea(widgetResizable=True)
        for b in (self.save, self.train):
            b.setEnabled(False)
        self.save.clicked.connect(self.write)
        self.train.clicked.connect(self._train)
        buttons = QHBoxLayout()
        buttons.addWidget(self.save)
        buttons.addWidget(self.train)
        buttons.addStretch()
        right = QWidget()
        lay = QVBoxLayout(right)
        for w in (self.title, self.area):
            lay.addWidget(w)
        lay.addLayout(buttons)
        sp = QSplitter()
        sp.addWidget(self.ex)
        sp.addWidget(right)
        sp.setSizes([420, 700])
        QVBoxLayout(self).addWidget(sp)

    def activated(self):
        self.ex.activated()

    def open(self, path, server_name=None):
        if server_name:
            idx = self.ex.srv.findText(server_name)
            if idx >= 0:
                self.ex.srv.blockSignals(True)
                self.ex.srv.setCurrentIndex(idx)
                self.ex.srv.blockSignals(False)
        self.path, srv = path, self.ex.server
        srv.call("config.read", {"path": path}, self.edit)

    def edit(self, r):                                 # r = {fields: <same schema as plugin TOML>, values: {...}}
        browse = lambda f: pick(self.win, self.win.servers, self.ex.srv.currentText(), f.get("root", "/"), f.get("exts", ()))
        self.form = Form({"fields": r["fields"]}, self.win.run, browse)
        self.form.load(r["values"])
        self.area.setWidget(self.form)
        self.title.setText(f"<b>{self.path}</b>  on {self.ex.srv.currentText()}")
        self.save.setEnabled(True)
        self.train.setEnabled(True)

    def write(self):
        self.ex.server.call("config.write", {"path": self.path, "values": self.form.values()},
                            lambda r: self.win.statusBar().showMessage(r["msg"], 5000))

    def _train(self):
        if not self.path:
            return
        values = self.form.values() if self.form else {}
        self.win.start_training(self.path, values, self.ex.srv.currentText())


class TrainingPage(QWidget):
    """Stub training page: live log of training jobs + Abort + 'Open in Plots'.

    `start()` is called by the file manager and config editor; it hits the server's `train.start`
    method (which must return {'run': <id>}) and logs incoming `job` events of kind 'training'.
    """
    def __init__(self, spec, win):
        super().__init__()
        self.win, self.run, self.srv_name = win, None, None
        self.title = QLabel("No run started from this page yet.")
        self.abort = QPushButton("Abort run"); self.abort.setEnabled(False)
        self.abort.clicked.connect(self._abort)
        self.plots_btn = QPushButton("Open in Plots"); self.plots_btn.setEnabled(False)
        self.plots_btn.clicked.connect(self._open_plots)
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setPlaceholderText("Training terminal output will appear here.")
        top = QHBoxLayout()
        top.addWidget(self.title, 1)
        top.addWidget(self.abort)
        top.addWidget(self.plots_btn)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.log, 1)
        self.log.appendPlainText("Training page ready.\n"
                                 "  • Double-click a YAML in the Files page.\n"
                                 "  • Or press 'Train this config' in the Config editor.")

    def start(self, config_path, values=None, server_name=None):
        srv_name = server_name or self.win.chat.target.currentText()
        srv = self.win.servers.get(srv_name)
        if srv is None:
            return
        self.srv_name, self.run = srv_name, None
        self.abort.setEnabled(True); self.plots_btn.setEnabled(False)
        self.title.setText(f"Starting <b>{config_path}</b> on {srv_name}…")
        self.log.appendPlainText(f"\n→ starting training with {config_path} on {srv_name}")
        srv.call("train.start",
                 {"config": config_path, "values": values or {}, "plugins": self.win.values()},
                 self._started)

    def _started(self, r):
        if isinstance(r, dict) and r.get("run"):
            self.run = r["run"]
            self.title.setText(f"Run <b>{r['run']}</b> on {self.srv_name}")
            self.plots_btn.setEnabled(True)

    def _abort(self):
        if self.run and self.srv_name:
            self.win.servers[self.srv_name].call("train.abort", {"id": self.run})
            self.log.appendPlainText(f"→ requested abort of {self.run}")

    def _open_plots(self):
        if not self.run or not self.srv_name:
            return
        self.win.select_page("plots")
        i = self.win.page_index("plots")
        if i is None:
            return
        page = self.win.pages[i]
        if hasattr(page, "_focus_run"):
            page._focus_run(f"{self.srv_name}/{self.run}")

    def on_event(self, server, name, data):
        if name != "job" or data.get("kind") != "training":
            return
        step = data.get("step")
        prog = data.get("progress", 0) or 0
        line = f"[{server}] {data.get('id','')}  {data.get('status','?')}  progress={prog:.1%}"
        if step is not None:
            line += f"  step={step}"
        self.log.appendPlainText(line)
"""Plugin schema (a dict loaded from TOML) -> a working Qt form. The only place that draws fields.

A new field type = one small function returning (widget, getter, setter) + one line in KINDS.
"""
from qtpy.QtCore import QLocale, QRegularExpression, Qt
from qtpy.QtGui import QDoubleValidator, QRegularExpressionValidator
from qtpy.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                            QLineEdit, QPushButton, QRadioButton, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget)


def _row(*ws):
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    for w in ws:
        lay.addWidget(w)
    return box


def _num(f):                                                  # "int" and "float"
    isf = f["type"] == "float"
    w = QDoubleSpinBox() if isf else QSpinBox()
    w.setRange(f.get("min", -1e9 if isf else -10**9), f.get("max", 1e9 if isf else 10**9))
    if isf:
        w.setSingleStep(f.get("step", 0.1))
    w.setValue(f.get("default", 0))
    if "auto" in f:                                           # shown instead of the minimum, e.g. seed -1 = "random"
        w.setSpecialValueText(f["auto"])
    if not f.get("slider"):
        return w, w.value, w.setValue
    k = round(1 / w.singleStep()) if isf else 1               # slider works in integers
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(round(w.minimum() * k), round(w.maximum() * k))
    s.setValue(round(w.value() * k))
    s.valueChanged.connect(lambda v: w.setValue(v / k))
    w.valueChanged.connect(lambda v: s.setValue(round(v * k)))
    return _row(s, w), w.value, w.setValue


def _bool(f):
    w = QCheckBox()
    w.setChecked(f.get("default", False))
    return w, w.isChecked, w.setChecked


def _choice(f):
    w = QComboBox()
    w.addItems(f.get("options", []))
    return w, w.currentText, w.setCurrentText


def _text(f):
    w = QLineEdit(f.get("default", ""))
    w.setPlaceholderText(f.get("hint", ""))
    return w, w.text, w.setText


def _radio(f):                                                # few options: all visible at once
    w = QWidget()
    lay, g = QHBoxLayout(w), QButtonGroup(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for o in f["options"]:
        b = QRadioButton(o)
        g.addButton(b)
        lay.addWidget(b)
    g.buttons()[0].setChecked(True)
    return w, lambda: g.checkedButton().text(), lambda v: [b.setChecked(b.text() == v) for b in g.buttons()]


def _bigint(f):                                               # beyond 32 bit; shown as 131,072
    w = QLineEdit(f"{f.get('default', 0):,}")
    w.setValidator(QRegularExpressionValidator(QRegularExpression(r"-?[\d,_]*")))
    return w, lambda: int(w.text().replace(",", "").replace("_", "") or 0), lambda v: w.setText(f"{int(v):,}")


def _sci(f):                                                  # 3e-4 style floats (learning rates, eps, ...)
    w, v = QLineEdit(f"{f.get('default', 0):g}"), QDoubleValidator()
    v.setLocale(QLocale.c())
    w.setValidator(v)
    return w, lambda: float(w.text() or 0), lambda x: w.setText(f"{x:g}")


class PathEdit(QWidget):                                      # text box + "…" that opens the remote explorer
    def __init__(self, f):
        super().__init__()
        self.edit, self.browse, b = QLineEdit(f.get("default", "")), None, QToolButton()
        self.edit.setPlaceholderText(f.get("hint", "path or hf://org/name"))
        b.setText("…")
        b.setToolTip("Browse files on the server")
        b.clicked.connect(self.pick)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.edit, 1)
        lay.addWidget(b)

    def pick(self):
        p = self.browse and self.browse()
        if p:
            self.edit.setText(p)


def _path(f):
    w = PathEdit(f)
    return w, w.edit.text, w.edit.setText


KINDS = {"float": _num, "int": _num, "bool": _bool, "choice": _choice, "text": _text,
         "radio": _radio, "bigint": _bigint, "sci": _sci, "path": _path}


class Form(QWidget):
    def __init__(self, spec, run, browse=None):
        """spec: plugin dict.  run(method, params): command buttons.  browse(field_spec) -> path: for 'path' fields."""
        super().__init__()
        lay = QFormLayout(self)
        lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.fields = {}                                      # name -> (spec, widget, getter, setter, toggle)
        sec = None
        for f in spec.get("fields", []):
            if f.get("section") and f["section"] != sec:      # optional heading row
                sec = f["section"]
                lay.addRow(QLabel(f"<b>{sec}</b>"))
            w, get, put = KINDS[f["type"]](f)
            if isinstance(w, PathEdit):
                w.browse = lambda f=f: browse and browse(f)
            ui, tog = w, None
            if f.get("toggle"):                               # optional parameter: unticked = absent from values()
                tog = QCheckBox()
                tog.toggled.connect(ui.setEnabled)
                tog.setChecked(f.get("on", False))
                ui.setEnabled(tog.isChecked())
                ui = _row(tog, ui)
            if f.get("doc"):                                  # documentation shown under the control
                d, box = QLabel(f["doc"]), QWidget()
                d.setWordWrap(True)
                d.setStyleSheet("color: gray")
                col = QVBoxLayout(box)
                col.setContentsMargins(0, 0, 0, 0)
                col.addWidget(ui)
                col.addWidget(d)
                ui = box
            ui.setToolTip(f.get("tip", ""))
            lay.addRow(f.get("label", f["name"].replace("_", " ").capitalize()), ui)
            self.fields[f["name"]] = (f, w, get, put, tog)
        for c in spec.get("commands", []):
            b = QPushButton(c["label"])
            b.clicked.connect(lambda _=False, c=c: run(c["call"], self.values()))
            lay.addRow(b)

    def values(self):
        return {n: get() for n, (_, _, get, _, tog) in self.fields.items() if tog is None or tog.isChecked()}

    def load(self, d):
        for n, (_, _, _, put, tog) in self.fields.items():
            if n in d:
                put(d[n])
            if tog:
                tog.setChecked(n in d)

    def bind(self, server):
        """(Re)fill every field declared with options_from from this server."""
        for f, w, *_ in self.fields.values():
            if "options_from" in f and server.up:
                server.call(f["options_from"], None, lambda rows, w=w: self._fill(w, rows))

    @staticmethod
    def _fill(w, rows):
        keep = w.currentText()
        w.clear()
        w.addItems([r["name"] for r in rows])
        w.setCurrentText(keep)
"""LLM-Lab desktop shell.   python client/main.py [--demo]     (--demo also launches the dummy servers)

This file only wires things together; what exists (servers, pages, menus, plugins, themes) comes from config/.
"""
import atexit
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from qtpy.QtCore import QSettings, Qt, QTimer
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (QApplication, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox,
                            QPlainTextEdit, QScrollArea, QStackedWidget, QTabWidget, QTreeWidget, QTreeWidgetItem)

from explorer import Explorer, TrainingPage, pick
from forms import Form
from net import Server
from pages import PAGES, Jobs

ROOT = Path(__file__).resolve().parent.parent
LEFT, RIGHT, BOTTOM = (Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea,
                       Qt.DockWidgetArea.BottomDockWidgetArea)

# Route the generic "Files" page through our own Explorer factory so it can give the
# explorer a back-reference to the main window (which lets double-click dispatch to other pages).
PAGES["files"] = lambda spec, win: Explorer(win.servers, spec.get("root", "/"), win=win)
# And register our TrainingPage under the "training" kind.
PAGES["training"] = TrainingPage


class Main(QMainWindow):
    def __init__(self, cfg, plugins):
        super().__init__()
        self.setWindowTitle("LLM-Lab")
        self.cfg, self.prefs = cfg, QSettings("llm-lab", "gui")
        self.plugins, self.forms, self.docks, self.menus, self.act = plugins, {}, [], {}, {}
        self.servers = {s["name"]: Server(s["name"], s["url"]) for s in cfg["server"]}
        self.logw = QPlainTextEdit(readOnly=True)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # The training page is declared as a "stub" in app.toml; upgrade it to our real page.
        for p in cfg["pages"]:
            if p.get("id") == "training":
                p["kind"] = "training"

        # Pages + navigator tree
        tree, groups = QTreeWidget(), {}
        tree.setHeaderHidden(True)
        self.pages = [PAGES[p["kind"]](p, self) for p in cfg["pages"]]
        self.chat = next(w for w in self.pages if hasattr(w, "send"))
        for i, (p, w) in enumerate(zip(cfg["pages"], self.pages)):
            self.stack.addWidget(w)
            if p["group"] not in groups:
                groups[p["group"]] = QTreeWidgetItem(tree, [p["group"]])
            QTreeWidgetItem(groups[p["group"]], [p["title"]]).setData(0, Qt.ItemDataRole.UserRole, i)
        tree.expandAll()
        tree.currentItemChanged.connect(lambda item, _: item.data(0, Qt.ItemDataRole.UserRole) is not None
                                        and self.show_page(item.data(0, Qt.ItemDataRole.UserRole)))
        nav = self.dock("Navigator", tree, LEFT)
        self.nav_tree, self.groups = tree, groups

        # One docked form per plugin file, tabbed together; shown only on the page they declare
        first = None
        for pid, spec in plugins.items():
            self.forms[pid] = form = Form(spec, self.run, self.browse)
            area = QScrollArea(widgetResizable=True)
            area.setWidget(form)
            d = self.dock(spec["title"], area, RIGHT, spec["page"])
            first = first or d
            if d is not first:
                self.tabifyDockWidget(first, d)
        self.setTabPosition(RIGHT, QTabWidget.TabPosition.North)
        self.jobs = Jobs(self.servers)
        jobs = self.dock("Jobs — all servers", self.jobs, BOTTOM)

        # Menus + toolbar, generated from config. An entry [label, name, key] triggers self.do_<name>
        for m in cfg["menu"]:
            menu = self.menus[m["title"]] = self.menuBar().addMenu(m["title"])
            for it in m["items"]:
                if it == "-":
                    menu.addSeparator()
                    continue
                label, name, key = it
                a = self.act[name] = menu.addAction(label)
                a.setShortcut(key)
                a.triggered.connect(lambda _=False, f=getattr(self, "do_" + name): f())
        for d in self.docks:
            self.menus["&View"].addAction(d.toggleViewAction())
        tb = self.addToolBar("Main")
        tb.setObjectName("Main")
        for n in cfg["toolbar"]:
            tb.addAction(self.act[n])

        # Status bar: one LED per server
        self.leds = {n: QLabel() for n in self.servers}
        for n, led in self.leds.items():
            self.statusBar().addPermanentWidget(led)
            self.paint_led(n, False)

        self.set_theme(self.prefs.value("theme", cfg["default_theme"]))
        if self.prefs.value("geometry"):
            self.restoreGeometry(self.prefs.value("geometry"))
            self.restoreState(self.prefs.value("state"))
        else:
            self.resize(1450, 880)
            self.resizeDocks([nav, first], [190, 440], Qt.Orientation.Horizontal)
            self.resizeDocks([jobs], [170], Qt.Orientation.Vertical)
        tree.setCurrentItem(groups["Work"].child(0))
        first.raise_()
        for s in self.servers.values():
            s.pushed.connect(self.on_event)
            s.state.connect(self.on_state)
            s.start()

    # ── building blocks ──────────────────────────────────────────────────────
    def dock(self, title, widget, area, page=None):
        d = QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        d.page = page
        self.addDockWidget(area, d)
        self.docks.append(d)
        return d

    def set_theme(self, name):
        self.theme, colors = name, self.cfg["themes"][name]
        p = QPalette(QColor(colors["Button"]))                 # derives bevel shades, then override per role
        for role, c in colors.items():
            p.setColor(getattr(QPalette.ColorRole, role), QColor(c))
        QApplication.setPalette(p)
        self.prefs.setValue("theme", name)

    def show_page(self, i):
        self.stack.setCurrentIndex(i)
        for d in self.docks:
            if d.page:
                d.setVisible(d.page == self.cfg["pages"][i]["id"])
        getattr(self.pages[i], "activated", lambda: 0)()

    def log(self, msg):
        self.logw.appendPlainText(f"{time.strftime('%H:%M:%S')}  {msg}")

    def values(self):
        """Everything that goes to the backend with a request: all non-local plugin forms."""
        return {k: f.values() for k, f in self.forms.items() if self.plugins[k].get("scope") != "local"}

    def browse(self, f):
        """Remote picker for 'path' fields: starts at f['root'] on the server chosen in the chat page."""
        return pick(self, self.servers, self.chat.target.currentText(), f.get("root", "/"), f.get("exts", ()))

    def rebind(self):
        for f in self.forms.values():
            f.bind(self.chat.server)

    def run(self, method, params):
        """Plugin command -> the server selected in the chat page."""
        self.chat.server.call(method, params, lambda r: self.statusBar().showMessage(str((r or {}).get("msg", r)), 6000))

    # ── cross-page navigation (used by the file manager) ─────────────────────
    def page_index(self, page_id):
        for i, p in enumerate(self.cfg["pages"]):
            if p.get("id") == page_id:
                return i
        return None

    def select_page(self, page_id):
        """Switch to the page with the given id (by selecting it in the navigator tree)."""
        i = self.page_index(page_id)
        if i is None:
            return False
        for g in range(self.nav_tree.topLevelItemCount()):
            g_item = self.nav_tree.topLevelItem(g)
            for c in range(g_item.childCount()):
                child = g_item.child(c)
                if child.data(0, Qt.ItemDataRole.UserRole) == i:
                    self.nav_tree.setCurrentItem(child)
                    return True
        self.show_page(i)
        return True

    def open_config(self, path, server_name=None):
        """Double-clicking a YAML in the file manager lands here."""
        i = self.page_index("config")
        if i is None:
            return
        self.select_page("config")
        page = self.pages[i]
        if hasattr(page, "open"):
            page.open(path, server_name)

    def open_model(self, path, server_name=None):
        """Double-clicking a .pt / .safetensors file lands here: models table + highlight."""
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if "." in name:                                        # file → use its parent directory
            name = path.rsplit("/", 1)[0].rsplit("/", 1)[-1]
        i = self.page_index("models")
        if i is None:
            return
        self.select_page("models")
        page = self.pages[i]
        if hasattr(page, "activated"):
            page.activated()
        if hasattr(page, "table"):
            self._select_async(page.table, name)

    def open_dataset(self, path, server_name=None):
        """Double-clicking a dataset folder or a shard lands here: datasets table + highlight."""
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if "." in name:
            name = path.rsplit("/", 1)[0].rsplit("/", 1)[-1]
        i = self.page_index("datasets")
        if i is None:
            return
        self.select_page("datasets")
        page = self.pages[i]
        if hasattr(page, "activated"):
            page.activated()
        if hasattr(page, "table"):
            self._select_async(page.table, name)

    def start_training(self, config_path, values=None, server_name=None):
        """Called from the file manager and the config editor."""
        i = self.page_index("training")
        if i is None:
            return
        self.select_page("training")
        page = self.pages[i]
        if hasattr(page, "start"):
            page.start(config_path, values or {}, server_name)

    def _select_async(self, table, needle, tries=25):
        """TablePage populates asynchronously (server round-trip), so retry a few times."""
        if self._select_row_by_match(table, needle):
            return
        if tries > 0:
            QTimer.singleShot(80, lambda: self._select_async(table, needle, tries - 1))

    @staticmethod
    def _select_row_by_match(table, needle):
        if not needle:
            return False
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                it = table.item(r, c)
                if it and it.text().strip().rstrip("/") == needle:
                    table.selectRow(r)
                    table.scrollToItem(it)
                    return True
        return False

    # ── reacting to servers ──────────────────────────────────────────────────
    def paint_led(self, name, up):
        self.leds[name].setText(f'<span style="color:{"#2a2" if up else "#c22"}">●</span> {name}  ')

    def on_state(self, name, up):
        self.paint_led(name, up)
        self.log(f"{name}: {'connected' if up else 'disconnected'}")
        if up:
            self.servers[name].call("jobs.watch")
            if name == self.chat.target.currentText():
                self.rebind()

    def on_event(self, server, name, data):
        if name == "job":
            self.jobs.upsert(server, data)
        elif name == "error":
            self.log(f"{server}: ERROR {data}")
            self.statusBar().showMessage(f"{server}: {data}", 6000)
        for p in self.pages:
            getattr(p, "on_event", lambda *a: 0)(server, name, data)

    # ── menu / toolbar actions (do_<name>) ───────────────────────────────────
    def do_connect(self):
        for s in self.servers.values():
            s.start()

    def do_send(self): self.chat.send()
    def do_stop(self): self.chat.stop()
    def do_clear(self): self.chat.view.reset()
    def do_quit(self): self.close()
    def do_refresh(self): getattr(self.pages[self.stack.currentIndex()], "activated", lambda: 0)()
    def do_about(self): QMessageBox.about(self, "LLM-Lab", "LLM-Lab — proof of concept.\nGUI and compute are separate processes.")

    def do_theme(self):
        names = list(self.cfg["themes"])
        self.set_theme(names[(names.index(self.theme) + 1) % len(names)])

    def do_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "settings.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps({k: f.values() for k, f in self.forms.items()}, indent=1), "utf-8")

    def do_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open settings", "", "JSON (*.json)")
        for k, v in (json.loads(Path(path).read_text("utf-8")).items() if path else []):
            if k in self.forms:
                self.forms[k].load(v)

    def closeEvent(self, e):
        self.prefs.setValue("geometry", self.saveGeometry())
        self.prefs.setValue("state", self.saveState())
        super().closeEvent(e)


def load_config():
    toml = lambda p: tomllib.loads(p.read_text("utf-8"))
    plugins = sorted(((p.stem, toml(p)) for p in (ROOT / "config" / "plugins").glob("*.toml")), key=lambda kv: kv[1].get("order", 99))
    return toml(ROOT / "config" / "app.toml"), dict(plugins)


def spawn_demo(cfg):
    for s in cfg["server"]:
        p = subprocess.Popen([sys.executable, str(ROOT / "server" / "server.py"), s["name"], s["url"].rsplit(":", 1)[1]])
        atexit.register(p.terminate)


if __name__ == "__main__":
    cfg, plugins = load_config()
    if "--demo" in sys.argv:
        spawn_demo(cfg)
    app = QApplication(sys.argv)
    app.setStyle("Windows")                                    # flat classic widgets, same on every OS
    win = Main(cfg, plugins)
    win.show()
    sys.exit(app.exec())
"""The only module that knows about the network. One Server object per backend.

Wire protocol (JSON text frames over a WebSocket, identical for local and remote servers):
  GUI    -> {"id": 7, "method": "generate", "params": {...}}
  server -> {"id": 7, "result": ...}  |  {"id": 7, "error": "..."}      (answer to a call)
  server -> {"event": "token", "data": {...}}                          (push, any time)

Note on naming: the push signal is called ``pushed`` rather than ``event`` on purpose.
``QObject.event`` is a virtual method Qt calls internally for every dispatched event;
overriding it with a Signal makes Qt raise "TypeError: native Qt signal is not callable".
"""
import json

from qtpy.QtCore import QObject, QTimer, QUrl, Signal
from qtpy.QtNetwork import QAbstractSocket
from qtpy.QtWebSockets import QWebSocket


class Server(QObject):
    pushed = Signal(str, str, object)   # (server, name, payload) for every push; remote errors arrive as "error"
    state = Signal(str, bool)           # (server, is_up)

    def __init__(self, name, url):
        super().__init__()
        self.name, self.url, self.up, self.pending, self.n = name, url, False, {}, 0
        self.ws, self.timer = QWebSocket(), QTimer(self)
        self.ws.connected.connect(lambda: self._set(True))
        self.ws.disconnected.connect(lambda: self._set(False))
        self.ws.textMessageReceived.connect(self._rx)
        self.timer.timeout.connect(self._open)                       # auto-reconnect

    def start(self):
        self.timer.start(3000)
        self._open()

    def call(self, method, params=None, cb=None):
        """Fire a request; cb(result) runs when the answer arrives. Never blocks the GUI."""
        if not self.up:
            return self.pushed.emit(self.name, "error", f"{method}: not connected")
        self.n += 1
        if cb:
            self.pending[self.n] = cb
        self.ws.sendTextMessage(json.dumps({"id": self.n, "method": method, "params": params or {}}))

    def _open(self):
        if self.ws.state() == QAbstractSocket.SocketState.UnconnectedState:
            self.ws.open(QUrl(self.url))

    def _set(self, up):
        self.up = up
        self.pending.clear()
        self.state.emit(self.name, up)

    def _rx(self, text):
        m = json.loads(text)
        if "event" in m:
            self.pushed.emit(self.name, m["event"], m["data"])
        elif "error" in m:
            self.pending.pop(m["id"], None)
            self.pushed.emit(self.name, "error", m["error"])
        elif cb := self.pending.pop(m["id"], None):
            cb(m["result"])
"""Center-area pages and the Jobs panel.

A page is any widget built by KINDS[kind](spec, win). Optional hooks: activated(), on_event(server, name, data).
"""
from bisect import bisect

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QColor, QFontDatabase, QTextCharFormat, QTextCursor
from qtpy.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QLabel, QMenu, QPlainTextEdit, QProgressBar,
                            QTableWidget, QTableWidgetItem, QTextEdit, QToolTip, QVBoxLayout, QWidget)

from explorer import ConfigPage, Explorer
from plots import PlotsPage


def fill_row(table, r, values):
    for c, v in enumerate(values):
        table.setItem(r, c, QTableWidgetItem(str(v)))


def make_table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setAlternatingRowColors(True)
    t.verticalHeader().hide()
    t.horizontalHeader().setStretchLastSection(True)
    return t


class Transcript(QTextEdit):
    """Streaming text; every token remembers its span + info so hovering shows probabilities/features."""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.reset()

    def reset(self):
        self.clear()
        self.starts, self.spans = [], []

    def put(self, text, bg=None, info=None):
        c = self.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        a, f = c.position(), QTextCharFormat()
        if bg:
            f.setBackground(bg)
        c.insertText(text, f)
        if info:
            self.starts.append(a)
            self.spans.append((a, c.position(), info))
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def viewportEvent(self, e):
        if e.type() == QEvent.Type.ToolTip:
            pos = self.cursorForPosition(e.pos()).position()
            i = bisect(self.starts, pos) - 1
            QToolTip.showText(e.globalPos(), self.spans[i][2] if i >= 0 and pos <= self.spans[i][1] else "")
            return True
        return super().viewportEvent(e)


class ChatPage(QWidget):
    def __init__(self, spec, win):
        super().__init__()
        self.win, self.busy, self.live = win, False, None
        self.target = QComboBox()
        self.target.addItems(win.servers)
        self.target.currentTextChanged.connect(lambda _: win.rebind())
        self.view, self.box = Transcript(), QPlainTextEdit()
        self.box.setPlaceholderText("Prompt…   Ctrl+Enter sends, Esc stops")
        self.box.setFixedHeight(90)
        top = QHBoxLayout()
        top.addWidget(QLabel("Run on server:"))
        top.addWidget(self.target)
        top.addStretch()
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.box)

    @property
    def server(self):
        return self.win.servers[self.target.currentText()]

    def send(self):
        text = self.box.toPlainText().strip()
        if not text or self.busy:
            return
        if not self.server.up:
            return self.win.statusBar().showMessage(f"{self.server.name} is offline", 4000)
        self.view.put(f"you> {text}\n")
        self.box.clear()
        self.busy, self.live = True, self.server.name
        self.server.call("generate", {"prompt": text, "plugins": self.win.values()}, self.done)

    def stop(self):
        if self.busy:
            self.server.call("stop")

    def done(self, r):
        self.busy = False
        self.view.put(f"\n■ {r['reason']} · {r['n']} tokens · {r['tps']:.1f} tok/s\n\n")

    def on_event(self, server, name, d):
        if server != self.live or not self.busy:
            return
        if name == "start":
            self.view.put(f"⚙ {d['note']}\nmodel> ")
        elif name == "token":
            p, heat = d["p"], self.win.forms["display"].values()["highlight"]     # local-only option
            info = (f"p = {p:.2f}\nalt: " + ", ".join(f"{t.strip() or '⏎'} {q:.2f}" for t, q in d["alt"])
                    + "\nSAE: " + ", ".join(f"#{i} ({a:.1f})" for i, a in d["feats"]))
            self.view.put(d["t"], QColor(255, 120, 60, int(255 * min(1, (1 - p) * 1.3))) if heat else None, info)


class TablePage(QWidget):
    """Any backend method that returns a list of dicts -> a table, merged across all connected servers."""

    def __init__(self, spec, win):
        super().__init__()
        self.spec, self.win, self.cols = spec, win, spec["columns"]
        self.table = make_table(["Server"] + [c.capitalize() for c in self.cols])
        QVBoxLayout(self).addWidget(self.table)

    def activated(self):
        self.table.setRowCount(0)
        for s in self.win.servers.values():
            if s.up:
                s.call(self.spec["call"], None, lambda rows, s=s: self.add(s.name, rows))

    def add(self, server, rows):
        for r in rows:
            i = self.table.rowCount()
            self.table.insertRow(i)
            fill_row(self.table, i, [server] + [r.get(c, "") for c in self.cols])


class Jobs(QWidget):
    """Every job on every server, live. Right-click to cancel."""

    def __init__(self, servers):
        super().__init__()
        self.servers, self.rows = servers, {}
        self.table = make_table(["Server", "ID", "Kind", "Name", "Status", "Progress"])
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.menu)
        QVBoxLayout(self).addWidget(self.table)

    def upsert(self, server, j):
        r = self.rows.get((server, j["id"]))
        if r is None:
            r = self.rows[server, j["id"]] = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setCellWidget(r, 5, QProgressBar())
        fill_row(self.table, r, [server, j["id"], j["kind"], j["name"], j["status"]])
        self.table.cellWidget(r, 5).setValue(int(j["progress"] * 100))

    def menu(self, pos):
        r = self.table.rowAt(pos.y())
        if r < 0:
            return
        server, jid = self.table.item(r, 0).text(), self.table.item(r, 1).text()
        m = QMenu(self)
        m.addAction(f"Cancel {jid} on {server}", lambda: self.servers[server].call("jobs.cancel", {"id": jid}))
        m.exec(self.table.viewport().mapToGlobal(pos))


def stub(spec, win):
    label = QLabel(f"<h2>{spec['title']}</h2>Placeholder — not implemented yet.")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


PAGES = {"chat": ChatPage, "table": TablePage, "log": lambda spec, win: win.logw, "stub": stub,
         "files": lambda spec, win: Explorer(win.servers, spec.get("root", "/")), "config": ConfigPage, "plots": PlotsPage}
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
