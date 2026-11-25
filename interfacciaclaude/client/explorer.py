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
