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
