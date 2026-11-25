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
