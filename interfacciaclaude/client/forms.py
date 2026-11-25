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
