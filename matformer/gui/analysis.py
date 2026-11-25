# matformer/gui/analysis.py
"""One reader for the flat metrics analysis tasks produce.

The same keys come from static weight analysis (`weights/.../kurtosis`), activation analysis
(`activations/.../std`), a training run's metrics (`train/loss` over steps) or a task summary.
Parsing them once into records gives the views every analysis needs: a table, a histogram, a
scatter of a metric across layers/modules, and the outliers with their key still visible."""
from __future__ import annotations

import math
import re
import tkinter as tk
from tkinter import ttk

from matformer.gui import exports
from matformer.gui.plotwidgets import MiniPlot
from matformer.gui.widgets import Page, TableList, popup

_LAYER = re.compile(r"(?:layers?|blocks?|layer)[._/](\d+)", re.IGNORECASE)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None
    return float(value) if math.isfinite(float(value)) else None


def parse_flat(summary, kind_prefix="summary"):
    """A flat key -> value dict into records: key, kind, metric, subject, layer, value."""
    rows = []
    for key, value in (summary or {}).items():
        number = _number(value)
        if number is None:
            continue
        parts = str(key).split("/")
        metric = parts[-1] if len(parts) > 1 else str(key)
        subject = "/".join(parts[1:-1]) if len(parts) > 2 else (parts[0] if len(parts) > 1 else "")
        kind = parts[0] if len(parts) > 1 else kind_prefix
        match = _LAYER.search("/".join(parts))
        layer = int(match.group(1)) if match else None
        rows.append({"key": str(key), "kind": kind, "metric": metric, "subject": subject,
                     "layer": layer, "value": number, "x": layer if layer is not None else 0.0})
    return rows


class AnalysisView(ttk.Frame):
    """Table + plot of a flat summary; the same widget serves task results and run metrics."""

    def __init__(self, app, parent):
        super().__init__(parent)
        self.app, self.rows, self.time_series = app, [], False
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="View").pack(side="left")
        self.view = ttk.Combobox(bar, values=["Table", "Histogram", "Scatter", "Outliers"],
                                 width=11, state="readonly")
        self.view.set("Table")
        self.view.pack(side="left", padx=4)
        self.view.bind("<<ComboboxSelected>>", lambda e: self._render())
        ttk.Label(bar, text="Kind").pack(side="left", padx=(8, 0))
        self.kind = ttk.Combobox(bar, width=13, state="readonly")
        self.kind.pack(side="left", padx=3)
        self.kind.bind("<<ComboboxSelected>>", lambda e: self._metrics())
        ttk.Label(bar, text="Metric").pack(side="left", padx=(8, 0))
        self.metric = ttk.Combobox(bar, width=18, state="readonly")
        self.metric.pack(side="left", padx=3)
        self.metric.bind("<<ComboboxSelected>>", lambda e: self._render())
        ttk.Label(bar, text="Subject").pack(side="left", padx=(8, 0))
        self.filter = ttk.Entry(bar, width=22)
        self.filter.pack(side="left", padx=3)
        self.filter.bind("<Return>", lambda e: self._render())
        ttk.Button(bar, text="Apply", style="Ghost.TButton",
                   command=self._render).pack(side="left", padx=3)
        ttk.Button(bar, text="Export…", style="Ghost.TButton",
                   command=self.export).pack(side="left", padx=3)
        self.count = ttk.Label(bar, text="", style="Muted.TLabel")
        self.count.pack(side="right")
        self.pan = ttk.PanedWindow(self, orient="horizontal")
        self.pan.pack(fill="both", expand=True)
        self.left = ttk.Frame(self.pan)
        self.pan.add(self.left, weight=2)
        left = self.left
        self.table = TableList(app, left, height=14,
                               on_double=lambda row: row and self.app.copy(row.get("key", "")))
        self.base_columns = ["subject", "metric", "value", "key"]
        self.table.headings(self.base_columns)
        self.table.fill([], "run an analysis first")
        self.table.pack(fill="both", expand=True)
        self.right = ttk.Frame(self.pan)
        self.pan.add(self.right, weight=3)
        self.plot = MiniPlot(app, self.right)
        self.plot.pack(fill="both", expand=True)

    def export(self):
        rows = [{key: row.get(key) for key in ("subject", "metric", "value", "key", "layer",
                                               "kind", "x")} for row in self._filtered()]
        if not rows:
            return self.app.toast("nothing to export", error=True)
        path = exports.ask("Export the analysis", f"{self.metric.get() or 'analysis'}.csv")
        if not path:
            return
        exports.write_rows(path, rows, title=f"Analysis — {self.metric.get()}")
        self.app.toast(f"saved {path}")

    # ── inputs ──────────────────────────────────────────────────────────────────
    def show_summary(self, summary, title=""):
        self.rows = parse_flat(summary)
        self.time_series = False
        self.title = title
        self._kinds()

    def show_metrics(self, metrics, title=""):
        rows = []
        for name, points in (metrics or {}).items():
            values = points if isinstance(points, list) else []
            for point in values:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    step, value = point[0], _number(point[1])
                    if value is None:
                        continue
                    rows.append({"key": name, "kind": "metric", "metric": name, "subject": name,
                                 "layer": None, "value": value,
                                 "x": float(_number(step) or len(rows))})
        self.rows = rows
        self.time_series = True
        self.title = title
        self._kinds()

    def _kinds(self):
        kinds = sorted({row["kind"] for row in self.rows})
        self.kind["values"] = ["all", *kinds]
        if self.kind.get() not in self.kind["values"]:
            self.kind.set("all")
        self._metrics()

    def _filtered(self):
        kind = self.kind.get()
        text = self.filter.get().strip().lower()
        rows = [row for row in self.rows if kind in ("", "all") or row["kind"] == kind]
        if text:
            rows = [row for row in rows if text in row["subject"].lower()
                    or text in row["key"].lower()]
        return rows

    def _metrics(self):
        rows = self._filtered()
        names = sorted({row["metric"] for row in rows})
        self.metric["values"] = names
        if self.metric.get() not in names:
            self.metric.set(names[0] if names else "")
        self._render()

    def _selected(self):
        return [row for row in self._filtered() if row["metric"] == self.metric.get()]

    def _pane(self, show_left, show_right):
        panes = set(self.pan.panes())
        if show_left and str(self.left) not in panes:
            self.pan.insert(0, self.left, weight=2)
        if not show_left and str(self.left) in panes:
            self.pan.forget(self.left)
        if show_right and str(self.right) not in panes:
            self.pan.add(self.right, weight=3)
        if not show_right and str(self.right) in panes:
            self.pan.forget(self.right)

    def _render(self):
        rows = self._filtered()
        view = self.view.get()
        self._pane(view in ("Table", "Outliers"), view in ("Histogram", "Scatter", "Outliers"))
        self.count.configure(text=f"{len(rows)} numeric values")
        if view != "Outliers" and self.table.columns != self.base_columns:
            self.table.headings(self.base_columns)
        if view != "Outliers":
            shown = rows[:2000]
            self.table.fill([{"subject": row["subject"], "metric": row["metric"],
                              "value": f"{row['value']:.6g}", "key": row["key"]}
                             for row in shown],
                            "no numeric value for this summary"
                            if not rows else f"showing {len(shown)} of {len(rows)} rows")
        picked = self._selected()
        if view == "Histogram":
            self._histogram(picked)
        elif view == "Scatter":
            self._scatter(picked)
        elif view == "Outliers":
            self._outliers(picked)
        else:
            self.plot.clear()

    def _histogram(self, rows):
        values = [row["value"] for row in rows]
        if not values:
            return self.plot.clear()
        low, high = min(values), max(values)
        if high <= low:
            high = low + 1e-9
        bins = 40
        width = (high - low) / bins
        counts = [0] * bins
        for value in values:
            index = min(bins - 1, int((value - low) / width))
            counts[index] += 1
        edges = [low + i * width for i in range(bins + 1)]
        self.plot.bars(edges, counts, f"{self.metric.get()}  ({len(values)} values)")

    def _scatter(self, rows):
        if not rows:
            return self.plot.clear()
        points = []
        for index, row in enumerate(rows):
            x = row["x"] if self.time_series or row.get("layer") is not None else index
            points.append({"x": float(x), "y": row["value"], "key": row["key"]})
        self.plot.scatter(points, title=f"{self.metric.get()} across "
                                      + ("steps" if self.time_series else "subjects"))

    def _outliers(self, rows):
        if not rows:
            return self.plot.clear()
        values = [row["value"] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        std = variance ** .5 or 1e-12
        ranked = sorted(((abs(row["value"] - mean) / std, row) for row in rows),
                        key=lambda pair: pair[0], reverse=True)
        shown = []
        for z, row in ranked[:30]:
            shown.append({"subject": row["subject"], "metric": row["metric"],
                          "value": f"{row['value']:.6g}", "z": f"{z:.2f}",
                          "key": row["key"]})
        self.table.headings(["subject", "metric", "value", "z", "key"])
        self.table.fill(shown, "no outlier")
        self.plot.scatter([{"x": float(index), "y": z} for index, (z, _row) in enumerate(ranked[:60])],
                          title=f"|z| of {self.metric.get()} — mean {mean:.4g}, std {std:.4g}")
