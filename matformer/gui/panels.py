# matformer/gui/panels.py
"""Panels: the abstraction under Runs & Plots, and the one any future visual will attach to.

A panel is a plain JSON dict — kind, title, what to pull, how to compose it — so a whole dashboard
is a file you can save, version and share. Rendering is a registry (`PANELS`) keyed by kind; a new
kind of picture is a new class with `render`, nothing else changes. That is the seam interpretability
will use: a direct-logit-attribution graph or an outlier-token table arrives as another kind, with
another `source`, and the grid, the menus, the export and the saved views come for free.

A panel reads from a **source**, always through the context, never by touching a socket:

    runs            the ticked runs' metric series (what training writes)
    rpc             any server command that answers with a Table  (tomorrow: attributions, stores)
    diff            two runs, field by field

Kinds: line, scatter, bar, histogram, table, image.
"""
import colorsys
import fnmatch
import statistics

import tkinter as tk
from tkinter import ttk

from matformer.gui.widgets import TableList
from matformer.utils.metrics import ema

try:
    import matplotlib
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False

AGGS = ("mean", "min", "max", "median", "last", "std", "p10", "p90")
SPLITS = ("per run", "per metric")
XAXES = ("step", "time", "epoch")
KINDS = ("line", "scatter", "bar", "histogram", "table")

PALETTE_CYCLE = ["#4c78d0", "#e45756", "#54a24b", "#f58529", "#b279a2", "#72b7c9",
                 "#ff9da6", "#9d755d", "#cab2d6", "#8c8c8c"]


def series_color(index, palette):
    """Deterministic and readable: the same line keeps its colour across redraws, and a dark theme
    gets a palette chosen to stay legible on it instead of a brightened light one."""
    if palette.get("name") == "dark":
        return PALETTE_CYCLE[index % len(PALETTE_CYCLE)]
    hue = (index * 0.618033988749895) % 1.0
    return hex_of(colorsys.hsv_to_rgb(hue, 0.72, 0.85))


def hex_of(rgb):
    return "#%02x%02x%02x" % tuple(int(round(channel * 255)) for channel in rgb)


def matches(key, globs):
    return any(fnmatch.fnmatch(key, pattern.strip()) for pattern in (globs or "*").split(","))


def aggregate(points, how):
    """One (x, y) series out of possibly many, aggregated per x: what a run-per-run line becomes
    when the metric is shared."""
    by = {}
    for x, y in points:
        by.setdefault(x, []).append(y)
    xs = sorted(by)
    take = {"mean": statistics.fmean, "min": min, "max": max, "median": statistics.median,
            "std": lambda values: statistics.pstdev(values) if len(values) > 1 else 0.0,
            "last": lambda values: values[-1], "p10": lambda values: _percentile(values, 0.10),
            "p90": lambda values: _percentile(values, 0.90)}.get(how, statistics.fmean)
    return xs, [take(by[x]) for x in xs]


def _percentile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Context:
    """What a panel is allowed to ask for. Everything a panel draws comes through here, so the same
    panel code serves training metrics today and server-side analyses tomorrow.

    runs()            the ticked run ids
    series(run, key)  (xs, ys) of one metric of one run
    keys()            every metric key seen
    config(run)       the flattened hyperparameters of a run
    rpc(method, params)  a Table dict, for panels whose source is a command"""

    def __init__(self, runs=None, series=None, keys=None, config=None, rpc=None):
        self._runs = runs or (lambda: [])
        self._series = series or (lambda run, key: ([], []))
        self._keys = keys or (lambda: [])
        self._config = config or (lambda run: {})
        self._rpc = rpc

    def runs(self):
        return list(self._runs())

    def series(self, run, key):
        return self._series(run, key)

    def keys(self):
        return list(self._keys())

    def config(self, run):
        return self._config(run) or {}

    def rpc(self, method, params=None):
        return self._rpc(method, params) if self._rpc else None


def panel_points(panel, context):
    """The (label, xs, ys, run, key) lines a panel asks its source for, split and aggregated exactly
    as the panel says."""
    wanted = [key for key in context.keys() if matches(key, panel.get("metrics", "*"))]
    lines = []
    if panel.get("split", "per run") == "per metric":
        for key in wanted:
            points = [point for run in context.runs() for point in zip(*context.series(run, key))]
            if points:
                xs, ys = aggregate(points, panel.get("agg", "mean"))
                lines.append((key, xs, ys, None, key))
        return lines
    for run in context.runs():
        for key in wanted:
            xs, ys = context.series(run, key)
            if xs:
                lines.append((f"{run[:18]} · {key}" if len(wanted) > 1 else run[:22],
                              xs, ys, run, key))
    return lines


def axis_values(panel, context, xs, key):
    """The abscissa a panel plots against: steps, wall-clock seconds, or epochs of the run."""
    if panel.get("xaxis", "step") == "step" or not xs:
        return list(xs)
    if panel.get("xaxis") == "epoch":
        total = context.config(key or "").get("training/total_steps") or 0
        return [x / total if total else x for x in xs]
    return list(xs)


class Panel:
    """Base: a panel knows how to draw itself into an existing matplotlib axes, and how to fill a
    Tk frame when it is not a picture at all (tables). Subclasses override `draw` or `widget`."""

    kind = "line"
    needs_axes = True

    def __init__(self, spec, context, page=None):
        self.spec, self.context, self.page = spec, context, page

    # ── the picture path ──────────────────────────────────────────────────────
    def draw(self, ax, palette):
        raise NotImplementedError

    def decorate(self, ax, palette):
        ax.set_title(self.spec.get("title", ""), fontsize=9, color=palette["text"])
        ax.grid(True, color=palette["zebra"], linewidth=0.6)
        ax.tick_params(colors=palette["muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(palette["border"])
        ax.set_facecolor(palette["surface"])
        if self.spec.get("logy"):
            ax.set_yscale("log")

    # ── the widget path (tables and anything Tk-native) ───────────────────────
    def widget(self, parent, app):
        return None

    def legend(self, ax, palette):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="best", framealpha=0.6, labels=labels)


class LinePanel(Panel):
    kind = "line"

    def draw(self, ax, palette):
        smooth = float(self.spec.get("smooth", 0.0) or 0.0)
        for index, (label, xs, ys, run, key) in enumerate(panel_points(self.spec, self.context)):
            values = ema(ys, smooth) if smooth else list(ys)
            if self.spec.get("logy"):
                values = [max(value, 1e-12) for value in values]
            colour = self.spec.get("color") or series_color(index, palette)
            emphasis = self.spec.get("focus")
            alpha = 1.0 if not emphasis or emphasis == run else 0.22
            ax.plot(xs, values, color=colour, alpha=alpha,
                    lw=2.1 if emphasis and emphasis == run else 1.5, label=label)
        self.legend(ax, palette)


class ScatterPanel(Panel):
    kind = "scatter"

    def draw(self, ax, palette):
        x_key = self.spec.get("x") or self.spec.get("metrics")
        for index, (label, xs, ys, run, key) in enumerate(panel_points(self.spec, self.context)):
            colour = self.spec.get("color") or series_color(index, palette)
            ax.scatter(xs, ys, s=10, color=colour, alpha=0.75, label=label)
        if self.spec.get("x"):
            ax.set_xlabel(self.spec["x"], fontsize=8, color=palette["muted"])
        self.legend(ax, palette)


class BarPanel(Panel):
    """One bar per (run, metric): the last value, or the aggregate — how runs are compared at a
    glance, and what a `diff` source turns into when a baseline is set."""

    kind = "bar"

    def draw(self, ax, palette):
        rows = []
        for run in self.context.runs():
            for key in [k for k in self.context.keys() if matches(k, self.spec.get("metrics", "*"))]:
                xs, ys = self.context.series(run, key)
                if ys:
                    _, value = aggregate(list(zip(xs, ys)), self.spec.get("agg", "last"))
                    rows.append((f"{run[:14]}\n{key}" if self.spec.get("split") != "per metric" else key,
                                 value[0], run))
        if self.spec.get("baseline"):
            base = dict((label, value) for label, value, _ in rows
                        if label.startswith(self.spec["baseline"][:14]))
            rows = [(label, value - base.get(label.split("\n")[-1], value) if base else value, run)
                    for label, value, run in rows]
        for index, (label, value, _run) in enumerate(rows[:40]):
            ax.bar(index, value, color=series_color(index, palette), width=0.72)
        ax.set_xticks(range(len(rows[:40])))
        ax.set_xticklabels([label for label, _, _ in rows[:40]], fontsize=6, rotation=35,
                           ha="right", color=palette["muted"])
        ax.axhline(0, color=palette["border"], linewidth=0.8)


class HistogramPanel(Panel):
    kind = "histogram"

    def draw(self, ax, palette):
        buckets = int(self.spec.get("bins", 30))
        values = []
        for _label, _xs, ys, _run, _key in panel_points(self.spec, self.context):
            values.extend(ys)
        if values:
            ax.hist(values, bins=buckets, color=self.spec.get("color")
                    or series_color(0, palette), alpha=0.8)
        ax.set_ylabel("count", fontsize=8, color=palette["muted"])


class TablePanel(Panel):
    """Rows, not pixels: the last value of every metric per run today; any server-side table
    (outlier tokens, attributions, activation stats) tomorrow, via `source: rpc`."""

    kind = "table"
    needs_axes = False

    def widget(self, parent, app):
        frame = ttk.Frame(parent)
        table = TableList(app, frame, height=10)
        rows, columns = self._rows()
        table.headings(columns or ["value"])
        table.fill(rows, "no data")
        table.pack(fill="both", expand=True)
        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=2)
        ttk.Button(bar, text="Export CSV…", style="Ghost.TButton",
                   command=lambda: _export_rows(app, columns, rows)).pack(side="right")
        return frame, table

    def _rows(self):
        if self.spec.get("source") == "rpc":
            payload = self.context.rpc(self.spec.get("method"), self.spec.get("params") or {})
            if isinstance(payload, dict) and payload.get("__table__"):
                return payload["rows"], payload["columns"]
            return [], ["value"]
        keys = [k for k in self.context.keys() if matches(k, self.spec.get("metrics", "*"))]
        columns = ["run"] + keys
        rows = []
        for run in self.context.runs():
            row = {"run": run}
            for key in keys:
                xs, ys = self.context.series(run, key)
                if ys:
                    _x, values = aggregate(list(zip(xs, ys)), self.spec.get("agg", "last"))
                    row[key] = values[0]
            rows.append(row)
        return rows, columns


def _export_rows(app, columns, rows):
    from tkinter import filedialog
    import csv
    path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
    if not path:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(column, "") for column in columns])
    app.toast(f"saved {path}")


PANELS = {panel.kind: panel for panel in
          (LinePanel, ScatterPanel, BarPanel, HistogramPanel, TablePanel)}


def panel_for(spec, context, page=None):
    """Unknown kinds fall back to a line instead of breaking a saved view."""
    return PANELS.get(spec.get("kind", "line"), LinePanel)(spec, context, page)


# ── the editor: every option a panel takes, in one place ───────────────────────────────────────
SPEC_FIELDS = [
    {"name": "title", "label": "Title", "kind": "text", "doc": "shown above the panel"},
    {"name": "kind", "label": "Picture", "kind": "choice", "options": list(KINDS)},
    {"name": "metrics", "label": "Metric globs", "kind": "text",
     "doc": "comma separated: train/loss, eval/*, lr*"},
    {"name": "split", "label": "One line", "kind": "choice", "options": list(SPLITS)},
    {"name": "agg", "label": "Aggregate", "kind": "choice", "options": list(AGGS)},
    {"name": "xaxis", "label": "X axis", "kind": "choice", "options": list(XAXES)},
    {"name": "smooth", "label": "Smoothing (EMA)", "kind": "float", "min": 0, "max": 0.99,
     "step": 0.05},
    {"name": "bins", "label": "Bins", "kind": "int", "min": 2, "max": 500},
    {"name": "logy", "label": "Log Y", "kind": "bool"},
    {"name": "span", "label": "Width (cells)", "kind": "int", "min": 1, "max": 4},
    {"name": "color", "label": "Colour", "kind": "text", "doc": "#rrggbb, empty = cycle"},
]


def defaults():
    return {"title": "", "kind": "line", "metrics": "*", "split": "per run", "agg": "mean",
            "xaxis": "step", "smooth": 0.0, "bins": 30, "logy": False, "span": 1, "color": ""}
