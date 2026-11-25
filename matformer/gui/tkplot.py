"""A small Tk plot engine for the Runs page.

Matplotlib gives a correct picture and costs a Tk canvas per panel every time data moves. This
canvas draws the same panels in Tk primitives: no backend, no figure, no redraw in another process.
It is also the place where the interaction lives: hover reads the exact value, dragging zooms the x
axis, the wheel zooms around the pointer, double-click resets, and right-click opens the panel menu.

Only the picture is Tk. The data semantics stay in `panels.py`: `panel_points` selects, splits and
aggregates the series, and the same spec drives the PNG export through matplotlib on demand.
"""
import math
import bisect

import tkinter as tk

from matformer.gui.panels import panel_points, series_color
from matformer.utils.metrics import ema


def _fmt(value):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "-"
    magnitude = abs(value)
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if magnitude >= limit:
            return f"{value / limit:.2f}{suffix}"
    if magnitude and magnitude < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3g}"


def _decimate(xs, ys, budget):
    """Min/max buckets: keep the shape of a spike when a series has more points than pixels."""
    count = len(xs)
    if count <= budget:
        return list(xs), list(ys)
    bucket = max(1, math.ceil(count / max(1, budget // 2)))
    out_x, out_y = [], []
    for start in range(0, count, bucket):
        end = min(count, start + bucket)
        chunk = ys[start:end]
        low = min(range(len(chunk)), key=chunk.__getitem__)
        high = max(range(len(chunk)), key=chunk.__getitem__)
        order = (low, high) if low <= high else (high, low)
        for offset in order:
            out_x.append(xs[start + offset])
            out_y.append(chunk[offset])
    return out_x, out_y


def _blend(first, second, amount):
    a = [int(first[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(second[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(x + (y - x) * amount) for x, y in zip(a, b))


class TkPlotCanvas(tk.Canvas):
    """One panel, drawn with Tk primitives and interactive."""

    def __init__(self, app, parent, spec, context, page):
        palette = app.palette()
        super().__init__(parent, bg=palette["surface"], highlightthickness=0, bd=0,
                         takefocus=1)
        self.app, self.spec, self.context, self.page = app, spec, context, page
        self.xlim = None
        self.full_xlim = None
        self._drag = None
        self._lines = []
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda e: self._clear_hover())
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._drag_motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-1>", lambda e: self.reset())
        self.bind("<Button-3>", self._menu)
        self.bind("<Button-4>", lambda e: self._wheel(e, 1))
        self.bind("<Button-5>", lambda e: self._wheel(e, -1))
        self.bind("<MouseWheel>", self._wheel_event)
        self.redraw()

    # ── data and geometry ─────────────────────────────────────────────────────
    def _visible_lines(self):
        smooth = float(self.spec.get("smooth", 0.0) or 0.0)
        budget = max(200, (self.winfo_width() - 60) * 2)
        lines = []
        for index, (label, xs, ys, run, key) in enumerate(
                panel_points(self.spec, self.context)):
            values = ema(ys, smooth) if smooth else list(ys)
            x_values = list(xs)
            if len(x_values) > budget and self.spec.get("kind", "line") in ("line", "scatter"):
                x_values, values = _decimate(x_values, values, budget)
            colour = self.spec.get("color") or series_color(index, self.app.palette())
            if self.spec.get("focus") and run and run != self.spec["focus"]:
                colour = _blend(colour, self.app.palette()["surface"], 0.72)
            lines.append((label, x_values, values, colour, run, key))
        return lines

    def _bounds(self, lines, kind):
        xs = [x for _l, x, _y, _c, _r, _k in lines for x in x]
        ys = [y for _l, _x, y, _c, _r, _k in lines for y in y]
        if kind == "bar":
            xs = list(range(len(lines)))
            ys = [line[2][-1] if line[2] else 0.0 for line in lines]
        if not xs or not ys:
            return None
        if self.xlim:
            low, high = self.xlim
        else:
            low, high = min(xs), max(xs)
            if low == high:
                low, high = low - 0.5, high + 0.5
            self.full_xlim = (low, high)
        if self.spec.get("logy") and kind in ("line", "scatter"):
            ys = [math.log10(max(y, 1e-12)) for y in ys]
        y_low, y_high = min(ys), max(ys)
        if y_low == y_high:
            y_low -= 0.5
            y_high += 0.5
        pad = (y_high - y_low) * 0.08
        return low, high, y_low - pad, y_high + pad, xs, ys

    def redraw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if width < 60 or height < 50:
            return
        palette = self.app.palette()
        self.configure(bg=palette["surface"])
        lines = self._visible_lines()
        kind = self.spec.get("kind", "line")
        bounds = self._bounds(lines, kind)
        if bounds is None:
            self.create_text(width / 2, height / 2, text="no data",
                             fill=palette["muted"], font=("TkDefaultFont", 9))
            return
        low, high, y_low, y_high, _xs, _ys = bounds
        left, right, top, bottom = 48, 12, 10, 26
        plot_w, plot_h = max(1, width - left - right), max(1, height - top - bottom)

        def px(x):
            return left + (x - low) / (high - low or 1) * plot_w

        def py(y):
            return top + (y_high - y) / (y_high - y_low or 1) * plot_h

        # plot background and grid
        self.create_rectangle(left, top, left + plot_w, top + plot_h,
                              fill=palette["surface"], outline=palette["border"])
        for i in range(5):
            x = left + plot_w * i / 4
            y = top + plot_h * i / 4
            self.create_line(x, top, x, top + plot_h, fill=palette["zebra"])
            self.create_line(left, y, left + plot_w, y, fill=palette["zebra"])
            value = low + (high - low) * i / 4
            self.create_text(x, top + plot_h + 12, text=_fmt(value),
                             fill=palette["muted"], font=("TkDefaultFont", 8))
            value = y_high - (y_high - y_low) * i / 4
            shown = 10 ** value if self.spec.get("logy") and kind in ("line", "scatter") else value
            self.create_text(left - 6, y, text=_fmt(shown), anchor="e",
                             fill=palette["muted"], font=("TkDefaultFont", 8))

        if kind == "bar":
            self._draw_bars(lines, left, top, plot_w, plot_h, low, high, y_low, y_high,
                            palette)
        elif kind == "histogram":
            self._draw_histogram(lines, left, top, plot_w, plot_h, low, high,
                                 y_low, y_high, palette)
        else:
            self._draw_series(lines, kind, px, py, palette)
            self._legend(lines, left + plot_w - 8, top + 6, palette)
        self._lines = lines

    def _draw_series(self, lines, kind, px, py, palette):
        for label, xs, ys, colour, _run, _key in lines:
            points = [(px(x), py(y)) for x, y in zip(xs, ys)
                      if math.isfinite(x) and math.isfinite(y)]
            if not points:
                continue
            if kind == "scatter":
                for x, y in points:
                    self.create_oval(x - 2, y - 2, x + 2, y + 2, fill=colour, outline="")
            elif len(points) == 1:
                # A task summary is one point per metric; a one-coordinate line is not a line.
                x, y = points[0]
                self.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, fill=colour, outline="")
            else:
                self.create_line(*[coordinate for point in points for coordinate in point],
                                 fill=colour, width=1.6, capstyle="round")

    def _draw_bars(self, lines, left, top, plot_w, plot_h, low, high, y_low, y_high, palette):
        count = max(1, len(lines))
        slot = plot_w / count
        zero = top + (y_high - 0) / (y_high - y_low or 1) * plot_h
        for index, (label, _xs, ys, colour, _run, _key) in enumerate(lines):
            value = ys[-1] if ys else 0.0
            y = top + (y_high - value) / (y_high - y_low or 1) * plot_h
            x0 = left + index * slot + slot * 0.12
            x1 = left + (index + 1) * slot - slot * 0.12
            self.create_rectangle(x0, y, x1, zero, fill=colour, outline="")
            self.create_text((x0 + x1) / 2, top + plot_h + 12, text=label[:14],
                             fill=palette["muted"], font=("TkDefaultFont", 7))

    def _draw_histogram(self, lines, left, top, plot_w, plot_h, low, high, y_low, y_high,
                        palette):
        values = [y for _label, _xs, ys, _c, _r, _k in lines for y in ys]
        if not values:
            return
        count = max(2, min(60, int(self.spec.get("bins", 30))))
        low, high = min(values), max(values)
        span = high - low or 1
        counts = [0] * count
        for value in values:
            index = min(count - 1, int((value - low) / span * count))
            counts[index] += 1
        peak = max(counts) or 1
        slot = plot_w / count
        for index, number in enumerate(counts):
            x0 = left + index * slot
            x1 = x0 + slot - 1
            y = top + plot_h - (number / peak) * plot_h
            self.create_rectangle(x0, y, x1, top + plot_h,
                                  fill=series_color(0, palette), outline="")
        self.create_text(left + plot_w / 2, top + plot_h + 12,
                         text=f"{_fmt(low)} .. {_fmt(high)}", fill=palette["muted"],
                         font=("TkDefaultFont", 8))

    def _legend(self, lines, right, top, palette):
        shown = lines[:8]
        x = right
        y = top
        for label, _xs, _ys, colour, _run, _key in shown:
            text = label[:26]
            self.create_text(x, y, text=text, anchor="ne", fill=palette["muted"],
                             font=("TkDefaultFont", 7))
            self.create_line(x - 4 - 7 * min(len(text), 26), y, x - 6, y,
                             fill=colour, width=2)
            y += 13

    # ── interaction ───────────────────────────────────────────────────────────
    def _nearest(self, x):
        low, high, _yl, _yh, _xs, _ys = self._bounds(self._lines or self._visible_lines(),
                                                     self.spec.get("kind", "line"))
        if not (low <= x <= high):
            return None
        width = self.winfo_width()
        left, right = 48, 12
        plot_w = max(1, width - left - right)
        target = low + (x - left) / plot_w * (high - low)
        best = None
        for label, xs, ys, _colour, _run, _key in self._lines:
            if not xs:
                continue
            index = min(range(len(xs)), key=lambda i: abs(xs[i] - target))
            distance = abs(xs[index] - target)
            if best is None or distance < best[0]:
                best = (distance, label, xs[index], ys[index] if index < len(ys) else None)
        return best

    def _motion(self, event):
        self._clear_hover()
        if not self._lines or self.spec.get("kind", "line") in ("bar", "histogram"):
            return
        best = self._nearest(event.x)
        if not best:
            return
        _distance, label, x, y = best
        palette = self.app.palette()
        low, high, _yl, _yh, _xs, _ys = self._bounds(self._lines, self.spec.get("kind"))
        width, height = self.winfo_width(), self.winfo_height()
        left, right, top, bottom = 48, 12, 10, 26
        plot_w, plot_h = max(1, width - left - right), max(1, height - top - bottom)
        px = left + (x - low) / (high - low or 1) * plot_w
        text = f"{label}: {_fmt(y)}   x {_fmt(x)}"
        self.create_rectangle(left + 3, top + 3, left + 15 + 7 * len(text), top + 21,
                              fill=palette["surface"], outline=palette["border"], tags="hover")
        self.create_text(left + 7, top + 6, text=text, anchor="nw", fill=palette["text"],
                         font=("TkDefaultFont", 8), tags="hover")
        self.create_line(px, top, px, top + plot_h, fill=palette["accent"], tags="hover")

    def _clear_hover(self):
        self.delete("hover")

    def _press(self, event):
        if event.num == 3:
            self.page._panel_menu(event, self.spec, self)
            return
        if 48 <= event.x <= self.winfo_width() - 12:
            self._drag = event.x

    def _drag_motion(self, event):
        if self._drag is None:
            return
        self.delete("drag")
        self.create_line(self._drag, 10, self._drag, self.winfo_height() - 26,
                         fill=self.app.palette()["accent"], tags="drag")
        self.create_line(event.x, 10, event.x, self.winfo_height() - 26,
                         fill=self.app.palette()["accent"], tags="drag")

    def _release(self, event):
        if self._drag is None:
            return
        start, self._drag = self._drag, None
        self.delete("drag")
        if abs(event.x - start) < 6:
            return
        width = self.winfo_width()
        plot_w = max(1, width - 60)
        low, high = self._pixel_domain(min(start, event.x), max(start, event.x), plot_w)
        self.set_xlim(low, high, source=self)

    def _pixel_domain(self, x0, x1, plot_w):
        low, high = self.full_xlim or (0, 1)
        return (low + (x0 - 48) / plot_w * (high - low),
                low + (x1 - 48) / plot_w * (high - low))

    def _wheel(self, event, direction):
        if not self.full_xlim:
            return
        low, high = self.xlim or self.full_xlim
        factor = 0.8 if direction > 0 else 1.25
        width = self.winfo_width()
        plot_w = max(1, width - 60)
        anchor = low + max(0, min(1, (event.x - 48) / plot_w)) * (high - low)
        self.set_xlim(anchor - (anchor - low) * factor, anchor + (high - anchor) * factor,
                      source=self)

    def _wheel_event(self, event):
        self._wheel(event, 1 if event.delta > 0 else -1)

    def _menu(self, event):
        self.page._panel_menu(event, self.spec, self)

    def reset(self):
        self.set_xlim(None, None, source=self)

    def set_xlim(self, low, high, source=None):
        self.xlim = None if low is None else (low, high)
        self.redraw()
        if self.page.xlink.get() and source is not None:
            for _spec, canvas, _panel in getattr(self.page, "_canvases", []):
                if canvas is not source and isinstance(canvas, TkPlotCanvas):
                    canvas.xlim = self.xlim
                    canvas.redraw()
