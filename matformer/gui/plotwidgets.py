# matformer/gui/plotwidgets.py
"""Small Tk-only plots shared by pages that do not need the full runs dashboard: histograms,
scatter plots, axes and labels. No matplotlib, no extra dependency on a laptop."""
import tkinter as tk


class MiniPlot(tk.Canvas):
    def __init__(self, app, parent):
        palette = app.palette()
        super().__init__(parent, bg=palette["surface"], highlightthickness=0)
        self.app, self.kind, self.data = app, None, None
        self.bind("<Configure>", lambda e: self.redraw())

    def bars(self, edges, counts, title=""):
        self.kind, self.data = "bars", (edges, counts, title)
        self.redraw()

    def scatter(self, points, variance=None, title=""):
        self.kind, self.data = "scatter", (points, variance, title)
        self.redraw()

    def clear(self):
        self.kind, self.data = None, None
        self.redraw()

    def redraw(self):
        self.delete("all")
        p, w, h = self.app.palette(), self.winfo_width(), self.winfo_height()
        if w < 40 or h < 40:
            return
        if not self.kind:
            self.create_text(w / 2, h / 2, text="Summary, histogram, scatter or outliers",
                             fill=p["faint"], font=("TkDefaultFont", 10))
            return
        left, top, right, bottom = 56, 26, w - 16, h - 38
        self.create_line(left, bottom, right, bottom, fill=p["border"])
        self.create_line(left, top, left, bottom, fill=p["border"])
        if self.kind == "bars":
            edges, counts, title = self.data
            high = max(counts) or 1
            width = (right - left) / max(len(counts), 1)
            for i, count in enumerate(counts):
                x0 = left + i * width + 1
                x1 = left + (i + 1) * width - 1
                y0 = bottom - (count / high) * (bottom - top)
                self.create_rectangle(x0, y0, x1, bottom, fill=p["accent"], outline="")
            for i in (0, len(counts) // 2, len(counts)):
                if 0 <= i < len(edges):
                    x = left + (i / max(len(counts), 1)) * (right - left)
                    self.create_text(x, bottom + 14, text=f"{edges[i]:.3g}", fill=p["muted"],
                                     font=("TkDefaultFont", 8))
            self.create_text(left, top - 10, anchor="w", text=title or f"max {high:.0f}",
                             fill=p["text"], font=("TkDefaultFont", 9, "bold"))
            self.create_text(right, top + 2, anchor="e", text=f"n={sum(counts):.0f}",
                             fill=p["muted"], font=("TkDefaultFont", 8))
        else:
            points, variance, title = self.data
            if not points:
                return
            xs = [one["x"] for one in points]
            ys = [one["y"] for one in points]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            spanx, spany = (x1 - x0) or 1, (y1 - y0) or 1
            for one in points:
                x = left + (one["x"] - x0) / spanx * (right - left)
                y = bottom - (one["y"] - y0) / spany * (bottom - top)
                self.create_oval(x - 3, y - 3, x + 3, y + 3, fill=p["accent"], outline="")
            self.create_text(left, top - 10, anchor="w", text=title or "scatter",
                             fill=p["text"], font=("TkDefaultFont", 9, "bold"))
            self.create_text(left - 6, bottom + 14, anchor="e", text=f"{x0:.3g}",
                             fill=p["muted"], font=("TkDefaultFont", 8))
            self.create_text(right, bottom + 14, anchor="e", text=f"{x1:.3g}",
                             fill=p["muted"], font=("TkDefaultFont", 8))
            self.create_text(left - 6, top, anchor="e", text=f"{y1:.3g}",
                             fill=p["muted"], font=("TkDefaultFont", 8))
            self.create_text(left - 6, bottom, anchor="e", text=f"{y0:.3g}",
                             fill=p["muted"], font=("TkDefaultFont", 8))
