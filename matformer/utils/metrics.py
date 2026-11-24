# matformer/utils/metrics.py
"""Metric series operations, renderer-agnostic: every UI (Tk plots today, web/ncurses tomorrow,
the TUI) consumes {key: [[step, value], ...]} produced by the server (`runs.metrics`) and shares
these helpers. Keeping them out of the widgets is what makes a second renderer cheap."""
import fnmatch


def ema(ys, a):
    """Exponential moving average, wandb-style: a=0 off, a→1 fully smoothed."""
    out, s = [], None
    for y in ys:
        s = y if s is None else a * s + (1 - a) * y
        out.append(s)
    return out


def select(series, globs):
    """Keys of `series` matching any of the comma-separated globs, in sorted order."""
    patterns = [g.strip() for g in globs.split(",") if g.strip()] or ["*"]
    keys = sorted({k for pat in patterns for k in series if fnmatch.fnmatch(k, pat)})
    return [(k, series[k]) for k in series if k in keys and series[k]]


def last(series, key):
    pts = series.get(key) or []
    return pts[-1][1] if pts else None


def downsampling_needed(points, max_points):
    """Plots don't need more vertices than pixels: stride-decimate above the threshold."""
    if len(points) <= max_points:
        return points
    stride = len(points) // max_points + 1
    return points[::stride] + [points[-1]]
