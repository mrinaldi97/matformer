# matformer/gui/icons.py
"""Icons from the system's freedesktop icon themes (/usr/share/icons/...).

GTK/Qt/KDE apps resolve icons through a named, inheritable theme ("folder", "view-refresh",
"text-x-generic"...). The same directories work for Tk: we walk the theme inheritance chain
(chosen theme -> inherits -> hicolor), pick the PNG closest to the requested size, subsample if
bigger, and cache PhotoImages. Tk 8.6 cannot render SVG, so PNG sizes win; scalable-only icons
fall back gracefully.

If nothing is installed (bare window, headless tests) the bundled programmatic shapes render —
the GUI never shows empty squares. Selection persists in Preferences ("icon_theme": "auto",
"none", or any installed theme name)."""
import configparser
import os
import tkinter as tk
from pathlib import Path

SEARCH_DIRS = ["/usr/share/icons", "/usr/local/share/icons",
               str(Path.home() / ".local" / "share" / "icons"), "/usr/share/pixmaps"]

# canonical app concept -> freedesktop name candidates, in priority order
NAMES = {
    "dir":     ["folder", "inode-directory", "text-directory", "folder-sound"],
    "file":    ["text-x-generic", "empty", "document"],
    "config":  ["text-x-yaml", "application-x-yaml", "text-x-generic", "preferences-other"],
    "model":   ["application-x-object", "package-x-generic", "drive-harddisk", "text-x-generic"],
    "dataset": ["x-office-spreadsheet", "text-x-generic", "database", "folder"],
    "run":     ["system-run", "media-playback-start", "application-x-executable"],
    "server":  ["network-server", "computer", "drive-harddisk"],
    "plot":    ["office-chart-line", "accessories-dictionary", "image-x-generic"],
    "fav":     ["user-bookmarks", "bookmark-new", "emblem-default"],
    "up":      ["go-up", "go-top"],
    "refresh": ["view-refresh", "gtk-refresh"],
    "open":    ["document-open", "folder-open"],
    "save":    ["document-save", "media-floppy"],
    "delete":  ["edit-delete", "user-trash"],
    "rename":  ["edit-rename", "document-edit"],
    "copy":    ["edit-copy"],
    "newfolder": ["folder-new"],
    "stop":    ["media-playback-stop", "process-stop"],
    "registry": ["application-x-addon", "preferences-system", "text-x-generic"],
    "log":     ["text-x-log", "utilities-terminal", "text-x-generic"],
}
_EXT = {".yaml": "config", ".yml": "config", ".json": "dataset", ".jsonl": "dataset",
        ".csv": "dataset", ".ckpt": "model", ".pt": "model", ".safetensors": "model",
        ".bin": "model", ".log": "log", ".txt": "file", ".md": "file", ".py": "file"}

# ── programmatic fallback (12x12 blocks, painted at any size) ────────────────────────────────
_SHAPES = {
    "dir":     [(0, 3, 2, 5, "#c8a030"), (0, 4, 12, 12, "#e8b840"), (1, 5, 11, 11, "#f8d878")],
    "file":    [(1, 0, 11, 14, "#f8f8f8"), (0, 0, 12, 1, "#808080"), (0, 13, 12, 14, "#808080"),
                (2, 3, 10, 4, "#b0b0b0"), (2, 6, 10, 7, "#b0b0b0"), (2, 9, 8, 10, "#b0b0b0")],
    "model":   [(3, 3, 9, 9, "#405080"), (4, 4, 8, 8, "#7890c0"), (0, 4, 3, 5, "#808080"),
                (0, 7, 3, 8, "#808080"), (9, 4, 12, 5, "#808080"), (9, 7, 12, 8, "#808080"),
                (4, 0, 5, 3, "#808080"), (7, 0, 8, 3, "#808080"), (4, 9, 5, 12, "#808080"),
                (7, 9, 8, 12, "#808080")],
    "dataset": [(2, 2, 10, 4, "#3f8f6f"), (2, 5, 10, 7, "#4fa883"), (2, 8, 10, 10, "#3f8f6f"),
                (2, 1, 10, 2, "#5fc89f"), (2, 10, 10, 11, "#2f7059")],
    "config":  [(1, 0, 11, 14, "#f8f8f8"), (2, 2, 4, 4, "#c05020"), (5, 3, 10, 4, "#b0b0b0"),
                (2, 6, 4, 8, "#c05020"), (5, 7, 10, 8, "#b0b0b0"), (2, 10, 4, 12, "#c05020"),
                (5, 11, 10, 12, "#b0b0b0")],
    "run":     [(1, 10, 3, 12, "#0a246a"), (4, 7, 6, 12, "#0a246a"), (7, 4, 9, 12, "#0a246a"),
                (10, 2, 12, 12, "#0a246a")],
}


def _shape(name, size):
    im = tk.PhotoImage(name=f"mt_{name}_{size}", width=size, height=size)
    k = max(1, size // 12)
    for x0, y0, x1, y1, color in _SHAPES.get(name, _SHAPES["file"]):
        im.put(color, (x0 * k, y0 * k, x1 * k, y1 * k))
    return im


def installed_themes():
    out = []
    for base in SEARCH_DIRS:
        try:
            for entry in sorted(os.scandir(base), key=lambda e: e.name):
                if entry.is_dir() and (Path(entry.path) / "index.theme").exists():
                    out.append(entry.name)
        except OSError:
            continue
    return sorted(set(out))


def _chain(theme):
    """theme + its `Inherits` closure + hicolor, deduped, depth-limited."""
    seen, chain, stack = set(), [], [theme, "hicolor"]
    while stack:
        name = stack.pop(0)
        if name in seen or len(chain) > 8:
            continue
        seen.add(name)
        chain.append(name)
        for base in SEARCH_DIRS:
            index = Path(base) / name / "index.theme"
            if index.exists():
                try:
                    cp = configparser.ConfigParser()
                    cp.read(index, encoding="utf-8")
                    stack = [t.strip() for t in cp.get("Icon Theme", "Inherits", fallback="").split(",")] + stack
                except configparser.Error:
                    pass
                break
    return chain


class IconManager:
    """icon(key, size) -> PhotoImage. `key` is a canonical concept from NAMES (dir, config, ...);
    unknown keys render as 'file'. Theme resolution is lazy and cached per (key,size)."""

    def __init__(self):
        self.theme = "auto"
        self.cache = {}
        self._miss = set()

    def set_theme(self, name):
        if name != self.theme:
            self.theme = name or "auto"
            self.cache.clear()
            self._miss.clear()

    def _candidates(self):
        if self.theme in ("", "none"):
            return []
        if self.theme == "auto":
            return [t for t in ("Adwaita", "Yaru", "breeze", "Breeze", "Papirus", "Numix",
                                "gnome", "hicolor") if t in installed_themes()] or installed_themes()
        return [self.theme]

    def _find_file(self, fname, size):
        """The PNG in the theme chain closest to `size` (exact wins), or None."""
        best, best_delta = None, 10 ** 9
        for theme in self._candidates():
            for th in _chain(theme):
                root = None
                for base in SEARCH_DIRS:
                    if (Path(base) / th).is_dir():
                        root = Path(base) / th
                        break
                if root is None:
                    continue
                for d in root.iterdir() if root.is_dir() else []:
                    if not d.is_dir() or d.name in ("scalable", "symbolic"):
                        continue
                    if "x" in d.name:
                        try:
                            s = int(d.name.split("x")[0])
                        except ValueError:
                            continue
                        for cat in ("", "actions/", "places/", "mimetypes/", "status/", "apps/",
                                    "devices/", "emblems/", "categories/"):
                            hit = d / cat / f"{fname}.png"
                            if hit.exists():
                                delta = abs(s - size)
                                if delta < best_delta:
                                    best, best_delta = hit, delta
                                    if delta == 0:
                                        return best
        return best

    def resolve(self, key, size=16):
        """Agnostic half: a filesystem Path to the best PNG for a canonical key, or None.
        Any renderer (web serving the file, ncurses ignoring icons, Tk drawing PhotoImage)
        starts here."""
        for fname in NAMES.get(key, NAMES["file"]):
            found = self._find_file(fname, size)
            if found:
                return found
        return None

    def get(self, key, size=16):
        im = self.cache.get((key, size))
        if im is not None:
            return im
        found = self.resolve(key, size)
        if found is not None:
            try:
                im = tk.PhotoImage(name=f"mti_{found.stem}_{size}", file=str(found))
                if im.width() < size or im.height() < size:
                    raise tk.TclError("icon smaller than requested")
                k = im.width() // size
                if k > 1:
                    im = im.subsample(k, k)
                self.cache[(key, size)] = im
                return im
            except tk.TclError:
                pass
        im = _shape(key if key in _SHAPES else "file", size)
        self.cache[(key, size)] = im
        return im


ICONS = IconManager()


def icon(key, size=16):
    return ICONS.get(key, size)


def icon_for(row):
    if row["kind"] == "dir":
        return "dir"
    return _EXT.get("." + row["name"].rsplit(".", 1)[-1].lower() if "." in row["name"] else "", "file")
