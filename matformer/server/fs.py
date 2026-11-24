# matformer/server/fs.py
"""Real filesystem commands, jailed to the roots declared in server.toml (a datasets folder, the
repo, a checkpoints dir... the user may even grant "/"). Plus the favourites lists (fav_models /
fav_datasets), kept in one small JSON file: few lines, human-editable, no database."""
import json
import os
import shutil
import time
from pathlib import Path

from matformer.server import config
from matformer.server.core import rpc
from matformer.utils import matformer_assets

MAX_READ = 2_000_000          # bytes a config the GUI may open


def resolve(path: str) -> Path:
    """Absolute, symlink-free path; refuses anything outside the allowed roots.
    MATFORMER_ASSETS tokens ('$DATA_ASSETS/IRONITA/train.jsonl') are expanded first, so the GUI can
    address the shared folders without knowing where they live on this machine.
    A path that starts with a root name ('/checkpoints/...') is matched against the roots."""
    roots = [Path(r) for r in config()["roots"]]
    candidate = Path(matformer_assets.expand(path))
    candidate = (candidate if candidate.is_absolute() else Path.cwd() / candidate).resolve()
    for root in roots:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise PermissionError(f"'{path}' is outside the allowed roots: {[str(r) for r in roots]}")


@rpc("fs.roots")
def fs_roots(conn, p):
    """The folders this server allows the GUI to explore."""
    return [{"path": r, "name": Path(r).name or r} for r in config()["roots"]]


@rpc("fs.list")
def fs_list(conn, p):
    """Rows for one folder: [{name, kind: dir|file, size, mtime}], dirs first."""
    node = resolve(p.get("path", ""))
    rows = []
    for entry in os.scandir(node):
        try:
            stat = entry.stat()
            rows.append({"name": entry.name, "kind": "dir" if entry.is_dir() else "file",
                         "size": 0 if entry.is_dir() else stat.st_size,
                         "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))})
        except OSError:
            continue
    return sorted(rows, key=lambda r: (r["kind"] != "dir", r["name"].lower()))


@rpc("fs.read")
def fs_read(conn, p):
    """Text content of a file (for the config editor)."""
    file = resolve(p["path"])
    if file.stat().st_size > MAX_READ:
        raise ValueError(f"{file.name} is too large to open in the editor")
    return {"path": str(file), "text": file.read_text("utf-8", errors="replace")}


@rpc("fs.write")
def fs_write(conn, p):
    """Save text over an existing-or-new file inside the roots."""
    file = resolve(p["path"])
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(p["text"], "utf-8")
    return {"msg": f"saved {file}"}


@rpc("fs.rename")
def fs_rename(conn, p):
    src, dst = resolve(p["path"]), resolve(p["to"])
    if dst.exists():
        raise ValueError(f"{dst} already exists")
    os.rename(src, dst)
    return {"msg": f"renamed to {dst.name}"}


@rpc("fs.delete")
def fs_delete(conn, p):
    target = resolve(p["path"])
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    return {"msg": f"deleted {target.name}"}


@rpc("fs.mkdir")
def fs_mkdir(conn, p):
    target = resolve(p["path"])
    target.mkdir(parents=True, exist_ok=True)
    return {"msg": f"created {target}"}


# ── favourites ─────────────────────────────────────────────────────────────────────────────────
KINDS = ("models", "datasets")


def _favs():
    file = Path(config()["fav_file"]).expanduser()
    data = json.loads(file.read_text("utf-8")) if file.exists() else {}
    return file, {kind: data.get(kind, []) for kind in KINDS}


def _save_favs(file, favs):
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(favs, indent=1), "utf-8")


@rpc("fav.list")
def fav_list(conn, p):
    """{'models': [...], 'datasets': [...]} — local paths or external ids (hf://...)."""
    _, favs = _favs()
    return favs


@rpc("fav.add")
def fav_add(conn, p):
    """fav.add(kind, item): remember a model/dataset (path or URI)."""
    file, favs = _favs()
    if p["kind"] not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if p["item"] not in favs[p["kind"]]:
        favs[p["kind"]].append(p["item"])
        _save_favs(file, favs)
    return favs


@rpc("fav.delete")
def fav_delete(conn, p):
    file, favs = _favs()
    if p["item"] in favs.get(p["kind"], []):
        favs[p["kind"]].remove(p["item"])
        _save_favs(file, favs)
    return favs
