# matformer/server/__init__.py
"""Matformer always-on server: plain NDJSON-RPC over TCP, zero dependencies.

One JSON object per line, in both directions (the wire language is the one prototyped in
interfacciaclaude/, minus the websocket):

    GUI    -> {"id": 7, "method": "fs.list", "params": {...}}
    server -> {"id": 7, "result": ...} | {"id": 7, "error": "..."}
    server -> {"event": "metric", "data": {...}}          (push, any time)

It is started by matformer/matformer_registry.py, so it is simply *always there* whenever the
framework is imported. In production it listens on localhost only and reaches the outside world
through an SSH tunnel (`ssh -L 8765:localhost:8765 box`).

Environment: MATFORMER_SERVER=0 disables it, MATFORMER_SERVER_PORT overrides the port.
Configuration: server.toml next to this file (missing values keep their defaults).
"""
import asyncio
import os
import threading
import tomllib
from pathlib import Path

from matformer.utils import matformer_assets

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULTS = {"host": "127.0.0.1", "port": 8765, "name": "local", "api_key": "",
            "roots": [str(REPO)], "run_roots": [str(REPO / "checkpoints")],
            "fav_file": str(Path.home() / ".matformer" / "favourites.json")}
CONFIG, _started = None, threading.Lock()


def config() -> dict:
    """server.toml merged over the defaults; paths are taken relative to the repo root and may use
    MATFORMER_ASSETS tokens. The assets folder itself, when configured, is always added."""
    global CONFIG
    if CONFIG is None:
        cfg = dict(DEFAULTS)
        file = HERE / "server.toml"
        if file.exists():
            cfg.update(tomllib.loads(file.read_text("utf-8")))
        for key in ("roots", "run_roots"):
            entries = [matformer_assets.expand(p) for p in cfg[key]]
            cfg[key] = [str((REPO / p if not Path(p).is_absolute() else Path(p)).expanduser().resolve())
                        for p in entries]
        if os.environ.get("MATFORMER_SERVER_PORT"):
            cfg["port"] = int(os.environ["MATFORMER_SERVER_PORT"])
        if os.environ.get("MATFORMER_API_KEY"):
            cfg["api_key"] = os.environ["MATFORMER_API_KEY"]
        _add_assets(cfg)
        CONFIG = cfg
    return CONFIG


def _add_assets(cfg):
    """When MATFORMER_ASSETS is configured it becomes the first browsable root (so the GUI pickers
    open on it) and its run folders are scanned too. Silent when there is no assets folder.
    Resolved like every other root: fs.resolve() jails against symlink-free paths."""
    root = matformer_assets.assets_root()
    if root is None or not root.is_dir():
        return
    root = root.resolve()
    cfg["roots"] = [r for r in cfg["roots"] if r != str(root)]
    cfg["roots"].insert(0, str(root))            # first root == where the pickers open
    for area in ("logs", "checkpoints", "experiments"):
        folder = (root / area).resolve()
        if folder.is_dir() and str(folder) not in cfg["run_roots"]:
            cfg["run_roots"].append(str(folder))


def start_server():
    """Fire-and-forget: daemon thread, quiet on every failure (a sibling process already
    serving on the port is the normal reason)."""
    if os.environ.get("MATFORMER_SERVER", "1").strip().lower() in ("0", "false", "no"):
        return
    with _started:
        thread = threading.Thread(target=_serve, name="matformer-server", daemon=True)
        thread.start()


def _serve():
    try:
        cfg = config()
        asyncio.run(_amain(cfg))
    except OSError:
        pass                                   # port taken: another matformer process is the server
    except Exception as problem:               # a broken server must never break the framework
        print(f"matformer-server: not started ({type(problem).__name__}: {problem})")


async def _amain(cfg):
    from matformer.server.core import Server
    server = Server(cfg)
    await server.serving()
    print(f"matformer-server [{cfg['name']}]: tcp://{cfg['host']}:{cfg['port']}", flush=True)
    await asyncio.get_running_loop().create_future()   # run forever
