# matformer/server/__main__.py
"""The Matformer server, standalone:

    python -m matformer.server                     # host/port from server.toml
    python -m matformer.server --port 8765         # overrides
    python -m matformer.server --root /data        # extra explore root (repeatable)

This import path is deliberately light: the registry (and torch) load lazily, only when a command
needs them, so starting the server is instant."""
import argparse
import asyncio
import os

from matformer.server import _amain, config


def main(argv=None):
    argp = argparse.ArgumentParser(prog="python -m matformer.server")
    argp.add_argument("--host", default=None)
    argp.add_argument("--port", type=int, default=None)
    argp.add_argument("--name", default=None)
    argp.add_argument("--root", action="append", default=None,
                      help="extra allowed filesystem root (repeatable)")
    args = argp.parse_args(argv)
    cfg = config()
    cfg.update({k: v for k, v in (("host", args.host), ("port", args.port), ("name", args.name))
                if v is not None})
    os.environ["MATFORMER_SERVER"] = "1"
    if args.root:
        cfg["roots"] = list(dict.fromkeys(cfg["roots"] + [os.path.abspath(r) for r in args.root]))
    asyncio.run(_amain(cfg))


if __name__ == "__main__":
    main()
