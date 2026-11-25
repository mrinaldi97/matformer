"""A synchronous command line for the Matformer NDJSON-RPC server.

    python -m matformer.remote.cli rpc.list
    python -m matformer.remote.cli task.state
    python -m matformer.remote.cli task.run --key summary/model/parameters
    python -m matformer.remote.cli train.heartbeat '{"id":"run-120000"}' --out /tmp/hb.json

The GUI uses the asynchronous :class:`matformer.remote.net.Server`; this module is the small,
blocking sibling for scripts and agents. One method per invocation, one JSON answer, optional
dotted/slash key selection so a large task summary can be inspected without dumping all of it.

Params are a JSON object (or ``@file.json``). ``--key`` walks a slash-separated path through the
result; repeat it to pick several values. ``--out`` writes the complete answer to a file, which is
the useful mode for the very large static/activation summaries.
"""
import argparse
import json
import socket
import sys
from pathlib import Path


DEFAULT_URL = "127.0.0.1:8765"


def request(method, params=None, url=DEFAULT_URL, key=None, timeout=300):
    """One blocking request. Raises RuntimeError with the server's message on error."""
    host, _, port = url.partition(":")
    sock = socket.create_connection((host or "127.0.0.1", int(port or 8765)), timeout=10)
    try:
        sock.settimeout(timeout)
        stream = sock.makefile("rwb")
        if key:
            stream.write((json.dumps({"auth": key}) + "\n").encode())
        stream.write((json.dumps({"id": 1, "method": method,
                                  "params": params or {}}) + "\n").encode())
        stream.flush()
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError("the server closed the connection")
            message = json.loads(line)
            if "event" in message:          # pushes do not answer this request
                continue
            if "error" in message:
                raise RuntimeError(message["error"])
            return message.get("result")
    finally:
        sock.close()


def _walk(result, path):
    """Follow a slash-separated path; a numeric segment indexes a list."""
    value, parts = result, path.split("/")
    for index, part in enumerate(parts):
        if not part:
            continue
        remaining = "/".join(parts[index:])
        if isinstance(value, dict) and remaining in value:
            return value[remaining]          # a flat metric key may itself contain slashes
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict):
            if part not in value:
                raise KeyError(f"'{path}': no key '{part}' (available: {', '.join(value)[:300]})")
            value = value[part]
        else:
            raise KeyError(f"'{path}': cannot descend into {type(value).__name__}")
    return value


def parser():
    p = argparse.ArgumentParser(prog="python -m matformer.remote.cli",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method", help="RPC method, e.g. rpc.list or task.run")
    p.add_argument("params", nargs="?", default="{}",
                   help="JSON object, or @file.json (default: {})")
    p.add_argument("--url", default=DEFAULT_URL, help="host:port (default 127.0.0.1:8765)")
    p.add_argument("--key", default=None,
                   help="server API key, if the server requires authentication")
    p.add_argument("--timeout", type=float, default=300, help="seconds to wait for the answer")
    p.add_argument("--out", metavar="FILE", help="write the complete JSON answer here")
    p.add_argument("--select", action="append", default=[], metavar="PATH",
                   help="print only this slash-separated key path (repeatable)")
    p.add_argument("--compact", action="store_true", help="one line instead of indented JSON")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    raw = args.params
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text()
    params = json.loads(raw)
    try:
        result = request(args.method, params, url=args.url, key=args.key,
                         timeout=args.timeout)
    except (OSError, RuntimeError, json.JSONDecodeError) as problem:
        print(f"{args.method}: {problem}", file=sys.stderr)
        return 1
    try:
        if args.out:
            Path(args.out).expanduser().write_text(json.dumps(result, indent=1, default=str))
        if args.select:
            for path in args.select:
                try:
                    value = _walk(result, path)
                except (KeyError, IndexError, ValueError) as problem:
                    print(f"{path}: {problem}", file=sys.stderr)
                    return 1
                print(json.dumps(value, indent=None if args.compact else 1, default=str))
            return 0
        print(json.dumps(result, indent=None if args.compact else 1, default=str))
    except BrokenPipeError:
        return 0                    # a pipe such as `| head` closed early: not an error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
