# matformer/server/mdat_host.py
"""MDAT integration: the server exposes MDatCommands — the very same UI-agnostic facade the cmd2
TUI sits on — as RPC. `mdat.methods` describes the facade (name, doc, arguments) so the GUI builds
the whole dataset panel without knowing what MDAT is; `mdat.line` runs a raw TUI line through the
real MDatShell parsers, so the GUI command bar and the terminal TUI accept exactly the same syntax."""
import inspect
import threading
import typing

from matformer.server.core import rpc

LOCK = threading.Lock()
HOST = {"cmds": None, "shell": None}


def commands():
    if HOST["cmds"] is None:
        from matformer.data.mdat.tui import MDatCommands
        HOST["cmds"] = MDatCommands()
    return HOST["cmds"]


def shell():
    if HOST["shell"] is None:
        from matformer.data.mdat.tui import MDatShell
        HOST["shell"] = MDatShell()
        HOST["shell"].cmds = commands()
    return HOST["shell"]


SKIP = ()      # every facade operation is reachable; the destructive ones ask inside their dialog


@rpc("mdat.methods")
def mdat_methods(conn, p):
    """[{name, doc, args: [{name, kind, default}]}] for every MDatCommands operation.
    Introspected on the class: bound methods are not functions, so an instance sweep would find
    nothing and the GUI would silently have no operations at all."""
    out = []
    cls = type(commands())
    for name in dir(cls):
        if name.startswith("_") or name in SKIP:
            continue
        fn = getattr(cls, name)
        if not inspect.isfunction(fn):
            continue
        args = []
        try:
            # `from __future__ import annotations` leaves annotations as strings;
            # resolve them so Literal choices and int fields reach the GUI.
            hints = typing.get_type_hints(fn)
        except Exception:
            hints = {}
        for parameter in inspect.signature(fn).parameters.values():
            if parameter.name == "self":
                continue
            annotation = hints.get(parameter.name, parameter.annotation)
            args_types = typing.get_args(annotation)
            kind, options = "str", []
            if typing.get_origin(annotation) is typing.Literal:
                kind, options = "choice", list(args_types)
            elif annotation is bool or isinstance(parameter.default, bool):
                kind = "bool"
            elif (annotation is int or int in args_types
                  or (isinstance(parameter.default, int) and not isinstance(parameter.default, bool))):
                kind = "int"
            args.append({"name": parameter.name, "kind": kind, "options": options,
                         "default": None if parameter.default is inspect.Parameter.empty
                         else parameter.default})
        out.append({"name": name, "doc": (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "",
                    "args": args})
    return out


@rpc("mdat.call")
def mdat_call(conn, p):
    """Run one operation programmatically: {cmd: 'peek', args: {n: 3}}. Tables cross the wire as
    structured rows (the GUI draws them); strings arrive as prose — one shape for both worlds."""
    fn = getattr(commands(), p["cmd"])
    with LOCK:
        return fn(**{k: v for k, v in (p.get("args") or {}).items() if v is not None})


@rpc("mdat.start")
def mdat_start(conn, p):
    """Queue a long MDAT operation — ingest, pretokenize, distributed precompute, export — as a job:
    its own process, its own log, abortable, visible in the dock. Same facade calls as `mdat.call`,
    so nothing is duplicated for the slow path."""
    from matformer.server.jobs import enqueue
    job_id = enqueue("mdat", p.get("name") or (p["commands"][0]["cmd"] if p.get("commands") else "mdat"),
                     p.get("project", "mdat"),
                     {"container": p["container"], "readonly": p.get("readonly", False),
                      "commands": p["commands"]},
                     "matformer.data.mdat.tui:execute")
    return {"run": job_id}


@rpc("mdat.line")
def mdat_line(conn, p):
    """Type what you would type in the TUI: 'peek --n 3', 'batch-import /data/x --key text'."""
    sh = shell()
    lines = []
    sh.poutput = lambda *a, **k: lines.append(str(a[0]) if a else "")   # capture instead of print
    command, _, rest = p["line"].strip().partition(" ")
    handler = getattr(sh, "do_" + command.replace("-", "_"), None)
    if handler is None:
        raise ValueError(f"no such mdat command: {command}")
    try:
        with LOCK:
            handler(rest)
    except ValueError as usage:                     # _ShlexArgumentParser raises instead of exiting
        return f"ERROR (usage): {usage}"
    return "\n".join(lines)
