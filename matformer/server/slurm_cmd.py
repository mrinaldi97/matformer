# matformer/server/slurm_cmd.py
"""The SLURM interpreter as RPC. Like MDAT, the server owns no cluster logic: it publishes
`SlurmCommands` — `slurm.methods` describes the operations so the interface generates itself,
`slurm.call` runs one, `slurm.line` accepts what a person would type in the terminal — and adds a
watchdog that turns a change of state of a job into a `slurm_job` broadcast, which is what lets a
GUI on another machine follow a training it does not own."""
import inspect
import threading
import time

from matformer.server.core import broadcast, rpc

WATCHER = [None]
SEEN = {}


def commands():
    from matformer.slurm import SlurmCommands
    return SlurmCommands()


@rpc("slurm.methods")
def slurm_methods(conn, p):
    """[{name, doc, args}] for every cluster operation, introspected from the interpreter."""
    out = []
    for name in dir(commands()):
        if name.startswith("_"):
            continue
        fn = getattr(type(commands()), name, None)
        if not inspect.isfunction(fn):
            continue
        args = []
        for parameter in inspect.signature(fn).parameters.values():
            if parameter.name == "self":
                continue
            annotation = str(parameter.annotation)
            kind = "bool" if annotation == "<class 'bool'>" else \
                "int" if annotation == "<class 'int'>" else "str"
            args.append({"name": parameter.name, "kind": kind,
                         "default": None if parameter.default is inspect.Parameter.empty
                         else parameter.default})
        out.append({"name": name, "args": args,
                    "doc": (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""})
    return out


@rpc("slurm.call")
def slurm_call(conn, p):
    """One operation, programmatically: {cmd: 'nodes', args: {partition: '…'}}."""
    fn = getattr(commands(), p["cmd"])
    return fn(**{k: v for k, v in (p.get("args") or {}).items() if v is not None})


@rpc("slurm.line")
def slurm_line(conn, p):
    """Type it as you would in a terminal: 'queue', 'nodes epito', 'save 4712'. Returns text for
    tables, structures for everything else."""
    from matformer.slurm import line
    result = line(p["line"])
    return result if isinstance(result, str) else {"result": result}


@rpc("slurm.fields")
def slurm_fields(conn, p):
    """Where a training can run, as fields: the dialog that submits a job is generated from here."""
    from matformer.slurm import SlurmCommands
    return {"fields": SlurmCommands().allocation_fields(),
            "cluster": commands().cluster(),
            "partitions": commands().partitions().asdict(),
            "venvs": commands().venvs().asdict()}


@rpc("slurm.watch")
def slurm_watch(conn, p):
    """Replay the cluster's jobs to this connection and keep pushing changes."""
    for row in _rows():
        conn.push("slurm_job", row)
    start_watcher()
    return {"msg": "watching cluster jobs"}


def _rows() -> list:
    """The interpreter's job table as plain dicts: one shape for the replay, the broadcast and the
    dock, so a client never has to know which of the three a row came from."""
    from matformer.slurm import SlurmCommands
    return SlurmCommands().jobs(fast=True).asdict()["rows"]


def start_watcher():
    if WATCHER[0] is None or not WATCHER[0].is_alive():
        WATCHER[0] = threading.Thread(target=_watch, name="slurm-watch", daemon=True)
        WATCHER[0].start()


def _watch():
    """Poll the controller, not the training: a job that changes state, step or checkpoint is worth
    a broadcast, and nothing else is."""
    global SEEN
    while True:
        try:
            for row in _rows():
                signature = (row["state"], row["step"], row["saved"], row["left"])
                if SEEN.get(row["job"]) != signature:
                    SEEN[row["job"]] = signature
                    broadcast("slurm_job", row)
        except Exception:
            pass
        time.sleep(10)
