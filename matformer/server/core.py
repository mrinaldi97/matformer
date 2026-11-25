# matformer/server/core.py
"""The protocol core: a dotted-name command registry plus one asyncio TCP server speaking
line-delimited JSON. Heavy commands run in worker threads so the loop never blocks; those
threads push events back with conn.push (thread-safe).

Authentication is per-connection, zero per-call overhead: when api_key is configured the client's
first line must be {"auth": "<key>"}; until validated every other message is rejected. Without a
key (the localhost default) the protocol is exactly as plain as it looks."""
import asyncio
import inspect
import json
import traceback
from collections import deque
from datetime import datetime

from matformer.utils.table import Table

RPC = {}
SERVER = [None]           # the one Server instance, for worker threads that just want to broadcast
FAILURES = deque(maxlen=100)   # every command that raised, with the traceback: sys.errors


def _encode(obj):
    """json default=: Tables cross the wire as structured rows (the GUI draws them, the TUI prints
    the very same object with str()); anything else falls back to its text form."""
    return obj.asdict() if isinstance(obj, Table) else str(obj)


def rpc(name):
    """Register a command handler: f(conn, params) -> result (sync handlers run in a thread)."""
    def decorator(f):
        RPC[name] = f
        return f
    return decorator


def broadcast(event, data):
    """Push an event to every connected GUI, from any thread."""
    if SERVER[0] is not None:
        SERVER[0].broadcast_threadsafe(event, data)


class Conn:
    """One connected (authenticated) GUI."""

    def __init__(self, server, writer):
        self.server, self.writer, self.stop = server, writer, asyncio.Event()
        self.authed = not server.cfg.get("api_key")

    async def emit(self, event, data):
        try:
            self.writer.write((json.dumps({"event": event, "data": data}, default=_encode) + "\n").encode())
            await self.writer.drain()
        except (ConnectionError, OSError):
            self.server.conns.discard(self)

    def push(self, event, data):
        """emit() from a worker thread."""
        asyncio.run_coroutine_threadsafe(self.emit(event, data), self.server.loop)


class Server:
    def __init__(self, cfg):
        self.cfg, self.conns, self.loop = cfg, set(), None

    async def broadcast(self, event, data):
        for conn in list(self.conns):
            await conn.emit(event, data)

    def broadcast_threadsafe(self, event, data):
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.broadcast(event, data), self.loop)

    async def dispatch(self, conn, msg):
        if "auth" in msg:
            conn.authed = bool(self.cfg.get("api_key")) and msg["auth"] == self.cfg["api_key"]
            await conn.emit("auth", "ok" if conn.authed else "rejected")
            return
        if not conn.authed:
            return await conn.emit("error", "auth required: this server needs an api key")
        try:
            handler = RPC[msg["method"]]
            if inspect.iscoroutinefunction(handler):
                result = await handler(conn, msg.get("params") or {})
            else:
                result = await asyncio.to_thread(handler, conn, msg.get("params") or {})
            out = {"id": msg.get("id"), "result": result}
        except Exception as problem:
            FAILURES.append({"at": datetime.now().isoformat(timespec="seconds"),
                             "method": msg.get("method"),
                             "error": f"{type(problem).__name__}: {problem}",
                             "traceback": traceback.format_exc()})
            out = {"id": msg.get("id"), "error": f"{type(problem).__name__}: {problem}"}
        try:
            conn.writer.write((json.dumps(out, default=_encode) + "\n").encode())
            await conn.writer.drain()
        except (ConnectionError, OSError):
            self.conns.discard(conn)

    async def _handle(self, reader, writer):
        conn = Conn(self, writer)
        self.conns.add(conn)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                asyncio.ensure_future(self.dispatch(conn, json.loads(line)))
        except (json.JSONDecodeError, ConnectionError, OSError):
            pass
        finally:
            self.conns.discard(conn)
            conn.stop.set()
            writer.close()

    async def serving(self):
        """Bind the port, importing the command modules so they register themselves."""
        self.loop = asyncio.get_running_loop()
        import matformer.server.collect_cmd    # noqa: F401
        import matformer.server.config_cmd     # noqa: F401
        import matformer.server.fs            # noqa: F401
        import matformer.server.jobs          # noqa: F401
        import matformer.server.mdat_host     # noqa: F401
        import matformer.server.memory        # noqa: F401
        import matformer.server.plugin_cmd    # noqa: F401
        import matformer.server.runs          # noqa: F401
        import matformer.server.slurm_cmd      # noqa: F401
        import matformer.server.store_cmd      # noqa: F401
        import matformer.server.sysinfo       # noqa: F401
        import matformer.server.task_jobs     # noqa: F401
        import matformer.server.tasks_host    # noqa: F401
        import matformer.server.vector_cmd    # noqa: F401
        import matformer.server.wizard_cmd    # noqa: F401
        self._tcp = await asyncio.start_server(
            self._handle, self.cfg["host"], self.cfg["port"],
            limit=16 * 1024 * 1024)   # one JSON line may carry a whole config or a data file
        SERVER[0] = self        # only a server that actually bound becomes the broadcast target

    async def shutdown(self):
        self._tcp.close()
        await self._tcp.wait_closed()


@rpc("rpc.list")
def rpc_list(conn, p):
    """Every command the server knows, with its first docstring line: makes the GUI introspective."""
    return [{"name": name, "doc": (f.__doc__ or "").strip().splitlines()[0] if f.__doc__ else ""}
            for name, f in sorted(RPC.items())]


@rpc("stop")
def stop(conn, p):
    """Ask whatever this connection is running (generation, ...) to stop at the next safe point."""
    conn.stop.set()


@rpc("rpc.failures")
def rpc_failures(conn, p):
    """Commands that raised since the server started, newest first, traceback included: the one
    place a UI looks when an answer says only `RuntimeError: …`. `clear` empties it."""
    if p.get("clear"):
        FAILURES.clear()
        return {"msg": "failures cleared"}
    rows = [{"at": f["at"], "method": f["method"], "error": f["error"]} for f in reversed(FAILURES)]
    if p.get("index") is not None:
        failure = list(reversed(FAILURES))[int(p["index"])]
        return {"text": failure["traceback"]}
    return Table(["at", "method", "error"], rows)
