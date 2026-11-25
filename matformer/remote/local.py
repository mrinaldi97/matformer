# matformer/remote/local.py
"""Use every server command directly from python, same process, no socket:

    from matformer.remote.local import LocalServer
    srv = LocalServer()
    srv.call("fs.list", {"path": "configs"})            # prints nothing, returns via callback
    rows = srv.result("runs.list")

It duck-types remote.net.Server (same name/up/call/push shape), so any GUI page, script or test can
run against a real in-process registry without a network — and against a remote one unchanged, just
swapping this class for remote.net.Server. Event pushes (token, metric, job, task_state) land on
the same events queue a UI would poll. Handlers run in the calling thread: keep that in mind for
long operations (use threading.Thread around call() like the async server does)."""
import queue
import threading


class _Conn:
    """The minimal Conn surface the command handlers use: push() and stop."""

    def __init__(self, local):
        self.local = local
        self.stop = threading.Event()

    def push(self, event, data):
        self.local.events.put((self.local.name, event, data))


class LocalServer:
    name, up = "local", True

    def __init__(self, events=None, trace=None):
        self.events = events if events is not None else queue.Queue()
        self.trace = trace
        self._conn = _Conn(self)
        import matformer.server.config_cmd    # noqa: F401  register every command module
        import matformer.server.fs            # noqa: F401
        import matformer.server.jobs          # noqa: F401
        import matformer.server.mdat_host     # noqa: F401
        import matformer.server.runs          # noqa: F401
        import matformer.server.slurm_cmd     # noqa: F401
        import matformer.server.sysinfo       # noqa: F401
        import matformer.server.tasks_host    # noqa: F401

    def call(self, method, params=None, cb=None):
        from matformer.server.core import RPC
        if self.trace is not None:
            self.trace(self.name, f"→ {method} {params or {}}")
        try:
            result = RPC[method](self._conn, params or {})
        except Exception as problem:
            # same contract as the socket client: the caller's callback learns about the failure,
            # so a page never waits for an answer that will not come
            if self.trace is not None:
                self.trace(self.name, f"✗ {type(problem).__name__}: {problem}")
            self.events.put((self.name, "error", f"{type(problem).__name__}: {problem}"))
            if cb:
                cb(("error", f"{type(problem).__name__}: {problem}"))
            return
        if self.trace is not None:
            self.trace(self.name, f"← {method} {str(result)[:400]}")
        if cb:
            cb(result)

    def result(self, method, params=None):
        """Synchronous convenience: the answer, or raise."""
        from matformer.server.core import RPC
        return RPC[method](self._conn, params or {})

    def stop(self, *_):
        self._conn.stop.set()
