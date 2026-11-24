# matformer/remote/net.py
"""Renderer-agnostic transport: one Server object per backend ("host:port"), auto-reconnecting;
every answer arrives through a callback, every push through the event queue the UI loop polls.
Nothing here touches Tk — an ncurses or web front-end would use this exact class."""
import json
import queue
import socket
import threading


class Server(threading.Thread):
    def __init__(self, name, url, events, key=None, trace=None):
        super().__init__(name=f"matformer-client-{name}", daemon=True)
        self.name, self.url, self.key = name, url, key
        self.host, self.port = url.split(":")[0], int(url.rsplit(":", 1)[1])
        self.events, self.up, self.pending, self.stop = events, False, {}, threading.Event()
        self.paused = threading.Event()
        self.queued, self.io = [], threading.Lock()
        self.n, self.sock, self.fh, self.lock = 0, None, None, threading.Lock()
        self.trace = trace                       # called with "method params" / "<- answer" lines

    def run(self):
        while not self.stop.is_set():
            while self.paused.is_set() and not self.stop.is_set():
                self.stop.wait(0.5)
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=10)
                self.sock.settimeout(None)
                self.fh = self.sock.makefile("rwb")
                if self.key:
                    self.fh.write((json.dumps({"auth": self.key}) + "\n").encode())
                    self.fh.flush()
                self._set(True)
                while True:
                    line = self.fh.readline()
                    if not line:
                        break
                    try:
                        self._rx(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            except (OSError, ValueError):
                pass
            self._set(False)
            self._dead_sockets()
            self.stop.wait(3)                       # retry soon

    def set_address(self, host, port):
        """Retarget the connection (used once by the SSH bootstrap thread)."""
        self.host, self.port, self.url = host, int(port), f"{host}:{int(port)}"
        if self.sock:
            try:
                self.sock.close()                   # forces the reader loop to redial
            except OSError:
                pass

    def disconnect(self):
        """Stop dialing and drop the current socket (user pressed Disconnect on the pill)."""
        self.paused.set()
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass

    def connect(self):
        self.paused.clear()
        if not self.is_alive():
            self.start()

    def shutdown(self):
        self.stop.set()
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass

    def call(self, method, params=None, cb=None):
        """Fire a request; cb(result) runs on the main thread (via the event queue). Never blocks.
        While the connection is still coming up (boot, reconnect) calls are queued and flushed
        on connect — pages must not care about connection timing."""
        if not self.up:
            self.queued.append((method, params, cb))
            del self.queued[:-40]
            return
        with self.lock:
            self.n += 1
            rid = self.n
        if cb:
            self.pending[rid] = cb
        self._trace(f"→ {method} {json.dumps(params or {}, default=str)[:400]}")
        try:
            with self.io:
                self.fh.write((json.dumps({"id": rid, "method": method, "params": params or {}})
                               + "\n").encode())
                self.fh.flush()
        except (OSError, ValueError):
            self.pending.pop(rid, None)
            self.events.put((self.name, "error", f"{method}: send failed (connection down)"))

    def _trace(self, line):
        if self.trace is not None:
            self.trace(self.name, line)

    def _rx(self, msg):
        if "event" in msg:
            self._trace(f"· {msg['event']} {json.dumps(msg.get('data'), default=str)[:200]}")
            self.events.put((self.name, msg["event"], msg["data"]))
        elif "error" in msg:
            # the caller's callback still runs: a page that asked must be allowed to un-block,
            # otherwise a failed request leaves it waiting forever (the stuck Stop button).
            self._trace(f"✗ {msg['id']} {msg['error']}")
            if msg["id"] in self.pending:
                self.events.put((self.name, "__result__", (msg["id"], ("error", msg["error"]))))
            self.events.put((self.name, "error", f"{msg['error']}"))
        else:
            self._trace(f"← {msg['id']} {json.dumps(msg.get('result'), default=str)[:400]}")
            self.events.put((self.name, "__result__", (msg["id"], msg.get("result"))))

    def _set(self, up):
        self.up = up
        self.events.put((self.name, "__state__", up))
        if up and self.queued:
            backlog, self.queued = self.queued, []
            for method, params, cb in backlog:
                self.call(method, params, cb)

    def _dead_sockets(self):
        """A socket that died takes its answers with it: every waiting callback is called with the
        error, so no page is left holding a request that will never come back. Calls that were
        still buffered (never sent) stay buffered: they go out on the next connection."""
        pending, self.pending = self.pending, {}
        if pending:
            self.events.put((self.name, "__dead__", pending))
