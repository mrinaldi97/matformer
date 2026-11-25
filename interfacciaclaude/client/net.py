"""The only module that knows about the network. One Server object per backend.

Wire protocol (JSON text frames over a WebSocket, identical for local and remote servers):
  GUI    -> {"id": 7, "method": "generate", "params": {...}}
  server -> {"id": 7, "result": ...}  |  {"id": 7, "error": "..."}      (answer to a call)
  server -> {"event": "token", "data": {...}}                          (push, any time)

Note on naming: the push signal is called ``pushed`` rather than ``event`` on purpose.
``QObject.event`` is a virtual method Qt calls internally for every dispatched event;
overriding it with a Signal makes Qt raise "TypeError: native Qt signal is not callable".
"""
import json

from qtpy.QtCore import QObject, QTimer, QUrl, Signal
from qtpy.QtNetwork import QAbstractSocket
from qtpy.QtWebSockets import QWebSocket


class Server(QObject):
    pushed = Signal(str, str, object)   # (server, name, payload) for every push; remote errors arrive as "error"
    state = Signal(str, bool)           # (server, is_up)

    def __init__(self, name, url):
        super().__init__()
        self.name, self.url, self.up, self.pending, self.n = name, url, False, {}, 0
        self.ws, self.timer = QWebSocket(), QTimer(self)
        self.ws.connected.connect(lambda: self._set(True))
        self.ws.disconnected.connect(lambda: self._set(False))
        self.ws.textMessageReceived.connect(self._rx)
        self.timer.timeout.connect(self._open)                       # auto-reconnect

    def start(self):
        self.timer.start(3000)
        self._open()

    def call(self, method, params=None, cb=None):
        """Fire a request; cb(result) runs when the answer arrives. Never blocks the GUI."""
        if not self.up:
            return self.pushed.emit(self.name, "error", f"{method}: not connected")
        self.n += 1
        if cb:
            self.pending[self.n] = cb
        self.ws.sendTextMessage(json.dumps({"id": self.n, "method": method, "params": params or {}}))

    def _open(self):
        if self.ws.state() == QAbstractSocket.SocketState.UnconnectedState:
            self.ws.open(QUrl(self.url))

    def _set(self, up):
        self.up = up
        self.pending.clear()
        self.state.emit(self.name, up)

    def _rx(self, text):
        m = json.loads(text)
        if "event" in m:
            self.pushed.emit(self.name, m["event"], m["data"])
        elif "error" in m:
            self.pending.pop(m["id"], None)
            self.pushed.emit(self.name, "error", m["error"])
        elif cb := self.pending.pop(m["id"], None):
            cb(m["result"])
