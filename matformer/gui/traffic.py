# matformer/gui/traffic.py
"""Console: what actually crossed the wire. Every command sent to a server, every answer, error and
event that came back, with the GUI's own tracebacks — the place to look when something fails and
the status bar said only one line.

Off by default and bounded: a training tail or a dataset browse moves a lot of JSON, and a log you
cannot silence is a log you turn off anyway. Rendered in batches, only while it is visible."""
import time
import tkinter as tk
from tkinter import filedialog, ttk

from matformer.gui.widgets import Console

MAX_LINES = 3000


class Traffic(ttk.Frame):
    """A console of server traffic: record toggle, bounded buffer, export."""

    def __init__(self, app, parent):
        super().__init__(parent)
        self.app, self.on, self.buffer = app, False, []
        head = ttk.Frame(self)
        head.pack(fill="x", pady=(0, 3))
        self.switch = ttk.Checkbutton(head, text="Record", command=lambda: self.set(self.switch.instate(["selected"])))
        self.switch.pack(side="left")
        self.count = ttk.Label(head, text="0 lines", style="Muted.TLabel")
        self.count.pack(side="left", padx=8)
        ttk.Button(head, text="Clear", style="Ghost.TButton", width=6, command=self.clear).pack(side="left")
        ttk.Button(head, text="Export…", style="Ghost.TButton", command=self.export).pack(side="left", padx=4)
        send = ttk.Frame(self)
        send.pack(fill="x", pady=(3, 3))
        ttk.Label(send, text="RPC").pack(side="left")
        self.method = ttk.Entry(send, width=24)
        self.method.pack(side="left", padx=(4, 2))
        self.method.insert(0, "task.state")
        self.params = ttk.Entry(send)
        self.params.pack(side="left", fill="x", expand=True, padx=2)
        self.params.insert(0, "{}")
        ttk.Button(send, text="Send", style="Accent.TButton", width=6,
                   command=self.send).pack(side="left", padx=2)
        self.view = Console(app, self)
        self.view.pack(fill="both", expand=True)

    @property
    def recording(self):
        return self.on

    def set(self, on):
        """The one way to turn recording on or off, from this widget or from the Local page."""
        self.on = bool(on)
        self.app.traffic_recording = self.on
        if self.switch.instate(["selected"]) != self.on:
            self.switch.state(["selected"] if self.on else ["!selected"])
        self.app.toast("console recording" if self.on else "console idle")

    def toggle(self):
        self.set(self.on)

    def record(self, server, line):
        """Called from the client threads: never touches Tk, only appends."""
        if not self.recording:
            return
        self.buffer.append(f"{time.strftime('%H:%M:%S')}  {server}  {line}")
        del self.buffer[:-MAX_LINES]

    def flush(self):
        """Called by the UI pump: moves what the threads appended into the text, in one go."""
        if not self.buffer:
            return
        lines, self.buffer = self.buffer, []
        for line in lines:
            self.view.put(line + "\n", "err" if " ✗ " in line else None)
        self.count.configure(text=f"{len(self.view.text.get('1.0', 'end-1c').splitlines())} lines")

    def clear(self):
        self.buffer.clear()
        self.view.clear()

    def export(self):
        path = filedialog.asksaveasfilename(defaultextension=".log", initialfile="console.log",
                                            filetypes=[("Text", "*.log"), ("Text", "*.txt")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.view.text.get("1.0", "end"))
        self.app.toast(f"console exported to {path}")

    def send(self):
        import json
        method = self.method.get().strip()
        if not method:
            return
        try:
            params = json.loads(self.params.get() or "{}")
        except json.JSONDecodeError as problem:
            self.view.put(f"{method}: params must be JSON: {problem}\n", "err")
            return
        self.view.put(f"→ {method} {json.dumps(params)}\n", "meta")

        def done(result):
            import json as _json
            if isinstance(result, tuple):
                self.view.put(f"✗ {result[1]}\n", "err")
            else:
                self.view.put(_json.dumps(result, indent=1, default=str) + "\n")
        self.app.server.call(method, params, done)

    def repaint(self, p):
        pass
