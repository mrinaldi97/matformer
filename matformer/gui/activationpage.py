# matformer/gui/activationpage.py
"""Activations: the activation store the model owns, read through its RPC (store.*).

The store is the plugin the model already mounts: mounting it is an internals action, this page
only looks at it. One address at a time: its statistics, a histogram of its values and a PCA of
its per-sequence means, all drawn with Tk so the laptop never installs matplotlib to read a store."""
import tkinter as tk
from tkinter import ttk

from matformer.gui.plotwidgets import MiniPlot
from matformer.gui.storebar import StoreBar
from matformer.gui.widgets import Page, TableList


class ActivationPage(Page):
    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Stores · activations", "addresses, distributions and PCA of an activation store")
        self.bar = StoreBar(self.body, app, on_open=self._open_spec,
                            on_refresh=lambda spec: self.refresh(), databank="activations")
        self.bar.pack(fill="x", pady=(0, 4))
        controls = ttk.Frame(self.body)
        controls.pack(fill="x", pady=(0, 4))
        ttk.Button(controls, text="Switch databank", style="Ghost.TButton",
                   command=self.switch).pack(side="left")
        ttk.Button(controls, text="Vectors →", style="Ghost.TButton",
                   command=lambda: self.app.select_page("steering")).pack(side="right")
        ttk.Label(controls, text="  Start").pack(side="left")
        self.start = ttk.Entry(controls, width=7)
        self.start.insert(0, "0")
        self.start.pack(side="left", padx=3)
        ttk.Label(controls, text="Limit").pack(side="left")
        self.limit = ttk.Entry(controls, width=7)
        self.limit.insert(0, "64")
        self.limit.pack(side="left", padx=3)
        pan = ttk.PanedWindow(self.body, orient="horizontal")
        pan.pack(fill="both", expand=True)
        left = ttk.Frame(pan)
        pan.add(left, weight=2)
        ttk.Label(left, text="Store", style="Head.TLabel").pack(fill="x")
        self.stores = TableList(app, left, height=4, on_select=self._store_picked)
        self.stores.headings(["store", "path", "databank", "sequences"])
        self.stores.fill([], "no store: mount one in Internals, or load a virtual model")
        self.stores.pack(fill="x")
        ttk.Label(left, text="Address").pack(fill="x", pady=(6, 0))
        self.address = ttk.Combobox(left, state="readonly")
        self.address.pack(fill="x", pady=2)
        row = ttk.Frame(left)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Summary", style="Accent.TButton",
                   command=self.summary).pack(side="left", padx=(0, 4))
        ttk.Button(row, text="Histogram", style="Ghost.TButton",
                   command=self.histogram).pack(side="left", padx=2)
        ttk.Button(row, text="Scatter", style="Ghost.TButton",
                   command=self.scatter).pack(side="left", padx=2)
        self.summary_table = TableList(app, left, height=12)
        self.summary_table.headings(["property", "value"])
        self.summary_table.fill([], "choose an address")
        self.summary_table.pack(fill="both", expand=True, pady=6)
        right = ttk.Frame(pan)
        pan.add(right, weight=3)
        ttk.Label(right, text="Distribution", style="Head.TLabel").pack(fill="x")
        self.plot = MiniPlot(app, right)
        self.plot.pack(fill="both", expand=True, pady=4)

    def activated(self):
        self.refresh()

    def server_changed(self):
        self.refresh()

    def _open_spec(self, spec):
        self.app.server.call("store.inspect", spec, self._state)

    def refresh(self):
        self.app.server.call("store.state", None, self._state)

    def _state(self, payload):
        if not payload or isinstance(payload, tuple):
            self.stores.fill([], "load a model and mount a store first")
            self.address["values"] = []
            return
        rows = payload.get("stores") or []
        self.stores.fill([{"store": r.get("store"), "path": r.get("path"),
                           "databank": r.get("databank"), "sequences": r.get("sequences")}
                          for r in rows], "no activation store on this model")
        if not rows:
            return
        first = rows[0]
        backend = "duckdb" if "duck" in str(first.get("backend", "")).lower() else "lmdb"
        self.bar.set_databanks(first.get("databanks"))
        self.bar.set_spec(first.get("databank") or "", backend)
        addresses = []
        for r in rows:
            addresses.extend(r.get("addresses") or [])
        addresses = sorted(set(addresses))
        self.address["values"] = addresses
        if addresses and self.address.get() not in addresses:
            self.address.set(addresses[0])
        self.set_subtitle(f"{len(rows)} store(s)   ·   {len(addresses)} address(es)")
        if addresses:
            self.summary()

    def _store_picked(self, row):
        if row and row.get("store"):
            self.app.server.call("store.use", {"store": row["store"]}, self._state)

    def _range(self):
        try:
            return int(self.start.get()), int(self.limit.get())
        except ValueError:
            return 0, 64

    def switch(self):
        start, _ = self._range()
        self.app.server.call("store.switch", {"databank": self.bar.databank.get(),
                                              "rewind": start == 0},
                             self._switched)

    def _switched(self, payload):
        if isinstance(payload, tuple):
            return self.app.toast(str(payload[1]), error=True)
        self.refresh()

    def summary(self):
        address = self.address.get()
        if not address:
            return
        start, limit = self._range()
        self.app.server.call("store.summary", {"address": address, "start": start, "limit": limit},
                             self._summary)

    def _summary(self, payload):
        if not payload or isinstance(payload, tuple):
            return self.summary_table.fill([], "no summary")
        flat = []
        norms = payload.get("norms") or {}
        for key, value in payload.items():
            if key in ("ids", "per_sequence_norms", "norms", "quantiles"):
                continue
            flat.append({"property": key, "value": value})
        for key, value in norms.items():
            flat.append({"property": f"norms.{key}", "value": f"{value:.4g}"})
        for key, value in (payload.get("quantiles") or {}).items():
            flat.append({"property": key, "value": f"{value:.4g}"})
        self.summary_table.fill(flat, "no summary")

    def histogram(self):
        address = self.address.get()
        if not address:
            return
        start, limit = self._range()
        self.app.server.call("store.histogram", {"address": address, "start": start,
                                                 "limit": limit, "bins": 50}, self._histogram)

    def _histogram(self, payload):
        if not payload or isinstance(payload, tuple):
            return self.app.toast(str(payload[1]) if isinstance(payload, tuple) else "no histogram",
                                  error=True)
        self.plot.bars(payload["edges"], payload["counts"])

    def scatter(self):
        address = self.address.get()
        if not address:
            return
        start, limit = self._range()
        if limit < 2 and start == 0:
            return self.app.toast("the PCA needs at least two sequences: raise Limit", error=True)
        self.app.server.call("store.matrix", {"address": address, "start": start,
                                              "limit": min(limit, 512)}, self._scatter)

    def _scatter(self, payload):
        if not payload or isinstance(payload, tuple):
            return self.app.toast(str(payload[1]) if isinstance(payload, tuple) else "no matrix",
                                  error=True)
        self.plot.scatter(payload["points"], payload.get("variance"))
