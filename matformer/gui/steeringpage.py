# matformer/gui/steeringpage.py
"""Steering: named directions kept in an activation-store databank, mounted on a socket.

A vector is one row in a store (`vectors.save`): name, address, alpha, description and the
direction itself. Applying it mounts the framework's `steering` plugin on the chosen socket; the
task pages then show a banner, because the answer is no longer only the model's."""
import json
import tkinter as tk
from tkinter import ttk

from matformer.gui.storebar import StoreBar
from matformer.gui.widgets import Console, Page, TableList, fit_dialog


class SteeringPage(Page):
    def __init__(self, app, parent, spec):
        super().__init__(app, parent, "Stores · vectors", "directions, embeddings and prototypes: one store, many uses")
        self.bar = StoreBar(self.body, app, on_open=lambda spec: self.refresh(),
                            on_refresh=lambda spec: self.refresh(), databank="vectors")
        self.bar.pack(fill="x", pady=(0, 4))
        row = ttk.Frame(self.body)
        row.pack(fill="x", pady=(0, 4))
        ttk.Button(row, text="Save vector…", command=self.save_dialog).pack(side="left")
        ttk.Button(row, text="Apply to socket…", command=self.apply_dialog).pack(side="left", padx=4)
        ttk.Button(row, text="Classify…", style="Ghost.TButton",
                   command=self.classify_dialog).pack(side="left", padx=4)
        ttk.Button(row, text="Steering off", style="Ghost.TButton",
                   command=self.clear).pack(side="left")
        ttk.Label(row, text="  Kind").pack(side="left", padx=(8, 0))
        self.kind = ttk.Combobox(row, values=["all", "steering", "embedding", "prototype",
                                              "probe", "vector"], width=11, state="readonly")
        self.kind.set("all")
        self.kind.pack(side="left")
        self.kind.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        ttk.Button(row, text="← Activations", style="Ghost.TButton",
                   command=lambda: self.app.select_page("activations")).pack(side="right")
        ttk.Label(row, text="  Socket").pack(side="left", padx=(8, 0))
        self.socket = ttk.Entry(row, width=38)
        self.socket.pack(side="left", padx=3)
        ttk.Button(row, text="Sockets…", style="Ghost.TButton",
                   command=self._socket_dialog).pack(side="left")
        ttk.Label(row, text="alpha").pack(side="left", padx=(8, 0))
        self.alpha = ttk.Entry(row, width=7)
        self.alpha.insert(0, "1.0")
        self.alpha.pack(side="left", padx=3)

        pan = ttk.PanedWindow(self.body, orient="vertical")
        pan.pack(fill="both", expand=True)
        top = ttk.Frame(pan)
        pan.add(top, weight=3)
        ttk.Label(top, text="Stored vectors", style="Head.TLabel").pack(fill="x")
        self.vectors = TableList(app, top, height=10, on_double=lambda row: self._prefill(row),
                                 on_select=lambda row: self._prefill(row))
        self.vectors.headings(["name", "kind", "label", "address", "alpha", "description"])
        self.vectors.fill([], "no vectors in this databank")
        self.vectors.pack(fill="both", expand=True)
        bottom = ttk.Frame(pan)
        pan.add(bottom, weight=2)
        ttk.Label(bottom, text="Mounted steering plugins", style="Head.TLabel").pack(fill="x")
        self.active = TableList(app, bottom, height=4)
        self.active.headings(["id", "socket", "kind", "class"])
        self.active.fill([], "steering is off")
        self.active.pack(fill="both", expand=True)
        self.note = Console(app, bottom)
        self.note.text.configure(height=2)
        self.note.pack(fill="x")

    # ── transport ───────────────────────────────────────────────────────────────
    def spec(self):
        spec = self.bar.spec()
        spec["databank"] = spec.get("databank") or "vectors"
        return spec

    def call(self, method, params, cb):
        self.app.server.call(method, params, cb)

    def activated(self):
        self.bar.seed_path()
        self.refresh()

    def server_changed(self):
        self.bar.seed_path()
        self.refresh()

    def refresh(self):
        if not self.spec().get("filename"):
            return
        params = dict(self.spec())
        if self.kind.get() not in ("", "all"):
            params["kind"] = self.kind.get()
        self.call("vectors.list", params, self._vectors)
        self.call("plugin.list", None, self._active)

    def _vectors(self, payload):
        if not payload or isinstance(payload, tuple):
            self.vectors.fill([], "no store yet: choose a file, or save the first vector")
            return
        rows = [{"name": r.get("name"), "kind": r.get("kind"), "label": r.get("label"),
                 "address": r.get("address"), "alpha": r.get("alpha"),
                 "description": r.get("description")}
                for r in payload.get("vectors") or []]
        self.vectors.fill(rows, "no vectors in this databank")
        self.bar.set_databanks(payload.get("databanks"))
        self.set_subtitle(f"{len(rows)} vector(s)   ·   {payload.get('databank', '')}")

    def _active(self, payload):
        rows = [r for r in (payload or {}).get("plugins", [])
                if r.get("kind") == "steering"] if not isinstance(payload, tuple) else []
        self.active.fill(rows, "steering is off")

    def _prefill(self, row):
        if row:
            self.socket.delete(0, "end")
            self.socket.insert(0, row.get("address") or "")
            self.alpha.delete(0, "end")
            self.alpha.insert(0, str(row.get("alpha") or 1.0))

    # ── save ────────────────────────────────────────────────────────────────────
    def save_dialog(self):
        if not self.spec().get("filename"):
            return self.app.toast("choose a vector store first", error=True)
        top = tk.Toplevel(self)
        top.title("Save a steering vector")
        top.transient(self.app)
        frame = ttk.Frame(top)
        frame.pack(fill="both", expand=True, padx=12, pady=10)
        fields = {}

        def field(row, key, label, value=""):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(frame, width=58)
            entry.insert(0, value)
            entry.grid(row=row, column=1, sticky="we", pady=3)
            fields[key] = entry
            return entry

        field(0, "name", "Name")
        ttk.Label(frame, text="Kind").grid(row=1, column=0, sticky="w", pady=3)
        kind = ttk.Combobox(frame, values=["steering", "embedding", "prototype", "probe", "vector"],
                            width=20, state="readonly")
        kind.set("steering")
        kind.grid(row=1, column=1, sticky="w", pady=3)
        field(2, "label", "Label (optional)")
        field(3, "address", "Address (socket / layer / position)")
        field(4, "alpha", "Alpha", "1.0")
        field(5, "description", "Description")
        field(6, "tags", "Tags (comma-separated)")
        field(7, "parameter", "Parameter (optional: take its first direction)")
        ttk.Label(frame, text="Direction (comma-separated floats, if no parameter)").grid(
            row=8, column=0, sticky="nw", pady=3)
        direction = tk.Text(frame, width=58, height=5, font=("TkFixedFont", 8))
        direction.grid(row=8, column=1, sticky="nsew", pady=3)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(8, weight=1)
        status = ttk.Label(frame, text="", style="Muted.TLabel")
        status.grid(row=9, column=1, sticky="w")

        def parsed():
            text = direction.get("1.0", "end").strip()
            if not text:
                return None
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return [float(part) for part in text.replace("\n", ",").split(",") if part.strip()]

        def go(direction_values=None):
            name = fields["name"].get().strip()
            if not name:
                return status.configure(text="a name is required")
            if direction_values is None:
                try:
                    direction_values = parsed()
                except ValueError as problem:
                    return status.configure(text=f"direction is not numbers: {problem}")
            if not direction_values:
                return status.configure(text="a direction or a parameter is required")
            self.call("vectors.save", {**self.spec(), "name": name,
                                       "kind": kind.get() or "steering",
                                       "label": fields["label"].get().strip() or None,
                                       "address": fields["address"].get().strip() or "vector",
                                       "alpha": float(fields["alpha"].get() or 1.0),
                                       "description": fields["description"].get().strip(),
                                       "tags": [tag.strip() for tag in
                                                fields["tags"].get().split(",") if tag.strip()],
                                       "vector": direction_values}, saved)
        ttk.Button(frame, text="Save", style="Accent.TButton",
                   command=go).grid(row=10, column=1, sticky="e", pady=(6, 0))
        ttk.Button(frame, text="Cancel", style="Ghost.TButton",
                   command=top.destroy).grid(row=10, column=0, sticky="w", pady=(6, 0))

        def from_parameter():
            name = fields["parameter"].get().strip()
            if not name:
                return status.configure(text="type a parameter name (see Internals)")
            self.call("param.directions", {"name": name, "k": 1}, got_direction)

        def got_direction(payload):
            if not payload or isinstance(payload, tuple):
                return status.configure(text="cannot read that parameter")
            go(payload["directions"][0])
        ttk.Button(frame, text="Use parameter", style="Ghost.TButton",
                   command=from_parameter).grid(row=7, column=2, padx=4)

        def saved(answer):
            if isinstance(answer, tuple):
                return status.configure(text=str(answer[1]))
            self.app.toast("vector saved")
            top.destroy()
            self.refresh()
        fit_dialog(top, 720, 420)

    # ── apply / clear ───────────────────────────────────────────────────────────
    def apply_dialog(self):
        row = self.vectors.selected()
        if not row:
            return self.app.toast("select a vector first", error=True)
        socket = self.socket.get().strip() or row.get("address") or "*"
        try:
            alpha = float(self.alpha.get() or row.get("alpha") or 1.0)
        except ValueError:
            return self.app.toast("alpha must be a number", error=True)
        self.call("steer.apply", {"vector": {**self.spec(), "name": row["name"]},
                                  "socket": socket, "alpha": alpha}, self._applied)

    def _applied(self, answer):
        if isinstance(answer, tuple):
            return self.app.toast(str(answer[1]), error=True)
        self.app.toast(f"steering '{answer['vector']['name']}' on {answer['socket']}")
        self.refresh()

    def clear(self):
        self.call("steer.clear", None, lambda answer: self.app.toast(
            f"removed {len((answer or {}).get('removed') or [])} steering plugin(s)"
            if not isinstance(answer, tuple) else str(answer[1])))
        self.refresh()

    def _socket_dialog(self):
        def got(payload):
            if not payload or isinstance(payload, tuple):
                return self.app.toast("no sockets", error=True)
            top = tk.Toplevel(self)
            top.title("Sockets")
            top.transient(self.app)
            table = TableList(self.app, top, height=16,
                              on_double=lambda row: self._chosen(row, top))
            table.headings(["address", "mounted", "count"])
            table.fill(payload.get("sockets") or [], "no socket")
            table.pack(fill="both", expand=True, padx=8, pady=8)
            ttk.Button(top, text="Close", command=top.destroy).pack(anchor="e", padx=8, pady=6)
            fit_dialog(top, 720, 480)
        self.call("plugin.sockets", {"pattern": "*"}, got)

    def _chosen(self, row, top):
        if row:
            self.socket.delete(0, "end")
            self.socket.insert(0, row["address"])
            top.destroy()

    # ── classify by similarity ──────────────────────────────────────────────────
    def classify_dialog(self):
        row = self.vectors.selected()
        if not row:
            return self.app.toast("select a vector to classify", error=True)
        top = tk.Toplevel(self)
        top.title(f"Classify «{row['name']}» by similarity")
        top.transient(self.app)
        bar = ttk.Frame(top)
        bar.pack(fill="x", padx=10, pady=8)
        ttk.Label(bar, text="Compare with").pack(side="left")
        kind = ttk.Combobox(bar, values=["all", "prototype", "embedding"], width=13,
                            state="readonly")
        kind.set("all")
        kind.pack(side="left", padx=4)
        ttk.Label(bar, text="k").pack(side="left")
        k = ttk.Entry(bar, width=5)
        k.insert(0, "5")
        k.pack(side="left", padx=4)
        holder = ttk.Frame(top)
        holder.pack(fill="both", expand=True, padx=10)
        labels = TableList(self.app, holder, height=6)
        labels.headings(["label", "vote", "best"])
        labels.fill([], "run to see the labels")
        labels.pack(fill="x")
        neighbours = TableList(self.app, holder, height=10)
        neighbours.headings(["name", "kind", "label", "score", "address"])
        neighbours.fill([], "neighbours")
        neighbours.pack(fill="both", expand=True, pady=(6, 0))
        status = ttk.Label(top, text="", style="Muted.TLabel")
        status.pack(fill="x", padx=10)

        def run():
            params = {**self.spec(), "name": row["name"], "k": int(k.get() or 5)}
            if kind.get() != "all":
                params["search_kind"] = kind.get()
            self.call("vectors.classify", params, got)

        def got(payload):
            if not payload or isinstance(payload, tuple):
                status.configure(text=str(payload[1]) if isinstance(payload, tuple) else "no answer")
                return
            labels.fill([{"label": one["label"], "vote": f"{one['vote']:.1%}",
                          "best": f"{one['best']:.3f}"} for one in payload.get("labels") or []],
                        "no labelled neighbour")
            neighbours.fill(payload.get("neighbours") or [], "no neighbour")

        ttk.Button(bar, text="Run", style="Accent.TButton", command=run).pack(side="left", padx=6)
        ttk.Button(bar, text="Close", style="Ghost.TButton", command=top.destroy).pack(side="right")
        fit_dialog(top, 760, 520)
        run()
