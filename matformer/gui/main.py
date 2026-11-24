# matformer/gui/main.py
"""Matformer Console — the desktop shell.   python -m matformer.gui

Design system: theme.py (tokens, tiers, focus). Pages, menus, toolbar, themes, servers come from
config/app.toml; the Tasks group is generated from the local registry. One global server selector
(the toolbar pill) is shared by every page — the server you picked anywhere is the server
everywhere. The bottom dock (Jobs + Log) starts collapsed, shows a badge, opens itself when a job
starts or fails, and toggles with Ctrl+J. Every server callback is guarded: tracebacks go to the
Log, one-line red toasts to the status bar, and the UI never dies."""
import json
import os
import queue
import sys
import threading
import time
import tomllib
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, ttk

os.environ.setdefault("MATFORMER_SERVER", "0")      # the GUI does not host a server

from matformer.gui import theme
from matformer.gui.clusterpage import ClusterPage
from matformer.gui.configpage import ConfigPage
from matformer.gui.explorer import Explorer, pick
from matformer.gui.favspage import FavPage
from matformer.gui.icons import ICONS, icon, installed_themes
from matformer.gui.local import LocalPage
from matformer.gui.mdatpage import MdatPage
from matformer.gui.regpage import RegistryPage
from matformer.gui.runspage import RunsPage
from matformer.gui.serverpage import ServerPage
from matformer.gui.syspage import SysPage
from matformer.gui.taskpage import TaskPage
from matformer.gui.commands import CommandsPage
from matformer.gui.trainpage import TrainPage
from matformer.gui.traffic import Traffic
from matformer.gui.widgets import Console, TableList, dismiss as widgets_dismiss, popup
from matformer.remote import connect as remote_connect
from matformer.remote.net import Server

ROOT = Path(__file__).resolve().parent
SETTINGS_FILE = Path.home() / ".matformer_gui.json"


def local_task_types():
    try:
        from matformer.matformer_registry import registry
        return sorted(registry.get_registry_info(category="task")["task"]["variants"])
    except Exception:
        return ["generation", "evaluation", "masking"]


PAGES = {"task": TaskPage,
         "files": lambda app, parent, spec: Explorer(app, parent, spec.get("root") or "", win=True),
         "config": ConfigPage, "training": TrainPage, "runs": RunsPage, "mdat": MdatPage,
         "favs": FavPage, "sys": SysPage, "reg": RegistryPage, "server": ServerPage,
         "commands": CommandsPage, "local": LocalPage, "cluster": ClusterPage}


def _mnemonic(label):
    return (label.replace("&", ""), label.index("&") - 1) if "&" in label else (label, None)


def _tk_accel(spec):
    mods = {"ctrl": "Control", "shift": "Shift", "alt": "Alt"}
    keys = {"/": "slash", ",": "comma", "+": "plus", "-": "minus", "escape": "Escape"}
    parts = []
    for p in spec.split("+"):
        low = p.lower()
        parts.append(mods.get(low) or keys.get(low) or (low if len(p) == 1 else p))
    return "<" + "-".join(parts) + ">"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.cfg = tomllib.loads((ROOT / "config" / "app.toml").read_text("utf-8"))
        self.settings = self._load_settings()
        self.title("Matformer Console")
        self.geometry(self.settings.get("geometry", "1440x880"))
        self.theme_name = self.settings.get("theme", self.cfg["default_theme"])
        self.appearance = {"family": "", "ui": 9, "mono": 9, "icon_theme": "auto",
                           **self.cfg.get("appearance", {}), **self.settings.get("appearance", {})}
        self._palette = theme.palette_of(self.cfg, self.theme_name)
        ICONS.set_theme(self.appearance.get("icon_theme", "auto"))
        self.events, self.paintables = queue.Queue(), []
        self.fm_mode = "details"
        self._toast_job = None
        self.traffic_recording = False
        self.server_var = tk.StringVar(value=(self.settings.get("servers") or self.cfg["server"])[0]["name"])
        servers = self.settings.get("servers") or self.cfg["server"]
        self.server_specs = servers
        self.servers = {}
        for s in servers:
            url = s.get("url") or f"127.0.0.1:{s['ssh'].get('remote_port', 8765)}"
            self.servers[s["name"]] = Server(s["name"], url, self.events, key=s.get("key"),
                                             trace=self._trace)
        self.server_names = list(self.servers)
        self.set_theme(self.theme_name, save=False)
        self.log_console = Console(self, self)
        self._build_pages()
        self._build_shell()
        for s in self.servers.values():
            s.start()
        for spec in servers:
            if spec.get("ssh"):
                threading.Thread(target=self._bootstrap_ssh,
                                 args=(self.servers[spec["name"]], spec["ssh"]), daemon=True).start()
        self._poll()

    # ── settings / theme ───────────────────────────────────────────────────────
    def _load_settings(self):
        try:
            return json.loads(SETTINGS_FILE.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_settings(self):
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps({"geometry": self.geometry(), "theme": self.theme_name,
                                                 "appearance": self.appearance,
                                                 "servers": self.server_specs}, indent=1), "utf-8")
        except OSError:
            pass

    def palette(self):
        return self._palette

    def register_paintable(self, widget):
        self.paintables.append(widget)

    def set_theme(self, name, save=True):
        self.theme_name = name
        self._palette = theme.palette_of(self.cfg, name)
        theme.apply(self, name, self._palette, self.appearance)
        for w in self.paintables:
            if isinstance(w, tk.Text):
                theme.text_colors(w, self._palette)
            elif hasattr(w, "repaint"):
                try:
                    w.repaint(self._palette)
                except tk.TclError:
                    pass
        for menu in self.menubar.winfo_children() if hasattr(self, "menubar") else []:
            menu.configure(**theme.menu_colors(self._palette))
        self._paint_pills()
        if save:
            self._save_settings()

    def set_appearance(self, **changes):
        self.appearance.update({k: v for k, v in changes.items() if v is not None})
        ICONS.set_theme(self.appearance.get("icon_theme", "auto"))
        self.set_theme(self.theme_name)
        self._save_settings()
        self.toast("appearance updated")

    def do_theme(self):
        names = list(self.cfg["themes"])
        self.set_theme(names[(names.index(self.theme_name) + 1) % len(names)])

    # ── global server ──────────────────────────────────────────────────────────
    @property
    def server(self):
        return self.servers[self.server_var.get()]

    def server_of(self, name=None):
        return self.servers[name or self.server_var.get()]

    # ── pages (Tasks group from the local registry) ────────────────────────────
    def _build_pages(self):
        task_pages = [{"id": f"task:{t}", "group": "Tasks", "title": t.replace("_", " ").capitalize(),
                       "kind": "task", "task_type": t} for t in local_task_types()]
        others = [p for p in self.cfg["pages"] if p.get("kind") != "task"]
        self.page_specs = task_pages + others

    # ── shell ──────────────────────────────────────────────────────────────────
    def _build_shell(self):
        self.menubar = tk.Menu(self)
        self.configure(menu=self.menubar)
        self._build_menus()

        tb = ttk.Frame(self)
        tb.pack(fill="x", padx=8, pady=(4, 0))
        for name in self.cfg.get("toolbar", []):
            label = name.capitalize()
            ttk.Button(tb, text=label, style="Ghost.TButton", command=getattr(self, "do_" + name)
                       ).pack(side="left", padx=1)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=8)
        pill = ttk.Frame(tb)
        pill.pack(side="left")
        self.pill_dot = tk.Canvas(pill, width=10, height=10, highlightthickness=0, bd=0,
                                  bg=self.palette()["bg"])
        self.pill_dot.pack(side="left", padx=(0, 3))
        self.pill = ttk.Combobox(pill, values=self.server_names, textvariable=self.server_var,
                                 state="readonly", width=11)
        self.pill.pack(side="left")
        self.pill.bind("<<ComboboxSelected>>", lambda e: self._server_changed())
        for w in (pill, self.pill, self.pill_dot):
            w.bind("<Button-3>", lambda e: self._pill_menu(e, self.server_var.get()))
        self.model_chip = ttk.Label(tb, text="no model", style="Faint.TLabel")
        self.model_chip.pack(side="left", padx=10)
        self.dock_open = False

        outer = ttk.PanedWindow(self, orient="vertical")
        outer.pack(fill="both", expand=True, padx=4, pady=4)
        self.outer = outer
        self.main_pane = ttk.PanedWindow(outer, orient="horizontal")
        outer.add(self.main_pane, weight=1)

        nav_holder = ttk.Frame(self.main_pane)
        self.main_pane.add(nav_holder, weight=0)
        self.nav = ttk.Treeview(nav_holder, show="tree", selectmode="browse")
        self.nav.pack(fill="both", expand=True)
        self.nav.column("#0", width=170)
        self.nav.bind("<<TreeviewSelect>>", self._on_nav)

        self.stack = ttk.Frame(self.main_pane)
        self.main_pane.add(self.stack, weight=1)
        self.pages = {}
        for spec in self.page_specs:
            page = PAGES[spec["kind"]](self, self.stack, spec)
            self.pages[spec["id"]] = page
        groups = {}
        for spec in self.page_specs:
            if spec["group"] not in groups:
                groups[spec["group"]] = self.nav.insert("", "end", text=spec["group"], open=True)
            self.nav.insert(groups[spec["group"]], "end", iid=spec["id"], text=spec["title"])

        self.dock_holder = ttk.Frame(outer)
        self.dock = ttk.Notebook(self.dock_holder)
        self.dock.pack(fill="both", expand=True)
        self.jobs = TableList(self, self.dock, status_col="status", height=4)
        self.jobs.headings(["server", "id", "name", "project", "status", "step", "progress"])
        self.jobs.tree.bind("<Button-3>", self._jobs_menu)
        self.jobs.fill([], "no jobs on any server")
        self.dock.add(self.jobs, text="Jobs")
        self.dock.add(self.log_console, text="Log")
        self.traffic = Traffic(self, self.dock)
        self.dock.add(self.traffic, text="Console")
        self._dock_job_count = 0

        status = ttk.Frame(self)
        status.pack(fill="x", padx=8)
        self.status_label = ttk.Label(status, text="ready")
        self.status_label.pack(side="left")
        self.leds = {}
        for name in self.servers:
            led = ttk.Label(status, text=name, style="Muted.TLabel")
            led.pack(side="right", padx=4)
            self.leds[name] = led
        self.current = None
        self.select_page(self.page_specs[0]["id"])
        self.toggle_dock(force=False)
        self.bind_all("<Button-1>", lambda e: widgets_dismiss(), add="+")

    def _paint_pills(self):
        for name, led in getattr(self, "leds", {}).items():
            up = self.servers[name].up
            led.configure(text=f"● {name}" if up else f"○ {name}",
                          foreground=self.palette()["ok"] if up else self.palette()["faint"])

    def _docked(self):
        """PanedWindow.panes() answers with paths, not widgets: compare those, or the dock can
        never be forgotten and never be re-added."""
        return str(self.dock_holder) in {str(pane) for pane in self.outer.panes()}

    def toggle_dock(self, force=None, height=None):
        """The dock is a pane of the vertical PanedWindow: it is added back at the bottom (never
        packed, which used to drop it above the pages) and sized by its sash."""
        self.dock_open = force if force is not None else not self.dock_open
        if not self.dock_open:
            if self._docked():
                self.outer.forget(self.dock_holder)
            return
        if not self._docked():
            self.outer.add(self.dock_holder, weight=0)
        self.update_idletasks()
        wanted = height or self.settings.get("dock_height", 210)
        self.settings["dock_height"] = wanted
        try:
            self.outer.sashpos(len(self.outer.panes()) - 2,
                               max(80, self.outer.winfo_height() - wanted))
            self._sash_set = True
        except tk.TclError:
            pass                                # sash not drawable yet: the next pump will set it

    def _pill_menu(self, e, name):
        srv = self.servers[name]
        m = tk.Menu(self, tearoff=0, **theme.menu_colors(self.palette()))
        (m.add_command if not srv.up else m.add_command)(
            label="Disconnect" if srv.up else "Connect",
            command=lambda: (srv.disconnect() if srv.up else srv.connect()))
        m.add_command(label="Server panel", command=lambda: self.select_page("server"))
        popup(m, e.x_root, e.y_root)

    def _server_changed(self):
        for page in self.pages.values():
            if hasattr(page, "server_changed"):
                page.server_changed()
        page = self.pages.get(self.current)
        page and page.activated()

    def _build_menus(self):
        self.menubar.delete(0, "end")
        for m in self.cfg.get("menu", []):
            label, under = _mnemonic(m["title"])
            menu = tk.Menu(self.menubar, tearoff=0, **theme.menu_colors(self.palette()))
            self.menubar.add_cascade(label=label, underline=under, menu=menu)
            for it in m["items"]:
                if it == "-":
                    menu.add_separator()
                    continue
                label, under = _mnemonic(it[0])
                menu.add_command(label=label, underline=under, accelerator=it[2],
                                 command=getattr(self, "do_" + it[1]))
                if it[2]:
                    try:
                        self.bind_all(_tk_accel(it[2]),
                                      lambda e, f=getattr(self, "do_" + it[1]): f())
                    except tk.TclError:
                        pass
        pages_menu = tk.Menu(self.menubar, tearoff=0, **theme.menu_colors(self.palette()))
        self.menubar.insert_cascade(1, label="Go", underline=1, menu=pages_menu)
        for i, spec in enumerate(self.page_specs):
            key = f"Ctrl+{i + 1}" if i < 9 else ""
            pages_menu.add_command(label=spec["title"], accelerator=key,
                                   command=lambda pid=spec["id"]: self.select_page(pid))
            if i < 9:
                self.bind_all(f"<Control-Key-{i + 1}>", lambda e, pid=spec["id"]: self.select_page(pid))

    # ── navigation / dispatch ──────────────────────────────────────────────────
    def select_page(self, page_id):
        if page_id in self.pages:
            self.nav.selection_set(page_id)
            self.nav.see(page_id)

    def _on_nav(self, _e):
        sel = self.nav.selection()
        if not sel or sel[0] not in self.pages:
            return
        for page in self.pages.values():
            page.place_forget()
        page = self.pages[sel[0]]
        page.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.current = sel[0]
        try:
            page.activated()
        except Exception as problem:
            self._page_error(problem)

    def _page_error(self, problem):
        self.log("GUI", traceback.format_exc())
        self.toast(f"{type(problem).__name__}: {problem}", error=True)

    def do_refresh(self):
        self.pages[self.current].activated()

    def do_send(self):
        page = self.active_task_page
        page and page.send()

    def do_stop(self):
        page = self.active_task_page
        page and page.stop()

    def do_clear(self):
        page = self.active_task_page
        page and page.view.clear()

    def do_connect(self):
        self.status("reconnection is automatic: down servers retry every 3 s")

    def do_quit(self):
        self.destroy()

    def do_settings(self):
        PrefsDialog(self)

    def do_open_config(self):
        p = self.browse("", (".yaml", ".yml"))
        if p:
            self.open_config(p)

    def do_open_file(self):
        p = self.browse()
        if p:
            self.open_file(p)

    def do_export_log(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            Path(path).write_text(self.log_console.text.get("1.0", "end"), "utf-8")
            self.toast(f"log exported to {path}")

    def do_export_transcript(self):
        page = self.active_task_page
        if page:
            path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
            if path:
                Path(path).write_text(page.view.text.get("1.0", "end"), "utf-8")
                self.toast(f"transcript exported to {path}")

    def do_run_task(self):
        page = self.active_task_page
        page and page.run_task()

    def do_annotate(self):
        page = self.active_task_page
        page and page.annotate()

    def do_unload_model(self):
        self.server.call("task.unload", None, lambda r: self.toast(str(r)))

    def do_free_memory(self):
        self.server.call("sys.gc", None, lambda r: self.toast(str(r)))

    def do_export_console(self):
        self.traffic.export()

    def do_open_dataset(self):
        p = self.browse("", ())
        if p:
            self.open_dataset(p)

    def do_token_detail(self):
        page = self.active_task_page
        page and page.show_details()

    def do_detail_off(self):
        self._detail_knob(0)

    def do_detail_5(self):
        self._detail_knob(5)

    def do_detail_25(self):
        self._detail_knob(25)

    def do_detail_all(self):
        self._detail_knob(-1)

    def _detail_knob(self, value):
        """How much of every prediction the server keeps: the speed/detail trade-off, one keystroke
        away and never decided for the user. 0 keeps nothing, -1 keeps everything."""
        self.server.call("task.set", {"path": "topk_present", "value": value},
                         lambda r: self.toast(
                             f"token detail: {'off' if value == 0 else ('everything' if value < 0 else f'top {value}')}")
                         if not isinstance(r, tuple) else self.toast(r[1], error=True))

    def do_rpc_reference(self):
        self.select_page("commands")

    def do_console(self):
        self.toggle_dock(force=True)
        self.dock.select(self.traffic)

    def do_server_errors(self):
        self.select_page("commands")
        self.pages["commands"].show_failures()

    def do_command_palette(self):
        CommandPalette(self)

    def do_about(self):
        from tkinter import messagebox
        messagebox.showinfo("Matformer Console", "Matformer Console")

    def do_sync_catalog(self):
        """Merge every connected server's runs into the local sqlite catalog, and push the merged
        view back: offline laptops keep seeing all runs, servers keep knowing about each other."""
        from matformer.utils.catalog import Catalog
        state = {"waiting": 0, "done": 0}

        def finish():
            state["done"] += 1
            if state["done"] < state["waiting"]:
                return
            rows = Catalog().as_rows()
            for srv in self.servers.values():
                if srv.up:
                    srv.call("catalog.upsert", {"rows": rows})
            self.toast(f"run catalog synced ({len(rows)} runs)")
            self.pages["runs"].catalog_reload()

        for name, srv in self.servers.items():
            if srv.up:
                state["waiting"] += 1
                srv.call("runs.list", None,
                         lambda rows, n=name: rows and not isinstance(rows, tuple)
                         and (Catalog().replace_server(n, rows), finish()))
        if not state["waiting"]:
            self.toast("no server connected", error=True)

    def do_reload_schema(self):
        page = self.active_task_page
        page and page.fields()

    def do_toggle_dock(self):
        self.toggle_dock()

    @property
    def active_task_page(self):
        cur = self.pages.get(self.current)
        if isinstance(cur, TaskPage):
            return cur
        return next((p for p in self.pages.values() if isinstance(p, TaskPage)), None)

    # ── cross-page actions ─────────────────────────────────────────────────────
    def open_file(self, path, server=None):
        low = path.lower()
        if low.endswith((".yaml", ".yml")):
            self.open_config(path, server)
        elif low.endswith((".ckpt", ".pt", ".safetensors", ".bin")) or "://" in path:
            self.open_model(path, server)
        elif low.endswith((".jsonl", ".json", ".csv")):
            self.annotate_file(path, server)
        else:
            self.open_config(path, server)

    def open_config(self, path, server=None):
        if server in self.servers:
            self.server_var.set(server)
        self.select_page("config")
        self.pages["config"].open(path)

    def open_model(self, path, server=None):
        if server in self.servers:
            self.server_var.set(server)
        page = self.active_task_page
        if page is None:
            return
        self.select_page(next(pid for pid, p in self.pages.items() if p is page))
        page.model.delete(0, "end")
        page.model.insert(0, path)
        page.load()

    def open_dataset(self, path, server=None):
        if server in self.servers:
            self.server_var.set(server)
        self.select_page("mdat")
        page = self.pages["mdat"]
        page.path.delete(0, "end")
        page.path.insert(0, path)
        page.call("open", {"path": path, "readonly": True})

    def annotate_file(self, path, server=None):
        if server in self.servers:
            self.server_var.set(server)
        page = self.active_task_page
        if page:
            page.server().call("task.annotate", {"input": path},
                               lambda r: r and page.view.put(f"annotated -> {r.get('output')}\n", "meta"))

    def start_training(self, config_path, server=None, overrides=None):
        if server in self.servers:
            self.server_var.set(server)
        self.select_page("training")
        page = self.pages["training"]
        page.add_config(config_path)
        if overrides is not None:
            page.overrides = overrides
        page.queue_dialog()

    def show_run(self, run_id):
        self.select_page("runs")
        self.pages["runs"].focus_run(run_id)

    def browse(self, root="", exts=()):
        return pick(self, self, root, tuple(exts), self.server_var.get())

    def copy(self, path):
        self.clipboard_clear()
        self.clipboard_append(str(path))
        self.toast(f"copied: {path}")

    def status(self, msg):
        self.status_label.configure(text=msg, foreground=self.palette()["text"])

    def toast(self, msg, error=False):
        p = self.palette()
        full = str(msg)
        short = full.splitlines()[0][:180] + (" …(full text in the Log)" if len(full) > 180 else "")
        self.status_label.configure(text=short, foreground=p["err"] if error else p["text"])
        if self._toast_job:
            self.after_cancel(self._toast_job)
        self._toast_job = self.after(8000, lambda: self.status_label.configure(
            text="ready", foreground=p["muted"]))
        if error:
            self.log("GUI", full)

    # ── server traffic ─────────────────────────────────────────────────────────
    def _bootstrap_ssh(self, srv, ssh_cfg):
        time.sleep(4)
        if srv.up or srv.stop.is_set():
            return
        try:
            host, port = remote_connect({"name": srv.name, "ssh": ssh_cfg})
            srv.set_address(host, port)
            self.log(srv.name, f"ssh tunnel ready on {host}:{port}")
        except Exception as problem:
            self.toast(f"{srv.name}: {problem}", error=True)

    def _poll(self):
        if not self.winfo_exists():
            return
        try:
            while True:
                sname, kind, data = self.events.get_nowait()
                if kind == "__result__":
                    rid, result = data
                    srv = self.servers.get(sname)
                    cb = srv.pending.pop(rid, None) if srv else None
                    if cb:
                        try:
                            cb(result)
                        except Exception:
                            self._page_error(Exception(traceback.format_exc()))
                elif kind == "__state__":
                    self._paint_pills()
                    self.log(sname, "connected" if data else "disconnected")
                    if data:
                        self.servers[sname].call("jobs.watch")
                elif kind == "__dead__":
                    for rid, cb in data.items():
                        try:
                            cb(("error", "connection lost"))
                        except Exception:
                            self.log("GUI", traceback.format_exc())
                else:
                    self.on_event(sname, kind, data)
        except queue.Empty:
            pass
        finally:
            if self.dock_open and not getattr(self, "_sash_set", False) and self.outer.winfo_height() > 120:
                try:
                    self.outer.sashpos(len(self.outer.panes()) - 2,
                                       self.outer.winfo_height() - self.settings.get("dock_height", 210))
                    self._sash_set = True
                except tk.TclError:
                    pass
            if self.traffic.recording:
                self.traffic.flush()
            self.after(60, self._poll)

    def _trace(self, server, line):
        """Called by the client threads for every command and answer: cheap, Tk-free, dropped
        unless the Console tab is recording."""
        self.traffic.record(server, line)

    def on_event(self, server, name, data):
        if name == "job":
            self.jobs.upsert("id", {"server": server, **data})
            self.log("JOB", f"{server}: {data.get('id')} {data.get('status')}"
                            + (f" step={data.get('step')}" if data.get("step") else ""))
            if data.get("status") in ("starting", "running", "failed"):
                self.toggle_dock(force=True)
        elif name == "task_state":
            self.log("SERVER", f"{server}: model {data.get('source') or 'unloaded'}"
                               f" task {data.get('task')}")
            self.model_chip.configure(text=data.get("source") or "no model")
            for page in self.pages.values():
                if hasattr(page, "task_state"):
                    page.task_state(data)
        elif name == "error":
            self.toast(f"{server}: {data}", error=True)
        for page in self.pages.values():
            if hasattr(page, "on_event"):
                try:
                    page.on_event(server, name, data)
                except Exception:
                    self.log("GUI", traceback.format_exc())

    def log(self, source, msg):
        self.log_console.put(f"{time.strftime('%H:%M:%S')}  [{source}] {msg}\n")

    def _jobs_menu(self, e):
        row = self.jobs.selected()
        if not row:
            return
        m = tk.Menu(self, tearoff=0, **theme.menu_colors(self.palette()))
        m.add_command(label="Abort job",
                      command=lambda: self.servers[row["server"]].call("train.abort", {"id": row["id"]}))
        m.add_command(label="Export log…",
                      command=lambda: self._export_job_log(row["server"], row["id"]))
        m.add_command(label="Open in Plots", command=lambda: self.show_run(row["id"]))
        m.add_command(label="Copy id", command=lambda: self.copy(row["id"]))
        popup(m, e.x_root, e.y_root)

    def _export_job_log(self, server, job_id):
        def got(chunk):
            if not chunk:
                return
            path = filedialog.asksaveasfilename(defaultextension=".log", initialfile=f"{job_id}.log")
            if path:
                Path(path).write_text(chunk.get("text", ""), "utf-8")
                self.toast(f"log exported: {path}")
        self.servers[server].call("jobs.log", {"id": job_id}, got)

    def destroy(self):
        self._save_settings()
        for s in self.servers.values():
            s.shutdown()
        super().destroy()


class CommandPalette(tk.Toplevel):
    """Every server command, searchable, runnable: the power-user back door."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Commands")
        self.geometry("640x480")
        self.transient(app)
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)
        self.filter = ttk.Entry(top)
        self.filter.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.filter.bind("<KeyRelease>", lambda e: self._show())
        self.filter.bind("<Return>", lambda e: self._run())
        self.filter.bind("<Escape>", lambda e: self.destroy())
        self.list = tk.Listbox(self, font=("TkFixedFont", 9))
        self.list.pack(fill="both", expand=True, padx=8)
        self.args = ttk.Entry(self)
        self.args.pack(fill="x", padx=8, pady=6)
        self.args.insert(0, '{}')
        ttk.Button(self.args, text="Run", width=5, command=self._run).pack(side="right")
        self.list.bind("<Double-1>", lambda e: self._run())
        self.all = []
        app.server.call("rpc.list", None, self._loaded)
        self.filter.focus_set()

    def _loaded(self, cmds):
        if cmds and not isinstance(cmds, tuple):
            self.all = cmds
            self._show()

    def _show(self):
        t = self.filter.get().lower()
        self.list.delete(0, "end")
        for c in self.all:
            if t in c["name"] or t in c["doc"].lower():
                self.list.insert("end", f"{c['name']:<26} {c['doc']}")

    def _run(self):
        i = self.list.curselection()
        if not i:
            return
        method = self.all[i[0]]["name"]
        try:
            params = json.loads(self.args.get() or "{}")
        except json.JSONDecodeError as problem:
            return self.app.toast(f"params must be JSON: {problem}", error=True)
        self.app.server.call(method, params, lambda r: (self.app.log("CMD", f"{method} -> {r}"),
                                                        self.app.toast(f"{method} done")))


class PrefsDialog(tk.Toplevel):
    """Preferences: Servers (full CRUD: add / edit / delete / test) and Appearance (palette,
    fonts, icon theme). Saved to ~/.matformer_gui.json; servers apply on restart."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Preferences")
        self.geometry("640x520")
        self.transient(app)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)
        self._servers_tab(nb)
        self._appearance_tab(nb)
        ttk.Button(self, text="Close", command=self.destroy).pack(anchor="e", padx=10, pady=(0, 10))

    def _servers_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Servers")
        self.table = TableList(self.app, tab, on_double=self._edit, height=8)
        self.table.headings(["name", "url", "ssh"])
        self._refresh_rows()
        self.table.pack(fill="both", expand=True, padx=8, pady=8)
        row = ttk.Frame(tab)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Button(row, text="Add…", command=self._add).pack(side="left")
        ttk.Button(row, text="Edit…", command=self._edit).pack(side="left", padx=4)
        ttk.Button(row, text="Delete", style="Danger.TButton", command=self._delete).pack(side="left")
        ttk.Button(row, text="Test connection", command=self._test).pack(side="right")
        ttk.Label(tab, style="Muted.TLabel", text="Changes are written to ~/.matformer_gui.json; "
                  "restart the console to (re)connect. Double-click a row to edit.").pack(
            anchor="w", padx=8, pady=(0, 8))

    def _list(self):
        return self.app.settings.get("servers") or self.app.cfg["server"]

    def _refresh_rows(self):
        self.table.fill([{"name": s["name"], "url": s.get("url", ""),
                          "ssh": s.get("ssh", {}).get("host", "")} for s in self._list()],
                        "No servers configured — press Add.")

    def _store(self, servers):
        data = self.app.settings
        data["servers"] = servers
        self.app.settings = data
        self.app._save_settings()
        self._refresh_rows()

    def _selected(self):
        row = self.table.selected()
        return row["name"] if row else None

    def _add(self):
        dlg = ServerDialog(self, {"name": "", "url": "127.0.0.1:8765"})
        self.wait_window(dlg)
        if dlg.result:
            servers = self._list() + [dlg.result]
            self._store(servers)

    def _edit(self, *_):
        name = self._selected()
        if not name:
            return
        current = next(s for s in self._list() if s["name"] == name)
        dlg = ServerDialog(self, current)
        self.wait_window(dlg)
        if dlg.result:
            self._store([dlg.result if s["name"] == name else s for s in self._list()])

    def _delete(self):
        name = self._selected()
        if name:
            self._store([s for s in self._list() if s["name"] != name])

    def _test(self):
        from matformer.remote.ssh import reachable
        name = self._selected()
        if not name:
            return
        s = next(x for x in self._list() if x["name"] == name)
        try:
            host, port = (remote_connect(s, name) if s.get("ssh")
                          else (s["url"].split(":")[0], s["url"].rsplit(":", 1)[1]))
            self.app.toast(f"{name}: {'reachable' if reachable(host, int(port)) else 'no answer'}")
        except Exception as problem:
            self.app.toast(f"{name}: {problem}", error=True)

    def _appearance_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Appearance")
        a = self.app.appearance
        form = ttk.Frame(tab)
        form.pack(fill="x", padx=14, pady=14)
        form.columnconfigure(1, weight=1, minsize=220)
        r = 0

        def row(label, widget):
            nonlocal r
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", pady=4)
            widget.grid(row=r, column=1, sticky="ew", pady=4)
            r += 1

        self.cb_theme = ttk.Combobox(form, values=list(self.app.cfg["themes"]), state="readonly")
        self.cb_theme.set(self.app.theme_name)
        self.cb_theme.bind("<<ComboboxSelected>>", lambda e: self.app.set_theme(self.cb_theme.get()))
        row("Palette:", self.cb_theme)
        from tkinter import font as tkfont
        fams = [""] + sorted(set(tkfont.families(self.app)))
        self.cb_font = ttk.Combobox(form, values=fams)
        self.cb_font.set(a.get("family", ""))
        row("Font family (empty = auto):", self.cb_font)
        self.ui_size = ttk.Spinbox(form, from_=7, to=14, width=5)
        self.ui_size.set(str(a["ui"]))
        row("UI font size:", self.ui_size)
        self.mono_size = ttk.Spinbox(form, from_=7, to=16, width=5)
        self.mono_size.set(str(a["mono"]))
        row("Console font size:", self.mono_size)
        self.cb_icons = ttk.Combobox(form, values=["auto", "none"] + installed_themes(),
                                     state="readonly")
        self.cb_icons.set(a.get("icon_theme", "auto"))
        row("Icon theme (freedesktop):", self.cb_icons)
        ttk.Button(tab, text="Apply",
                   command=lambda: self.app.set_appearance(
                       family=self.cb_font.get(), icon_theme=self.cb_icons.get(),
                       ui=int(self.ui_size.get() or 9), mono=int(self.mono_size.get() or 9))
                   ).pack(anchor="w", padx=14, pady=4)
        lbl = ttk.Label(tab, style="Muted.TLabel",
                        text="Icon themes are read live from /usr/share/icons (PNG; Tk cannot "
                             "render SVG — missing sizes fall back to the nearest, then to the "
                             "built-in drawn shapes).")
        lbl.configure(wraplength=520)
        lbl.pack(anchor="w", padx=14)


class ServerDialog(tk.Toplevel):
    """Add/edit one server: plain url or full ssh block (host, port, remote_port, python,
    matformer root, start + tunnel flags)."""

    def __init__(self, parent, current=None):
        super().__init__(parent)
        self.title("Server")
        self.geometry("440x360")
        self.transient(parent)
        self.result = None
        cur = current or {"name": "", "url": "127.0.0.1:8765"}
        ssh = cur.get("ssh") or {}
        form = ttk.Frame(self)
        form.pack(fill="both", expand=True, padx=12, pady=12)
        form.columnconfigure(1, weight=1)
        fields = [("name", cur["name"]), ("url", cur.get("url", "")), ("key", cur.get("key", "")),
                  ("ssh.host", ssh.get("host", "")),
                  ("ssh.remote_port", str(ssh.get("remote_port", 8765))),
                  ("ssh.python", ssh.get("python", "python")), ("ssh.root", ssh.get("root", ""))]
        self.entries = {}
        for i, (key, value) in enumerate(fields):
            ttk.Label(form, text=key.replace("ssh.", "ssh · ") + ":").grid(row=i, column=0, sticky="w", pady=3)
            e = ttk.Entry(form)
            e.insert(0, value)
            e.grid(row=i, column=1, sticky="ew", pady=3)
            self.entries[key] = e
        self.start_var = tk.BooleanVar(value=ssh.get("start", True))
        ttk.Checkbutton(form, text="start the server remotely if nothing answers",
                        variable=self.start_var).grid(row=len(fields), column=1, sticky="w")
        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=12, pady=8)
        ttk.Button(btns, text="Save", style="Accent.TButton", command=self._save).pack(side="right", padx=4)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")

    def _save(self):
        g = lambda k: self.entries[k].get().strip()
        spec = {"name": g("name"), "url": g("url")}
        if g("key"):
            spec["key"] = g("key")
        if g("ssh.host"):
            spec["ssh"] = {"host": g("ssh.host"), "remote_port": int(g("ssh.remote_port") or 8765),
                           "python": g("ssh.python") or "python", "root": g("ssh.root") or ".",
                           "start": bool(self.start_var.get())}
        if spec["name"]:
            self.result = spec
        self.destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    sys.exit(main())
