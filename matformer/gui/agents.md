# Matformer Console — agents guide

Read fully before touching anything. This file encodes every decision the owner made across many
iterations. When in doubt: fewer lines, reuse what exists, never leak internals into user-facing text.
If you (a future agent) are low on context: update the TODO section honestly before you stop.

## Layout & layering

    matformer/server/    NDJSON-RPC server (asyncio TCP). Renderer-agnostic, no Tk ever.
                         python -m matformer.server [--port N --host H --root P ...]   (canonical launch)
                         Also auto-started by matformer_registry.py ("always there" decision; opt out
                         MATFORMER_SERVER=0; port/key via MATFORMER_SERVER_PORT / MATFORMER_API_KEY).
                         server.toml: host port name api_key roots run_roots fav_file.
                         core.py = @rpc decorator registry + Conn (push/stop/authed) + dispatch in threads
                         + broadcast + _encode (Table → asdict) + FAILURES ring (rpc.failures: every
                         command that raised, traceback included). Commands live in sibling modules,
                         imported once in Server.serving(). Handlers are plain f(conn, params) — that is
                         the ONLY contract, which is what makes them callable locally too (remote/local.py).
    matformer/remote/    Client layer. No Tk, no server imports (protocol only):
                         net.py    Server(name,url,events,key,trace): threaded reader, auto-reconnect,
                                   call() buffers while down and flushes on connect, trace() sees every
                                   command/answer/event (feeds the Console tab).
                         ssh.py    optional paramiko: reachable() probe → start_remote() launches
                                   `python -m matformer.server` far side → Tunnel (direct-tcpip) →
                                   ensure() returns (host,port) to dial. Plain urls bypass all this.
                         __init__.connect(cfg) single entry point.
    matformer/utils/     Framework-level, UI-free, reusable by TUI/GUI/server/scripts:
                         table.py    Table(columns, rows): str() → aligned TUI text; asdict() →
                                     {"__table__":…} JSON the GUI draws as a real table. ONE object,
                                     two renderings — the canonical shape for all command output.
                         schema.py   dataclass ↔ field language: kind_of (Literal→choice, bool/float/int,
                                     path heuristic is_path_field, json fallback), default_of, field_docs
                                     (#-comment extraction over the MRO), fields_schema(cls). Shared by
                                     task console, task.fields, config.read, override catalog, launch.
                         metrics.py  ema, glob-select, last, downsampling (plot math for any renderer).
                         catalog.py  sqlite ~/.matformer/catalog.db run catalog. replace_server (a
                                     server mirrors its own rows), upsert (absorb another machine's
                                     knowledge), forget (drop one remembered run), as_rows.
    matformer/training/launch.py    THE way a training starts. `execute(spec)` takes a plain dict
                         (config, overrides, options, device, gpus, save_model, sweep) and runs it —
                         expanding the sweep its configs declare, one run per point; `main(argv)` is
                         the CLI that builds the same dict (`--sweeps` prints the plan); `--queue
                         HOST:PORT` submits to a server and tails the job.
                         train_model.py is a two-line forwarder to this module. NEVER spawn a
                         repository script from the server: the queue spawns this function.
    matformer/sweeps.py  THE sweep system. A pipe inside a config value declares its candidates
                         (`lr: 2e-3 | 1e-4`, `\|` for a literal one); MatformerConfigs collects them
                         in `.sweeps`, keeps the first as its own value, and `sweep_variants()` is
                         the product — one detached config per point, each with the artifact names
                         its config classes declare `_sweep_scoped` (run_name, checkpoint_name,
                         store files) renamed after the point, so points never overwrite each other. `checkpoint` entries expand a folder of saved models
                         into runs too (CheckpointDirectoryManager.select). docs/sweeps.md.
    matformer/slurm.py   THE cluster door, and the only place that knows Slurm exists. One module:
                         parsed state (queue/nodes/partitions/gres/shapes), venvs (a venv is chosen by
                         the *node's* architecture, read out of a compiled .so — never guessed), the
                         communication environment, sbatch generation, lifecycle (submit/cancel/requeue/
                         hold/tail/status), preflight + doctor, and `SlurmCommands`, the interpreter the
                         server, the GUI, the terminal and launch all drive. `python -m matformer.slurm
                         run <spec>` is what runs inside the allocation; `<line>` is the console.
                         Settings: slurm.toml here, overridden by $CONFIG_ASSETS/slurm.toml. Every Slurm
                         call degrades to empty tables + a note in cluster(), never a traceback. No Tk,
                         no server imports, no job is ever described in terms of a shell command.
    matformer/training/control.py   File mailbox + heartbeat between a supervisor and a training that is
                         already running. The trainer calls `control.poll(self)` once per optimizer step:
                         save / save_and_stop / stop / set(lr, max_steps, save_every_n_steps). Every rank
                         reads the same file, so a checkpoint asked for at the end of a time slot is
                         collective and safe. SIGUSR1 (the wrapper's "your slot is ending") and SIGTERM
                         mean the same thing: leave a checkpoint behind. Dormant when no mailbox is set,
                         so nothing here costs a local run anything.
    matformer/gui/       Tk renderer ONLY. python -m matformer.gui
                         main.py      shell: nav, pages, menus, dock (Jobs/Log/Console), palette,
                                      command palette, preferences, event pump, traffic plumbing.
                         panels.py    the visualization abstraction: a panel is a JSON dict
                                      (kind/metrics/split/agg/xaxis/smooth/span/…) rendered by a
                                      registry (line, scatter, bar, histogram, table) through a
                                      Context (runs, series, keys, config, rpc). Interpretability
                                      attaches here: new kind + new source, grid/menus/export/views free.
                         clusterpage.py the submission host's view of the cluster: jobs, queue, nodes,
                                      GPUs, node sets, partitions, environments — all of them tables the
                                      interpreter produced — plus per-job actions (checkpoint, stop,
                                      requeue, cancel, boost, log, node address) and Doctor. Slurm is
                                      never named here: only command names are.
                         traffic.py   Console: bounded, off by default, exportable server traffic log.
                         commands.py  rpc.list reference + server-side tracebacks (rpc.failures).
                         local.py     this machine: connections (connect/disconnect/current), run
                                      catalog browser (forget, CSV), console switch.
                         pages: explorer (file manager + the one picker), configpage, trainpage
                                      (+QueueDialog), runspage, mdatpage, taskpage, serverpage,
                                      syspage, regpage, favspage; shared: widgets.py, forms.py,
                                      theme.py, icons.py, override.py.
    matformer/training/sqlite_logger.py  default logger backend "sqlite": writes the run database
                         (matformer/utils/run_db.py: runs, parameters, metrics, task_summaries,
                         task_records), batched with WAL. The server and the GUI read that file
                         (runs.*, jobs._monitor) — the database IS the API for history.
    interfacciaclaude/   old Qt/Tk prototypes, kept as design reference until parity. w.py/tk.py dead.

## Wire protocol

    →  {"id":7,"method":"fs.list","params":{...}}
    ←  {"id":7,"result":…} | {"id":7,"error":"Type: msg"}
    ←  {"event":"token|metric|job|slurm_job|task_state|auth|error","data":{…}}
Auth: if server api_key set, first line must be {"auth":key} — per-connection check, ZERO per-call
overhead. Production transport: localhost bind + ssh -L (GUI can bootstrap tunnel+start itself).
**An error response still runs the caller's callback**, with the tuple ("error", text). Every page
therefore always un-blocks: a failed or lost request cannot leave a button stuck (this was bug #1).
Pages must keep testing `isinstance(result, tuple)`. A dead socket flushes its pending callbacks the
same way ("connection lost"). Errors are one-line red toasts + full text in the Log; the server keeps
tracebacks in core.FAILURES (rpc.failures) shown by the Commands page.

## Command surface (rpc.list is the truth; GUI must never hardcode knowledge)

fs.* (jailed roots, list/read/write/rename/delete/mkdir) · fav.* (models/datasets, one JSON) ·
sys.info/mem/gc · reg.categories/info · rpc.list · **rpc.failures** (ring of raised commands,
{index} → traceback, {clear:true} → empty) · stop ·
runs.list/config/metrics/watch/unwatch (watch polls the run database → live "metric" events, same message
shape as history replay) · **catalog.list / catalog.merge / catalog.upsert** ·
train.start(spec-shaped params)/train.abort/**train.options** (the launcher's own switches, so the
queue dialog is generated, not hardcoded) · jobs.list/watch/log ·
task.* full TaskSession driver (load/state/use/fields/set/feed/stop/run/annotate/info/status/clear/
context/system/template/unload/**probs**/**details**) — one server-wide session; "task_state"
broadcast keeps every connected GUI in sync ·
**slurm.methods/call/line/fields/watch** — the SLURM interpreter published the same way MDAT's facade
is: `slurm.fields` returns the allocation fields (partition, nodes, gpus, gpu model, venv, time, with
the live choices of this cluster) so the queue dialog is generated, `slurm.call` runs one operation,
`slurm.line` takes what a person would type, `slurm.watch` replays and follows cluster jobs
("slurm_job" events). A job submitted with `backend=cluster` goes through the interpreter and comes
back in `jobs.list` with the local ones; `jobs.log`/`train.abort` on such an id are forwarded to the
cluster (abort checkpoints on the way out). Zero Slurm code in the GUI, zero GUI code in slurm.py ·
mdat.methods/call/line/**start** — the SAME MDatCommands facade the cmd2 TUI drives; `mdat.start`
queues a long operation (ingest, pretok, precompute, export) as a job in its own process
(matformer.data.mdat.tui:execute) with log + status + abort ·
config.new/read/save/write/derive/delete_section/sections/add_entry/fields_catalog/**home** — ALL
config understanding goes through MatformerConfigs (includes, refs, defaults) + registry dataclass
introspection. Never parse a config by hand anywhere else.
**wizard.list/schema/plan/submit/context** — TOML wizard definitions (matformer/wizards), renderer
agnostic. Fields come from the same schema language; config steps are generated from registered
dataclasses with focused/advanced presets; actions queue the same train.start / mdat.start paths,
or the annotation job. The GUI dialog is a renderer only.

## Design principles (enforced — the owner will check)

1. DRY absolute. One field language (schema.py ↔ forms.py ↔ task.fields ↔ config.read ↔ launch
   spec_fields). One Table type (TUI str / GUI asdict). One payload router (widgets.show_payload).
   One picker (explorer.pick, used by every path field everywhere). One context-menu mechanism
   (widgets.popup, one grab ever). One way to start a training (training/launch.execute). Server must
   reuse MatformerConfigs/registry — zero duplicated config logic.
2. Renderer independence. server/, remote/, utils/ NEVER import tkinter. A new feature = server
   command first, thin GUI second. GUI pages are transport+render only; state lives server-side
   (session, jobs, catalog) and is mirrored via events.
3. Server-wide vs per-page truth. The model session and the job queue are SERVER state: any page can
   load/unload/use; task_state/job broadcasts keep all pages (and all GUIs) honest. The global server
   pill is the single server selection (app.server / server_var); pages react to server_changed().
   LOCAL matters (connections, run catalog, console switch) live on the Local page, never in Server.
4. Registry-driven everything. Tasks group in the navigator = local registry. Task/config/launch/
   MDAT forms are generated server-side from dataclasses and signatures (task.fields,
   config.fields_catalog, train.options, mdat.methods). Adding a task type, config field or MDAT
   operation must require ZERO GUI changes.
5. Low LOC, terse code, comments only for *why*. No emojis. No defensive boilerplate that can't fail.
6. Resilience: every callback guarded; every lazy load TclError-safe; client buffers calls while
   down; **every request has a path back to its caller, success or not**. The GUI must never be able
   to hang waiting on a server: watchdogs release busy states, Stop always works.
7. **UI voice (owner is allergic to slop).** State a fact or say nothing. NEVER leak implementation
   details or instructions into labels/subtitles/empty-states. "no model loaded", "no jobs running on
   any server", "no container open". Power user is the target: menus/shortcuts/panels over
   hand-holding. Dialogs never *do* something on close — accepting is an explicit button.
8. Late-90s discipline, 2020s cleanliness: dense, hierarchical (nav, menus, tabs, context menus,
   command palette Ctrl+P), everything reachable by keyboard; flat modern skin (NOT retro bevels):
   tokens in app.toml [themes], elevation by tone not border, Accent/Default/Ghost/Danger button
   tiers, focus rings, slim auto-hiding scrollbars, zebra tables, right-aligned numerics, text
   progress bars ▰▱, status dots, tooltips, empty-states, collapsible dock.
9. **Tables fit their content** (widgets.TableList.fit, called on every fill): ids and numbers stay
   narrow, prose columns take the room. Never stretch column 0 by default.
10. **The user decides cost/benefit knobs.** Token-level detail (topk_present / save_probs) defaults
    to fast; off / 5 / 25 / -1 (whole vocabulary) is one menu or one field away, never decided for
    him, never silently forced at load.

## Sync model (owner decision)

SYNCED via ~/.matformer/catalog.db (sqlite, client-side): the RUN CATALOG (id, server, project, run,
version, host, started, dir, steps). Conflict-free: each server owns its rows (replace_server
wipes+reinserts → deletions propagate). do_sync_catalog pulls every connected server's runs, then
pushes the merged snapshot back with catalog.upsert (additive, each row under its own server).
Offline runs stay browsable (metadata; metrics/logs when that server returns). Local page can forget
single runs and export the catalog as CSV.
NOT synced (by design): live jobs, session state, filesystem, machine info, registry.
Log import into catalog: STUB ONLY (owner plans switching jsonl logging → sqlite first).

## Testing

    python -m py_compile matformer/gui/*.py matformer/server/*.py matformer/remote/*.py matformer/utils/*.py
    python -m matformer.server &                       # then drive it from a client
    xvfb-run -a env MATFORMER_SERVER=0 PYTHONPATH=$PWD python /tmp/.../harness.py
    # harness rules learned the hard way: os._exit() from callbacks; winfo_children() takes NO args
    # (recurse manually); call App() AFTER registry import; pythonw swallows stdout → write to a file;
    # keep waits ≥1.5s for the first heavy config.read.
    # GPU is SHARED: the owner has approved loading checkpoints/token_classification/pos and
    # hf://Qwen/Qwen3-0.6B for verification. Use flash attention — it is fine on this box.
    # Raw-socket wire-test pattern: makefile("rwb") + one JSON per line over :8765.

## Verified working (this session, GPU + wire + Xvfb)

Backend (wire/LocalServer, pos ckpt on cuda + cpu fallbacks, flash attention):
- **evaluation and masking no longer crash** ("Dimension out of range [-3,2] got 3" was
  flash_kv calling the 4-dim flash entry point on packed varlen batches → now routes to
  flash_attn_varlen_func; generation was unaffected because it always runs padded).
- masking no longer feeds vocabulary ids into a class head (was a CUDA device-side assert in
  log_softmax.gather, also reproducible from the CLI); scoring and rendering adapt to the head.
- _cells/_name index correctly per position; cells carry `index`; task.probs answers the exact
  position (verified: every cell's label matches its detail), with stats
  {position, classes, predicted, probability, margin, entropy, gold, gold_probability, correct}.
- detail knob verified at 0 / N / -1 for evaluation, masking and generation (generation keeps
  per-step top-k candidates when topk_present != 0; task.details returns the whole table).
- presentation regression fixed: the friendly token/label/p table no longer depends on topk being
  on; surface forms show ▁ instead of Ġ.
- train queue end-to-end: train.start → spawn → launch.execute → jsonl run discovered by runs.list,
  job events starting→running→done, jobs.log tail, abort path unchanged. train.options returns the
  generated field list.
- mdat over the wire: create → add → ingest → open → ls_datasets → peek (offset pages in order;
  random only when no offset) → doc → views → strategies. mdat.methods now reports all 40 facade
  operations (bound methods are not functions — instance introspection returned 2).
GUI (Xvfb): 16 pages boot clean, zero page errors across a full page sweep; config editor renders
model/tokenizer/data.train/training/file tabs (A1 done); dock is a PanedWindow pane — toggling keeps
it at the bottom and restores its height; Console records and flushes traffic; MDAT page auto-lists
datasets/views/strategies on open; table columns fit.

Cluster (real SLURM host, mixed aarch64/x86_64; the Cluster page is compile-only — the login node has
no python3-tk, so it has not been drawn yet):
- squeue/sinfo/scontrol/sacct parsed into Tables; refused squeue columns are dropped and the query
  retried (Slurm 26.05 does not know RunTime/NCPUS/ExitCode, and the row count must match the headers).
- venvs are identified by the architecture read out of a compiled extension, not by the host running
  the query: epito/gracehopper are aarch64, cascadelake/mirri/broadwell x86_64, and one venv cannot
  serve both. `slurm pip` installs into a venv on a node that can actually run it.
- trainings ran on 1 and 2 nodes (4 DDP ranks) with flash attention; heartbeat (step/total/host/memory)
  reaches the server through the shared filesystem, no socket inside the job.
- control channel verified live: `slurm save` wrote a checkpoint mid-run; SIGUSR1 at the end of the
  slot checkpointed, exited 0, the wrapper requeued, and the requeued job resumed from that file.
- preflight catches what would otherwise be paid for: venv/partition architecture mismatch, GPUs more
  than a node has, walltime above the partition ceiling, exclusive-node partitions, missing configs.
- framework bugs found by the cluster path and fixed: a saved config carried an *implementation*
  variant in `model.type` (so no checkpoint could be resumed), `_setup_checkpoint` read
  `model.base_model` unguarded, and the model was always built in bf16 — which no V100 can train in.

## Wizards (this session)

    matformer/wizards/        TOML definitions + engine (conditions, presets, config generation,
                              declarative actions). No Tk, no server imports.
    matformer/server/wizard_cmd.py   wizard.* RPC, live MDAT providers, train/mdat/annotate submit.
    matformer/tasks/annotate_job.py  the annotation queue entry (loads its own model).
    matformer/gui/wizard.py          generic dialog: focused/advanced, choice/radio/multi,
                              path/paths/mapping, summary before queue.
    matformer/gui/wizardpage.py      browser grouped by category; menus and pages open dialogs.
    Definitions: train, annotate, mdat_create/dataset/link/ingest/pretokenize/view/prepare.
    MDAT is the linked-dataset model: a container links self-contained datasets; the legacy
    all-in-one mdat folder is not offered by wizards. Container-level strategy/view/prepare
    operations cover every linked dataset; strategies are copied into each linked manifest.
    The training page and the MDAT page keep their raw advanced controls; the wizard is the
    primary entry, not a replacement for a power user's dialogs.

    Jobs are its own page (Training only launches). Rows forward from the dock on double-click.
    The page has Save/Pause/Resume/Stop/Abort through train.command. Every page shows the
    selected server under its subtitle, and the toolbar has a Server: label before the pill.
    The model chip and Unload are global. Tools > Start local server boots a server with the
    repository venv on a free port and adds it to the pills.
    Dialogs call widgets.fit_dialog (content-sized, centered, never taller than the screen).
    SSH uses paramiko when present, otherwise the system ssh binary; remote start uses a detached
    Python Popen so ssh returns. The remote python (venv) is a server field.
    Plots display in tkplot.TkPlotCanvas with min/max decimation, hover, drag/wheel zoom and
    shared x; the Recent button ticks the newest runs and default panels are created once.

## TODO

### A — asked for, still open

A19. **Wizards on the real desktop.** Page and training dialog verified by screenshot; still
    walk every wizard, the advanced expander, dynamic MDAT choices and the summary by hand on the
    owner's screen before calling it done.
A20. **Native menu bar crash: fixed.** `root.configure(menu=...)` raises
    `X Error: BadValue (X_ConfigureWindow)` on this display with Tk 8.6 and 9.0, even in a bare
    Tk window. The bar is now a Frame of Menubutton+popup menus (`_menu_button`), with the same
    accelerators, mnemonics and themed popdowns. Screenshots on X11 workspace 10 confirm the
    wizard page and dialog render correctly under system Python (Tk 8.6). The uv Python's Tk 9
    build on this box advertises only the `fixed` family; the theme now says so explicitly instead
    of configuring a font name as a family, but the system Python remains the one with real fonts.
A2. **Slop-string sweep.** Owner's standing rule (Principle 7). Partially done this session
    (jobs/server/config/empty-states cleaned, NewConfigDialog rewritten). Still to audit: explorer
    info line, welcome tab, syspage/regpage/favspage subtitles, taskpage header hints, PrefsDialog
    helper paragraphs (the icon-theme paragraph belongs in a tooltip).
A3. **Jobs/Log/Console dock tab heights unequal.** Still true: log_console was constructed with
    parent=root then adopted into the notebook. Rebuild it with the notebook as parent and verify
    all three tabs have the same inner height.
A9. **SSH bootstrap real-world test.** Fields exist ([server.ssh] host/remote_port/python/root/start
    + key). Verify probe→start→tunnel→green pill against a real box; harden ensure() failure paths
    (auth/hostkey must toast, not hang the bootstrap thread).
A10. **Dark theme visual pass.** Owner called the previous dark "total shit": verify panels
    (runspage canvases use palette surface/zebra/muted ✓), taskpage cell shading contrast, tooltips
    (fixed dark, acceptable), icons fallbacks, QueueDialog, PanelDialog, Traffic tags, menus.
A13. **Real-desktop verification.** Everything above is Xvfb/wire-verified, not owner-verified:
    MDAT browsing ergonomics, panels grid feel, queue dialog sizing on his screen, Console overhead
    during a long training tail (it buffers 3000 lines max).
A14. **task.probs for a *sequence-level* head**: cells carry one entry with the whole text as label;
    the modal is right but the position semantics ("position 0") are worth a clearer title.
A15. **MDAT long ops from the GUI are queued, but their progress is only the log tail.** When MDAT
    grows a progress callback, add a `progress` field to job events (jobs._monitor already sets it).
A16. **Cluster page has never been drawn.** The submission host has no python3-tk, so clusterpage.py
    is compile-verified only: check tabs, per-job menus, column fit and the detail/log pane on the
    real desktop, then on a laptop with no Slurm (it must show empty tables, not an error).
A17. **`slurm attach` is untested against a live allocation.** It returns the tunnel and the command to
    start a server on the node running a job (the third layer: laptop → submission host → node);
    same gap as A9. Per-GPU VRAM inside a job can only come from that node, not from Slurm.
A18. **Slurm signals the steps, not always the batch script.** The wrapper's SIGUSR1 trap may fire only
    after `srun` returns, so requeue rests on `stopped.json` written by rank 0 rather than on the
    shell's own flag. Both paths are kept; if a cluster delivers neither, the job simply ends.

### B — agreed, deliberately future

B1. **Scripts via server**: LocalServer (remote/local.py) exists with net.Server's exact shape;
    `launch --queue [HOST:PORT]` is the first step (implemented this session, untested against a
    remote box). run_task.py is untouched; wiring it the same way (or MATFORMER_VIA_SERVER) still
    needs the owner's nod. Do NOT spawn repository scripts from the server, ever.
B2. **Security audit** (owner: future, dedicated pass). Current stance: localhost bind, optional
    api_key (connection-level), SSH tunnel is the transport, fs jailed to roots (currently "/" for
    debug — tighten server.toml back), delete/rename have no undo. Known gaps: key exchange is
    plaintext-on-loopback, no per-method ACLs, config surgery follows symlinks. The job queue
    executes framework functions with dict specs only (no shell), which is deliberate.
B3. **SQLite logging** replacing jsonl (jsonl is interim; catalog stays the seam).
B4. **Interpretability** (design ONLY, do not build yet): plugin attach/swap, steering, ablations,
    logit attribution, activation stores / virtual models. The seam is fixed: new server command
    group returning Tables/events; GUI = a new `panels.PANELS` kind (+ `source: rpc`) and/or
    context menus with on-demand heavy fetch (task.probs is the template). tasks_host's session model
    must stay ready for component swapping. Outlier-token tables and DLA graphs fit this exactly.
B5. **@register_commands / @register_gui_component sugar** — owner floated it; the @rpc decorator +
    registry introspection already cover the need. Reconsider when a second renderer appears.
B6. favs currently per-server JSON; candidate to move into the shared catalog.

## Session notes (fresh agents, read me)

Sessions before this one built the logger, server (+auth, catalog, ssh, local client), the Tk suite,
the icon layer, MDAT Table-ization, config wizard. This session fixed the owner's audit: the stuck
Stop button (transport errors never reached their callbacks), the masking/evaluation crash (a real
framework bug in flash_kv + two indexing bugs in tasks_host + masking scoring), added the Console
traffic tab + server-side failure ring, wired the interactive token detail end to end (eval,
masking, generation, with the cost/detail knob left to the user), rebuilt the config editor tabs +
raw editing + New wizard, fixed the dock pane, rebuilt the training workflow (QueueDialog with
generated switches, OverrideDialog buttons that can no longer fall off screen), rebuilt MDAT browsing
(menus instead of a button wall, signature-generated wizards, job-queued long operations, ordered
paging, auto-refresh), split Commands out of the Server page and moved local concerns to a new Local
page, and replaced the plot system with the panels abstraction (kinds, per-panel options, hover/drag
zoom, saved views, column chooser, config diff). It also removed the server→train_model.py→framework
cycle: train_model.py is now a forwarder and the queue spawns matformer.training.launch.execute.
The owner tests on a real desktop — Xvfb passes ≠ user-verified. Keep this file updated.

## Session notes: vectors, collect, analysis (this pass)

- **One task job**: `matformer.tasks.task_job` now owns annotation (`annotate`), dataset passes
  (`run`) and texts (`text`); `task.enqueue` queues any of them, and the annotate/collect wizards go
  through `server.task_jobs.enqueue_task` instead of separate entries. `annotate_job` is a thin
  compatibility re-export.
- **Collect wizard** (`collect.toml`): one forward with activation-cache/store plugins in the config;
  keep metadata, then pool per sequence into the vector store with its label. `collect.start` queues a
  config that already mounts the plugins; `collect.vectorize` re-pools an existing store.
- **Vectors are general** (`matformer/vectors/`): names, kinds (steering/embedding/prototype/probe),
  labels, address, metric, tags and metadata in an activation-store databank. `vectors.classify`
  labels by top-k cosine vote; `vectors.embed` extracts pooled vectors; steering plugin reads its
  direction from the store, so `steer.apply` no longer ships thousands of floats.
- **Analysis view** (`gui/analysis.py`): one parser for flat keys (`weights/...`, `activations/...`)
  and run metrics; table/histogram/scatter/outliers; the task page has an *Analysis* tab and Runs owns
  *Analyze metrics…*. Table/plot panes hide each other so the chart gets the room.
- **Stores**: *Stores · activations* and *Stores · vectors* share `StoreBar` (path/backend/databank
  dropdown, last store). `store.inspect` opens a file read-only without mounting it; *Internals* mounts
  plugins on sockets and edits weights. `plugin.list` reports `origin=model|runtime`; task pages warn
  only about runtime plugins (cache/store excluded because they do not change the answer).
- **Fix**: `ActivationCache` now carries `metadata` into `metadata_cache`, so contrastive labels reach
  the store; `ActivationStore.forward` skips non-batched sockets; datamodule keeps an explicit
  `max_seq_len` instead of always inheriting the model window.
