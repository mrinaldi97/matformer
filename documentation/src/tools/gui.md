---
title: GUI
sources:
  - matformer/gui/main.py
  - matformer/gui/config/app.toml
  - matformer/gui/taskpage.py
  - matformer/gui/runspage.py
  - matformer/gui/analysis.py
  - matformer/gui/jobspage.py
  - matformer/gui/clusterpage.py
  - matformer/gui/internalspage.py
  - matformer/gui/storebar.py
  - matformer/gui/activationpage.py
  - matformer/gui/steeringpage.py
  - matformer/gui/wizardpage.py
---
# GUI

`python -m matformer.gui` opens the desktop console. One window, one active server (switchable
from the top bar), pages grouped in the navigator:

* **Work**: Wizards, File Manager, Config Editor, Training (queue configs), Jobs (every queued
  job, log, abort, control menu), Cluster (SLURM queue, jobs, nodes, GPUs, partitions, envs).
* **Tasks**: one page per registered task type, generated from the task dataclass. Same model
  session across pages; each page keeps its transcript. *Table* holds numbers, *Analysis* turns
  task summaries into histograms/scatter/outliers, *Transcript* the interactive answer.
* **Measure**: Runs & Plots (panels over the run database, task results, metrics analysis,
  config diff, delete).
* **Data**: MDAT (linked containers and datasets), Favourites.
* **Server**: Server, Commands, Machine, Internals (sockets, plugins, weights), Stores ·
  activations, Stores · vectors, Registry.
* **Local**: connections and the shared run catalog.

The layout, pages, menus and themes are declared in `matformer/gui/config/app.toml`; menus are
widget-based because a native Tk menubar crashes on some X servers. Dialogs are content-sized and
centered; tables are sortable and zebra-striped; Tk-only plots avoid a matplotlib dependency.

Server pages always show which server they read (`server: name` under the title). A task page also
warns when a *runtime* plugin (mounted after load) is active. The model chip in the top bar is
global: loading or unloading a model changes the session for every task page.

See [[tools/analysis | Analysis, vectors and steering]] for the analysis view, the collect
wizard, activation/vector stores and steering; [[tools/wizards | Wizards]] for the declarative
forms.
