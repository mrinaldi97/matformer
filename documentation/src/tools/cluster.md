---
title: Cluster
sources:
  - matformer/slurm.py
  - matformer/slurm.toml
---
# Cluster

`matformer/slurm.py` provides `SlurmCommands`, the interface used by the command line, the server and the GUI to submit and monitor jobs. A job directory contains `job.sh`, `job.out`, `spec.json`, `control`, `status.json` and `saved.json`.

The module covers cluster state, allocations, environments, submission, cancellation, requeue, logs, status and diagnostics. On a host without SLURM, calls return empty tables and a note instead of raising.

Settings are read from `matformer/slurm.toml` and from `$CONFIG_ASSETS/slurm.toml`. The command line is `python -m matformer.slurm <line>`, and `python -m matformer.slurm run <spec.json>` is the entry point used inside an allocation.

The method list is generated in [[tools/commands | Commands]].
