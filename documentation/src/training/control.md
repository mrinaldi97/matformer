---
title: Control
sources:
  - matformer/training/control.py
  - matformer/slurm.py
---
# Control

`Control` lets an external process send commands to a running training through the filesystem.

## Files

The mailbox is a JSON file. A supervisor writes a command into it; every rank reads the command at its next optimizer step. Reading the same command on every rank keeps a distributed checkpoint consistent.

The heartbeat is written by rank 0. It contains the step, the epoch, the host and the memory. A server can use it to describe a run on another machine.

The paths are passed to `Control` or read from `MATFORMER_CONTROL` and `MATFORMER_STATUS`. Without a mailbox, the control channel is inactive.

## Commands

| `cmd` | Effect |
|-------|--------|
| `save` | save a checkpoint |
| `save_and_stop` | save a checkpoint, then stop after the step |
| `stop` | stop after the step |
| `set` | change a training field (`{"cmd": "set", "field": "lr", "value": 1e-4}`) |

Each command has an `at` timestamp. Each rank remembers the last timestamp it consumed, so one write is one command.

## Signals

`install_signals()` handles `SIGTERM`, `SIGINT` and `SIGUSR1`. The SLURM wrapper sends `SIGUSR1` before the time slot ends and `SIGTERM` when the job is stopped. Both requests save a checkpoint at the next optimizer step. The handler is installed after `Fabric.launch()`.

## Loop

The trainer calls `control.poll(self)` once per optimizer step. The method executes one pending command and writes the heartbeat. `save` and `save_and_stop` call `trainer.save(...)`. After a save, a `saved.json` marker is written next to the heartbeat.

## Filesystem transport

The mailbox and the heartbeat are plain files on the shared filesystem. No socket is opened inside the training, so the channel works with a local run, a SLURM job or a server on another machine.
