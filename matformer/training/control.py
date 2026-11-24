# matformer/training/control.py
"""Talking to a training that is already running.

A mailbox file plus a heartbeat file, both on the shared filesystem: whoever supervises the run (the
SLURM wrapper, the server, a person from a terminal) writes a command, and every rank reads the same
command at its next optimizer step, which is what makes a distributed checkpoint safe. No socket is
opened inside the training, so nothing here depends on who launched it or on the network holding.

The heartbeat is rank 0's only: step, epoch, host, memory. It is how a server on another machine
describes a run it does not own."""
import json
import os
import signal
import socket
import time
from pathlib import Path

BEAT_SECONDS = 5.0


class Control:
    def __init__(self, mailbox: str = "", heartbeat: str = ""):
        self.mailbox = Path(mailbox or os.environ.get("MATFORMER_CONTROL", "") or "")
        self.heartbeat = Path(heartbeat or os.environ.get("MATFORMER_STATUS", "") or "")
        self.stop = False
        self.reason = ""
        self.step = None
        self._seen = ""
        self._pending = None
        self._beat = 0.0

    @property
    def active(self) -> bool:
        return bool(str(self.mailbox))

    def install_signals(self):
        """SIGUSR1 is what the SLURM wrapper sends when the time slot is nearly over; SIGTERM is what
        it sends when the job is being taken away. Both mean: leave a checkpoint behind. Installed
        after Fabric.launch(), so this handler is the one the ranks actually run, not Lightning's own
        requeue machinery, which cannot know about the framework's checkpoint format."""
        for number in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", None)):
            if number is None:
                continue
            try:
                signal.signal(number, lambda signum, _: self._interrupted(signum))
            except (ValueError, OSError, AttributeError):
                pass                      # not the main thread, or the platform has no such signal

    def _interrupted(self, signum):
        self._pending = {"cmd": "save_and_stop", "at": f"signal-{signum}-{time.time()}",
                         "reason": signal.Signals(signum).name}

    def command(self) -> dict:
        """The next unread command, from the mailbox or from a signal. Each rank tracks what it has
        already consumed, so one write is one command for everybody."""
        pending, self._pending = self._pending, None
        if pending:
            return pending
        if not self.active or not self.mailbox.is_file():
            return {}
        try:
            payload = json.loads(self.mailbox.read_text())
        except (OSError, ValueError):
            return {}
        stamp = payload.get("at")
        if not stamp or stamp == self._seen:
            return {}
        self._seen = stamp
        return payload

    def poll(self, trainer) -> None:
        """Call once per optimizer step: consumes a command if one arrived, then beats."""
        self.step = trainer.step
        payload = self.command()
        if payload:
            self.execute(trainer, payload)
        self.beat(trainer, force=bool(payload))

    def execute(self, trainer, payload: dict):
        cmd, step, epoch = payload.get("cmd"), trainer.step, trainer.epoch
        if cmd == "save":
            self._save(trainer, step, epoch, reason=payload.get("reason", "requested"))
        elif cmd == "save_and_stop":
            self._save(trainer, step, epoch, reason=payload.get("reason", "stop"))
            self.stop, self.reason = True, payload.get("reason", "stop")
            print(f"[control] stopping: {self.reason}", flush=True)
        elif cmd == "stop":
            self.stop, self.reason = True, payload.get("reason", "stop")
        elif cmd == "set":
            self.apply(trainer, payload.get("field", ""), payload.get("value"))
        else:
            print(f"[control] unknown command: {cmd}", flush=True)

    def _save(self, trainer, step: int, epoch: int, reason: str = ""):
        what = getattr(trainer.configs.training, "save_what", "model")
        path = trainer.save(step=step, epoch=epoch, what=what)
        marker = (self.heartbeat.parent / "saved.json") if str(self.heartbeat) else None
        if marker is not None and _is_zero(trainer):
            _write(marker, {"step": step, "epoch": epoch, "reason": reason,
                            "path": str(path) if path else "",
                            "at": datetime_now()})
        return path

    def apply(self, trainer, path: str, value):
        """Change a live value. Only what can be changed without rebuilding the run."""
        key = str(path).split(".")[-1]
        try:
            if key == "lr":
                rate = float(value)
                for optimizer in trainer.optimizers:
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                print(f"[control] lr set to {rate}", flush=True)
            elif key in ("max_steps", "total_steps"):
                trainer.total_steps = int(float(value))
                trainer.configs.set_override(f"training.max_steps={trainer.total_steps}")
                print(f"[control] max_steps set to {trainer.total_steps}", flush=True)
            elif key == "save_every_n_steps":
                trainer.configs.training.save_every_n_steps = int(float(value))
                print(f"[control] checkpointing every "
                      f"{trainer.configs.training.save_every_n_steps} steps", flush=True)
            elif key in ("accumulate_grad_batches", "validate_every_n_steps"):
                trainer.configs.training[key] = int(float(value))
            else:
                trainer.configs.set_override(f"{path}={value}")
        except (TypeError, ValueError) as problem:
            print(f"[control] cannot set {path}={value}: {problem}", flush=True)

    def beat(self, trainer, force: bool = False):
        if not str(self.heartbeat) or not _is_zero(trainer):
            return
        if not force and time.time() - self._beat < BEAT_SECONDS:
            return
        self._beat = time.time()
        _write(self.heartbeat, {"job": os.environ.get("MATFORMER_JOB", ""),
                                "host": socket.gethostname(), "pid": os.getpid(),
                                "step": trainer.step, "total": trainer.total_steps,
                                "epoch": trainer.epoch, "memory": _memory(),
                                "at": datetime_now()})


CHANNEL = None


def channel() -> Control:
    """The one Control of this process, dormant when no mailbox was configured."""
    global CHANNEL
    if CHANNEL is None:
        CHANNEL = Control()
    return CHANNEL


def poll(trainer) -> bool:
    """What the training loop calls. True when the run should stop."""
    control = channel()
    if control.active or control._pending:
        control.poll(trainer)
    elif str(control.heartbeat):
        control.beat(trainer)
    return control.stop


def _is_zero(trainer) -> bool:
    fabric = getattr(trainer, "fabric", None)
    return bool(fabric.is_global_zero) if fabric is not None else True


def _memory() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        to_mib = lambda value: round(value / 2 ** 20)
        return {"allocated": to_mib(torch.cuda.memory_allocated()),
                "reserved": to_mib(torch.cuda.memory_reserved()),
                "peak": to_mib(torch.cuda.max_memory_allocated())}
    except Exception:
        return {}


def _write(path: Path, payload: dict):
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, default=str))
        os.replace(temporary, path)
    except OSError:
        pass


def datetime_now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")
