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
        raw_mailbox = mailbox or os.environ.get("MATFORMER_CONTROL", "")
        raw_heartbeat = heartbeat or os.environ.get("MATFORMER_STATUS", "")
        self.mailbox = Path(raw_mailbox) if raw_mailbox else None
        self.heartbeat = Path(raw_heartbeat) if raw_heartbeat else None
        self.stop = False
        self.paused = False
        self.reason = ""
        self.step = None
        self._seen = ""
        self._pending = None
        self._beat = 0.0

    @property
    def active(self) -> bool:
        return self.mailbox is not None

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
        result = None
        if payload:
            result = self.execute(trainer, payload)
        self.beat(trainer, force=bool(payload))
        if result is not None:
            self._write_result(trainer, result)

    def wait(self, trainer) -> None:
        """Block at a clean step boundary while the run is paused.

        Commands keep being consumed, so a paused training can be resumed, inspected, asked for an
        extra analysis, or stopped; nothing advances (no forward, no optimizer) until ``resume``.
        """
        while self.paused and not self.stop:
            time.sleep(1.0)
            self.poll(trainer)

    def execute(self, trainer, payload: dict):
        cmd, step, epoch = payload.get("cmd"), trainer.step, trainer.epoch
        if cmd == "save":
            return {"cmd": cmd, "path": str(self._save(trainer, step, epoch,
                                                       reason=payload.get("reason", "requested")))}
        if cmd == "save_and_stop":
            path = self._save(trainer, step, epoch, reason=payload.get("reason", "stop"))
            self.stop, self.reason = True, payload.get("reason", "stop")
            print(f"[control] stopping: {self.reason}", flush=True)
            return {"cmd": cmd, "path": str(path)}
        if cmd == "stop":
            self.stop, self.reason = True, payload.get("reason", "stop")
            print(f"[control] stopping: {self.reason}", flush=True)
            return {"cmd": cmd, "stopped": True}
        if cmd == "pause":
            self.paused = True
            print("[control] paused at the next step boundary", flush=True)
            return {"cmd": cmd, "paused": True}
        if cmd == "resume":
            self.paused = False
            print("[control] resumed", flush=True)
            return {"cmd": cmd, "paused": False}
        if cmd == "set":
            return {"cmd": cmd, "field": payload.get("field"),
                    "applied": self.apply(trainer, payload.get("field", ""),
                                          payload.get("value"))}
        if cmd == "flush":
            getattr(trainer.logger, "drain", trainer.logger.flush)()
            return {"cmd": cmd, "flushed": True}
        if cmd in ("task", "run_task", "eval", "evaluate"):
            try:
                summary = trainer.run_task(payload)
            except Exception as problem:      # a bad ad-hoc task must not end a healthy training
                import traceback
                traceback.print_exc()
                return {"cmd": cmd, "task": payload.get("name") or payload.get("type"),
                        "error": f"{type(problem).__name__}: {problem}"}
            return {"cmd": cmd, "task": payload.get("name") or payload.get("type"),
                    "summary": summary}
        print(f"[control] unknown command: {cmd}", flush=True)
        return {"cmd": cmd, "error": f"unknown command: {cmd}"}

    def _write_result(self, trainer, result: dict) -> None:
        """Leave the answer to the last command next to the heartbeat, for RPC readers."""
        if not _is_zero(trainer):
            return
        if self.heartbeat is not None:
            path = self.heartbeat.parent / "control_result.json"
        elif self.mailbox is not None:
            path = self.mailbox.parent / "control_result.json"
        else:
            return
        _write(path, {"result": result, "step": trainer.step, "epoch": trainer.epoch,
                      "at": datetime_now()})

    def _save(self, trainer, step: int, epoch: int, reason: str = ""):
        what = getattr(trainer.configs.training, "save_what", "model")
        path = trainer.save(step=step, epoch=epoch, what=what)
        marker = (self.heartbeat.parent / "saved.json") if self.heartbeat is not None else None
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
            elif key == "weight_decay":
                decay = float(value)
                for optimizer in trainer.optimizers:
                    for group in optimizer.param_groups:
                        group["weight_decay"] = decay
                print(f"[control] weight_decay set to {decay}", flush=True)
            elif key in ("max_steps", "total_steps"):
                trainer.total_steps = int(float(value))
                trainer.configs.set_override(f"training.max_steps={trainer.total_steps}")
                print(f"[control] max_steps set to {trainer.total_steps}", flush=True)
            elif key == "save_every_n_steps":
                trainer.configs.training.save_every_n_steps = int(float(value))
                print(f"[control] checkpointing every "
                      f"{trainer.configs.training.save_every_n_steps} steps", flush=True)
            elif key in ("accumulate_grad_batches", "validate_every_n_steps",
                         "log_speed_every_n_steps"):
                trainer.configs.training[key] = int(float(value))
            elif key == "gradient_clip_val":
                trainer.configs.training.gradient_clip_val = float(value)
            elif key == "save_what":
                trainer.configs.training.save_what = str(value)
            else:
                trainer.configs.set_override(f"{path}={value}")
        except (TypeError, ValueError) as problem:
            print(f"[control] cannot set {path}={value}: {problem}", flush=True)
            return False
        return True

    def beat(self, trainer, force: bool = False):
        if self.heartbeat is None or not _is_zero(trainer):
            return
        if not force and time.time() - self._beat < BEAT_SECONDS:
            return
        self._beat = time.time()
        payload = {"job": os.environ.get("MATFORMER_JOB", ""),
                   "host": socket.gethostname(), "pid": os.getpid(),
                   "step": trainer.step, "total": trainer.total_steps,
                   "epoch": trainer.epoch, "memory": _memory(),
                   "paused": self.paused, "stop": self.stop,
                   "at": datetime_now()}
        loss = _metric(getattr(trainer, "_last_loss", None))
        if loss is not None:
            payload["loss"] = loss
        speed = getattr(trainer, "_last_speed", None)
        if isinstance(speed, dict):
            payload["speed"] = {key: value for key, value in speed.items()
                                if isinstance(value, (int, float))}
        schedulers = getattr(trainer, "schedulers", None) or []
        if schedulers:
            try:
                payload["lr"] = float(schedulers[0].get_last_lr()[0])
            except (AttributeError, IndexError, TypeError):
                pass
        _write(self.heartbeat, payload)


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
    elif control.heartbeat is not None:
        control.beat(trainer)
    return control.stop


def _is_zero(trainer) -> bool:
    fabric = getattr(trainer, "fabric", None)
    return bool(fabric.is_global_zero) if fabric is not None else True


def _metric(value):
    """A tensor/float from the training converted for the heartbeat without keeping the graph."""
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


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
