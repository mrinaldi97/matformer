"""Launching a training: one home for what `train_model.py` used to embed — build the configs,
apply the training switches, run the trainer, expand the sweep a config declares.

There is exactly one way a training starts, `execute(spec)`, where the spec is a plain dict. The
command line builds that dict from the arguments; the server's queue builds it from a job and runs
it in a fresh process (a training owns a CUDA context, spawns DDP ranks and must be killable
without touching anything else, so a process boundary is the right one — but the *target* is this
library function, never a script and never a command line).

    python -m matformer.training.launch --cfg exp.yaml --override training.lr=3e-4
    python -m matformer.training.launch --cfg exp.yaml --sweeps       # what the sweep would run
    python -m matformer.training.launch --cfg exp.yaml --sweep 1,4    # only those two runs
    python -m matformer.training.launch --queue 10.0.0.4:8765 --cfg exp.yaml   # to a server
"""
import argparse
import gc
import json
import os
import queue
import sys
import time

import torch

from matformer.matformer_configs import MatformerConfigs
from matformer.spec_fields import CLI_TO_TRAINING_FIELD, DEVICES, RESET_CHOICES, spec_fields
from matformer.training.utils import detect_device
from matformer.utils import env_variables
from matformer import sweeps

# What a caller may set on a training, in the field language every form in this project speaks.
# The GUI's queue dialog is generated from `spec_fields()`: a new switch is declared here and
# appears in the terminal, the queue and the dialog without touching any of them.

def probe_parameters():
    """The memory probe's own knobs, taken from the method that runs them: `probe_memory` stays the
    one place a default is written, and the terminal, the queue and the GUI all follow it."""
    import inspect
    from matformer.training.matformer_trainer import MatformerTrainer
    return {name: parameter for name, parameter
            in inspect.signature(MatformerTrainer.probe_memory).parameters.items()
            if parameter.default is not inspect.Parameter.empty}


PROBE_DOCS = {
    "fill": "Fraction of the GPU's memory the probe is allowed to reach",
    "max_rows": "Hard ceiling on rows per rank; 0 means none",
    "steps": "Consecutive steps run at the winning size: confirms it holds, and times it",
    "apply": "Train at the batch size the probe found, in the same job",
    "keep_weights": "Give back the weights and optimizer state the probe's steps touched",
}
PROBE_SKIP = ("seq_len", "report_path")


def probe_fields() -> list:
    """The probe dialog, generated: name, kind, default and doc for each knob."""
    fields = []
    for name, parameter in probe_parameters().items():
        if name in PROBE_SKIP:
            continue
        default = parameter.default
        kind = "bool" if isinstance(default, bool) else \
            "float" if isinstance(default, float) else "int" if isinstance(default, int) else "text"
        fields.append({"name": name, "label": name.replace("_", " ").capitalize(), "kind": kind,
                       "default": default, "doc": PROBE_DOCS.get(name, "")})
    return fields


def probe_settings(tokens) -> dict:
    """`--probe fill=0.9 steps=5` typed the way `probe_memory` declares them."""
    parameters = probe_parameters()
    settings = {}
    for token in tokens or []:
        key, _, value = str(token).partition("=")
        if key not in parameters:
            raise SystemExit(f"--probe accepts {sorted(parameters)}, not '{key}'")
        kind = type(parameters[key].default)
        if kind is bool:
            settings[key] = value.strip().lower() in ("1", "true", "yes", "on")
        elif kind in (int, float):
            settings[key] = kind(value)
        else:
            settings[key] = value
    return settings


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Matformer model")
    p.add_argument("--cfg", action="append", metavar="YAML",
                   help="One or more YAMLs; they merge left to right.")
    p.add_argument("--override", action="append",
                   help="In-line overrides (ex. data.batch_size=48), merged left to right.")
    p.add_argument("--gpu", type=int, default=None, help="Number of GPU(s)")
    p.add_argument("--nodes", type=int, default=None, help="Number of Node(s)")
    p.add_argument("--checkpoint", help="Checkpoint to resume from (see --reset)")
    p.add_argument("--reset", help="What to reset: " + ", ".join(RESET_CHOICES))
    p.add_argument("--start-from-scratch", action="store_true",
                   help="Ignore the checkpoint's weights, advance the version")
    p.add_argument("--max-steps", type=int, help="Train one epoch over this many steps")
    p.add_argument("--ckpt-version", type=int, help="Resume a specific checkpoint version")
    p.add_argument("--compile", action="store_true", help="torch.compile the whole model")
    p.add_argument("--device", default="auto", choices=DEVICES, help="Accelerator (default: detect)")
    p.add_argument("--no-save-model", action="store_true", help="Skip model saving")
    p.add_argument("--no-save-plugins", action="store_true", help="Skip plugins saving")
    p.add_argument("--sweep", metavar="POINTS",
                   help="Only these points of the config's sweep: numbers (1-based) or name globs, "
                        "comma separated. Default: every point.")
    p.add_argument("--sweeps", action="store_true",
                   help="Print the sweep the configs declare -- every point, and what it changes -- "
                        "and run nothing")
    p.add_argument("--gpus", help="Pin the run to these devices, e.g. '2,3' (CUDA_VISIBLE_DEVICES)")
    p.add_argument("--probe", nargs="*", default=None, metavar="KEY=VALUE",
                   help="Measure the largest batch the GPUs fit and how fast it runs (knobs: fill, "
                        "max_rows, steps, keep_weights); without --then-train nothing is trained")
    p.add_argument("--then-train", action="store_true",
                   help="After probing, train at the batch size the probe found")
    p.add_argument("--queue", nargs="?", const="", metavar="HOST:PORT",
                   help="Ask a server to run this instead: its address, or nothing for a local one")
    return p


def build_configs(config_files, overrides, options: dict) -> MatformerConfigs:
    """The configs every caller ends up with: YAMLs merged left to right, then the training
    switches (only the ones actually given) on top."""
    configs = MatformerConfigs.create_configs(list(config_files or []), overrides=list(overrides or []))
    from matformer.dataclasses.training_dataclasses import TrainingConfig

    known = set(TrainingConfig.__dataclass_fields__)
    picked = {field: value for field, value in (options or {}).items()
              if field in known and value is not None}
    if picked:
        configs.load_config(stdin={"training": picked})
    return configs


def options_from_args(args) -> dict:
    """The training switches the user actually typed, translated to config fields."""
    defaults = parser().parse_args([])
    return {CLI_TO_TRAINING_FIELD[dest]: getattr(args, dest)
            for dest in CLI_TO_TRAINING_FIELD if getattr(args, dest) != getattr(defaults, dest)}


def execute(spec: dict) -> None:
    """Run what a job description asks for: one training, or every run of the sweep its configs
    declare. Everything a caller needs to say is in the dict: config files, overrides, the training
    switches, which sweep points to run, and how to save. Nothing here reads argv, and nothing here
    knows who called it.

    One configs object is built, then expanded (`MatformerConfigs.sweep_variants`): each point of
    the sweep runs on a detached copy of it, whose logs, checkpoints and saved outputs carry the
    point's name. A probe (`spec['probe']`) is run per point too, which is what makes a sweep a way
    to measure before launching: the batch size each combination fits, in one job."""
    from matformer.training.matformer_trainer import MatformerTrainer

    torch.set_float32_matmul_precision(env_variables.matmul_precision)
    if spec.get("gpus"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spec["gpus"])
    device = spec.get("device") or detect_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("device cuda, but torch sees no GPU")
    configs = build_configs(spec.get("config"), spec.get("overrides"), spec.get("options"))
    variants = configs.sweep_variants(spec.get("sweep"))
    if len(variants) > 1:
        sweeps.summarize([variant.active_sweep for variant in variants])
    for index, variant in enumerate(variants, start=1):
        if len(variants) > 1:
            print(f"\n=== Run {index}/{len(variants)} of the sweep ===  {variant.active_sweep}",
                  flush=True)
        trainer = MatformerTrainer(variant, device=device)
        apply = False
        probe = spec.get("probe")
        if probe is not None:
            # One allocation answers both questions: the biggest batch these GPUs run, and how fast it
            # goes there.
            apply = bool(probe.get("apply"))
            trainer.probe_memory(apply=apply,
                                 **{key: value for key, value in probe.items() if key != "apply"})
            if not apply:
                # Without `apply` the probe is the whole job: report the ceiling, free the GPU, leave.
                del trainer, variant
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        trainer.run(save_at_end=True, save_model=spec.get("save_model", True),
                    save_plugins=spec.get("save_plugins", False))
        del trainer, variant
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── command line ───────────────────────────────────────────────────────────────────────────────
def spec_from_args(args) -> dict:
    probe = probe_settings(args.probe) if args.probe is not None else ({} if args.then_train else None)
    if probe is not None and args.then_train:
        probe["apply"] = True
    return {"config": list(args.cfg or []), "overrides": list(args.override or []),
            "options": options_from_args(args), "device": None if args.device == "auto" else args.device,
            "save_model": not args.no_save_model, "save_plugins": not args.no_save_plugins,
            "sweep": args.sweep, "gpus": args.gpus, "probe": probe}


def _validate(args):
    if args.reset is not None:
        unknown = [one for one in args.reset.split(",") if one.strip() not in RESET_CHOICES]
        if unknown:
            raise SystemExit(f"--reset accepts {list(RESET_CHOICES)}, not {unknown}")


def _await(srv, method, params=None, timeout=60):
    """One blocking request/response over the asynchronous client: what a script wants. The client
    hands answers to its events queue and the callbacks are run from there — the GUI pumps it from its
    main loop, so a script has to pump it too, otherwise it waits for a reply that is already sitting
    in the queue and the request looks like it never came back."""
    box = {}

    def callback(result):
        box.setdefault("result", result)

    srv.call(method, params, callback)
    deadline = time.time() + timeout
    while "result" not in box and time.time() < deadline:
        try:
            _, kind, data = srv.events.get(timeout=0.1)
        except queue.Empty:
            continue
        if kind == "__result__":
            rid, result = data
            if srv.pending.pop(rid, None) is callback:
                callback(result)
        elif kind == "__dead__":
            for rid, waiting in (data or {}).items():
                if waiting is callback:
                    callback(("error", "connection lost"))
    if "result" not in box:
        raise SystemExit(f"{method}: no answer from {srv.url} within {timeout}s")
    if isinstance(box["result"], tuple):
        raise SystemExit(f"{method}: {box['result'][1]}")
    return box["result"]


def _submit(args) -> int:
    """Hand the whole thing to a server's queue and follow the job until it ends: the terminal keeps
    printing what the training prints, the server owns the process."""
    from matformer.remote.net import Server

    events = queue.Queue()
    address = args.queue or os.environ.get("MATFORMER_SERVER_ADDRESS", "127.0.0.1:8765")
    if ":" not in address:
        address += ":8765"
    srv = Server("cli", address, events, key=os.environ.get("MATFORMER_API_KEY"))
    srv.start()
    deadline = time.time() + 15
    while not srv.up and time.time() < deadline:
        time.sleep(0.2)
    if not srv.up:
        srv.shutdown()
        raise SystemExit(f"no server answering on {address}")
    job_id = _await(srv, "train.start", spec_from_args(args))["run"]
    print(f"queued as {job_id} on {address}", flush=True)
    shown, job = 0, {}
    while job.get("status") not in ("done", "failed", "cancelled"):
        text = _await(srv, "jobs.log", {"id": job_id}).get("text") or ""
        sys.stdout.write(text[shown:])
        sys.stdout.flush()
        shown = len(text)
        job = next((j for j in _await(srv, "jobs.list") if j.get("id") == job_id), job)
        time.sleep(2)
    srv.shutdown()
    print(f"\n{job_id}: {job.get('status')}", flush=True)
    return 0 if job.get("status") == "done" else 1


def main(argv=None) -> int:
    job_spec = os.environ.get("MATFORMER_JOB_SPEC")
    if job_spec:
        # A server job enters here (also when Fabric re-executes this module for each rank).
        execute(json.loads(job_spec))
        print("Training complete.  Bye.")
        return 0
    args = parser().parse_args(argv)
    _validate(args)
    if args.sweeps:
        # What the configs declare, without running any of it: the plan, one line per point.
        configs = build_configs(args.cfg, args.override, options_from_args(args))
        sweeps.summarize(configs.sweep_points(args.sweep))
        return 0
    if args.queue is not None:
        return _submit(args)
    execute(spec_from_args(args))
    print("Training complete.  Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
