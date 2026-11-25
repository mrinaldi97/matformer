"""Dummy backend.   python server/server.py NAME PORT      (pip install websockets)

Run one per machine/GPU/SLURM node. Methods are plain async functions registered with @rpc("dotted.name");
the GUI calls them by name. A real backend replaces the bodies, not the protocol (see client/net.py).
Everything below is fake: random characters, random "probabilities", random jobs.
"""
import asyncio
import json
import math
import random
import string
import sys
import time

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

NAME, PORT = sys.argv[1], int(sys.argv[2])
RPC, JOBS, WATCHERS, rng = {}, {}, set(), random.Random(NAME)


def rpc(name):
    def register(f):
        RPC[name] = f
        return f
    return register


class Conn:
    """One connected GUI. emit() pushes an event to it at any time (full duplex)."""

    def __init__(self, ws):
        self.ws, self.stop = ws, asyncio.Event()

    async def emit(self, event, data):
        await self.ws.send(json.dumps({"event": event, "data": data}))


# ── fake inventory: any method that just returns rows can be a table page in the GUI ──────────
MODELS = rng.sample([("tiny-llama-1.1b", "1.1B"), ("qwen2.5-7b-instruct", "7B"), ("llama-3-70b", "70B"),
                     ("gemma-2-2b", "2B"), ("my-finetune-step-4200", "7B")], 3)
TABLES = {
    "models.list":   [{"name": n, "params": p, "dtype": "bf16", "path": f"/models/{n}"} for n, p in MODELS],
    "datasets.list": [{"name": n, "split": "train", "tokens": t} for n, t in [("fineweb-edu", "1.3T"), ("pile-subset", "40B")]],
    "evals.list":    [{"name": e, "model": MODELS[0][0], "score": f"{rng.uniform(.3, .8):.3f}"} for e in ("mmlu", "arc-c", "hellaswag")],
    "sae.list":      [{"name": f"{k} · L{layer}"} for k in ("res-16k", "res-65k") for layer in (12, 20)],
}
for _name, _rows in TABLES.items():
    async def _table(c, p, rows=_rows):
        return rows
    RPC[_name] = _table


# ── metrics (what a training loop would log) ──────────────────────────────────────────────────
METRICS = {}                                                  # (run, key) -> [[step, value], ...]


def sample(step):
    return {"loss": 4 * math.exp(-step / 400) + .3 + rng.gauss(0, .03),
            "lr": 3e-4 * .5 * (1 + math.cos(math.pi * min(step / 2000, 1))),
            "grad_norm": 1 + 2 * math.exp(-step / 300) + abs(rng.gauss(0, .15)),
            **{f"norm/{c}": w * (1 + step / 800) + rng.gauss(0, .05) for c, w in (("embed", 1.0), ("attn", 2.0), ("mlp", 3.0), ("unembed", 1.5))}}


async def record(run, step):
    for key, v in sample(step).items():
        METRICS.setdefault((run, key), []).append([step, v])
        await broadcast("metric", {"run": run, "project": JOBS[run]["project"], "key": key, "points": [[step, v]]})


# ── fake file tree (dict = folder, int = file size) and config schemas ─────────────────────
def _ckpt(*steps):
    return {f"step_{n}.pt": 2_100_000_000 for n in steps}


FS = {"models": {m: {"config.json": 900, "model.safetensors": 15_000_000_000, "tokenizer.json": 7_000_000} for m, _ in MODELS},
      "checkpoints": {"sae-train-L20": _ckpt(1000, 2000), "finetune-step-4200": _ckpt(4200)},
      "datasets": {"fineweb-edu": {f"shard-{i:03d}.jsonl": 500_000_000 for i in range(3)}, "pile-subset": {"train.jsonl": 9_000_000_000}},
      "configs": {"model_qwen.yaml": 1200, "model_tiny.yaml": 900, "train_sae.yaml": 800, "train_finetune.yaml": 950}}


@rpc("fs.list")
async def fs_list(c, p):
    node = FS
    for part in filter(None, p.get("path", "/").split("/")):
        node = node[part]
    rows = [{"name": k, "kind": "dir" if isinstance(v, dict) else "file", "size": 0 if isinstance(v, dict) else v} for k, v in node.items()]
    return sorted(rows, key=lambda r: (r["kind"] != "dir", r["name"]))


# Same field language as config/plugins/*.toml (+ "section", "doc"). A real backend would derive this from its dataclasses.
MODEL_FIELDS = [
    {"section": "Architecture", "name": "num_layers", "type": "int", "min": 1, "max": 256, "default": 28, "doc": "Number of transformer blocks."},
    {"name": "hidden_size", "type": "int", "min": 64, "max": 32768, "default": 3584, "doc": "Residual stream width (d_model)."},
    {"name": "num_heads", "type": "int", "min": 1, "max": 256, "default": 28},
    {"name": "max_position_embeddings", "type": "bigint", "default": 131072, "doc": "Longest supported context, in tokens."},
    {"name": "vocab_size", "type": "bigint", "default": 152064},
    {"section": "Components", "name": "activation", "type": "choice", "default": "swiglu", "doc": "Many options → dropdown.",
     "options": ["gelu", "gelu_new", "silu", "relu", "relu2", "swiglu", "geglu"]},
    {"name": "norm", "type": "radio", "options": ["layernorm", "rmsnorm"], "default": "rmsnorm", "doc": "Few options → radio."},
    {"name": "position_encoding", "type": "radio", "options": ["rope", "alibi", "learned"], "default": "rope"},
    {"name": "rope_theta", "type": "sci", "default": 1e6},
    {"name": "tie_embeddings", "type": "bool", "default": False, "doc": "Share input and output embedding matrices."},
    {"name": "dtype", "type": "radio", "options": ["float32", "bfloat16", "float16"], "default": "bfloat16"},
    {"name": "init_from", "type": "path", "root": "/checkpoints", "default": "", "doc": "Optional: initialise weights from a checkpoint."},
]
TRAIN_FIELDS = [
    {"section": "Optimisation", "name": "learning_rate", "type": "sci", "default": 3e-4},
    {"name": "min_lr", "type": "sci", "default": 3e-5},
    {"name": "optimizer", "type": "radio", "options": ["adamw", "lion", "sgd"], "default": "adamw"},
    {"name": "scheduler", "type": "choice", "options": ["cosine", "linear", "constant", "wsd", "polynomial"], "default": "cosine"},
    {"name": "weight_decay", "type": "float", "min": 0, "max": 1, "step": 0.01, "default": 0.1},
    {"name": "grad_clip", "type": "float", "min": 0, "max": 100, "step": 0.1, "default": 1.0},
    {"section": "Schedule", "name": "max_steps", "type": "bigint", "default": 200000},
    {"name": "warmup_steps", "type": "int", "min": 0, "max": 100000, "default": 2000},
    {"name": "batch_size", "type": "int", "min": 1, "max": 65536, "default": 256},
    {"name": "grad_accum", "type": "int", "min": 1, "max": 1024, "default": 4},
    {"name": "mixed_precision", "type": "radio", "options": ["no", "bf16", "fp16"], "default": "bf16"},
    {"section": "Data", "name": "dataset", "type": "path", "root": "/datasets", "default": "/datasets/fineweb-edu"},
    {"name": "resume_from", "type": "path", "root": "/checkpoints", "default": ""},
    {"name": "seed", "type": "int", "min": 0, "max": 2147483647, "default": 1234},
]
VALUES = {}


@rpc("config.read")
async def config_read(c, p):
    fields = MODEL_FIELDS if p["path"].rsplit("/", 1)[-1].startswith("model") else TRAIN_FIELDS
    return {"fields": fields, "values": VALUES.get(p["path"]) or {f["name"]: f["default"] for f in fields if "default" in f}}


@rpc("config.write")
async def config_write(c, p):
    VALUES[p["path"]] = p["values"]
    return {"msg": f"[{NAME}] saved {p['path']} ({len(p['values'])} keys)"}


# ── jobs: shared by every client of this server ───────────────────────────────────────────────
async def broadcast(event, data):
    for c in list(WATCHERS):
        try:
            await c.emit(event, data)
        except ConnectionClosed:
            WATCHERS.discard(c)


async def push(job):
    JOBS[job["id"]] = job
    await broadcast("job", job)


@rpc("jobs.watch")
async def jobs_watch(c, p):
    WATCHERS.add(c)
    for j in JOBS.values():
        await c.emit("job", j)
    for (run, key), pts in METRICS.items():                  # history, same message type as live updates
        await c.emit("metric", {"run": run, "project": JOBS[run]["project"], "key": key, "points": pts})


@rpc("jobs.cancel")
async def jobs_cancel(c, p):
    JOBS[p["id"]]["status"] = "cancelled"
    await push(JOBS[p["id"]])


async def ticker():
    """Advances the fake training jobs."""
    while True:
        await asyncio.sleep(1)
        for j in list(JOBS.values()):
            if j["kind"] != "training":
                continue
            if j["status"] == "queued" and rng.random() < .2:
                j["status"] = "running"
            elif j["status"] == "running":
                j["step"] = j.get("step", 0) + 10
                await record(j["id"], j["step"])
                j["progress"] = min(1, j["progress"] + rng.uniform(.005, .03))
                j["status"] = "done" if j["progress"] >= 1 else "running"
            else:
                continue
            await push(j)


# ── commands that plugin buttons call (stubs) ─────────────────────────────────────────────────
@rpc("models.load")
async def models_load(c, p):
    return {"msg": f"[{NAME}] would load {p.get('source')} now (stub)"}


@rpc("steering.attach")
async def steering_attach(c, p):
    return {"msg": f"[{NAME}] vector picker not implemented; current strengths: "
                   + ", ".join(f"{k[2:]}={v:+g}" for k, v in p.items() if k.startswith("v_"))}


# ── inference ─────────────────────────────────────────────────────────────────────────────────
def rand_tok():
    if rng.random() < .03:
        return "\n"
    tok = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(1, 4)))
    return " " + tok if rng.random() < .6 else tok


def describe(P):
    s, steer = P.get("sampling", {}), [f"{k[2:]}×{v:+g}" for k, v in P.get("steering", {}).items() if k.startswith("v_")]
    bits = [f"{k}={s[k]}" for k in ("temperature", "top_k", "top_p") if k in s]
    if steer:
        bits.append("steer " + ", ".join(steer))
    if "sae" in P.get("sae", {}):
        bits.append("SAE " + P["sae"]["sae"])
    if P.get("interventions", {}).get("ablation", "none") != "none":
        bits.append(f"ablate {P['interventions']['ablation']} {P['interventions']['target']}")
    return f"[{NAME}] " + " | ".join(bits)


@rpc("stop")
async def stop(c, p):
    c.stop.set()


@rpc("generate")
async def generate(c, p):
    P = p["plugins"]
    s = P.get("sampling", {})
    temp, n_max = s.get("temperature", .8), s.get("max_new_tokens", 256)
    job = {"id": f"run-{len(JOBS) + 1}", "kind": "inference", "name": p["prompt"][:28], "status": "running", "progress": 0}
    c.stop.clear()
    await push(job)
    await c.emit("start", {"note": describe(P)})
    t0, n, reason = time.monotonic(), 0, "length"
    while n < n_max:
        if c.stop.is_set() or job["status"] == "cancelled":
            reason = "stopped" if c.stop.is_set() else "cancelled"
            break
        n += 1
        pr = rng.betavariate(6, 2 + 2 * temp)                      # hotter -> less confident
        await c.emit("token", {"t": rand_tok(), "p": pr,
                               "alt": [[rand_tok(), (1 - pr) * w] for w in (.5, .3, .2)],
                               "feats": [[rng.randint(0, 16383), rng.uniform(.5, 9)] for _ in range(3)]})
        if n % 5 == 0:
            job["progress"] = n / n_max
            await push(job)
        await asyncio.sleep(.03)
    job.update(status="done" if reason == "length" else reason, progress=n / n_max)
    await push(job)
    return {"reason": reason, "n": n, "tps": n / max(time.monotonic() - t0, 1e-6)}


# ── plumbing ──────────────────────────────────────────────────────────────────────────────────
async def dispatch(c, m):
    """Every request runs as its own task, so a long generate never blocks 'stop'."""
    try:
        if m["method"] not in RPC:
            raise NotImplementedError(m["method"])
        out = {"id": m["id"], "result": await RPC[m["method"]](c, m.get("params") or {})}
    except Exception as e:
        out = {"id": m["id"], "error": f"{type(e).__name__}: {e}"}
    try:
        await c.ws.send(json.dumps(out))
    except ConnectionClosed:
        pass


async def handle(ws):
    c = Conn(ws)
    try:
        async for raw in ws:
            asyncio.create_task(dispatch(c, json.loads(raw)))
    finally:
        WATCHERS.discard(c)
        c.stop.set()


async def main():
    for i, (project, name, lr) in enumerate([("sae-sweep", "sae L20 lr=1e-4", 1e-4), ("sae-sweep", "sae L20 lr=3e-4", 3e-4), ("finetune-qwen", "finetune step-4200", 2e-5)]):
        JOBS[f"train-{i + 1}"] = {"id": f"train-{i + 1}", "kind": "training", "name": name, "project": project,
                                  "config": {"lr": lr, "seed": rng.randint(0, 999), "batch_size": 256, "dataset": "/datasets/fineweb-edu",
                                             "checkpoint_dir": f"/checkpoints/{project}/{i + 1}"},
                                  "status": "running" if i < 2 else "queued", "progress": rng.random() * .5}
    for j in JOBS.values():                                  # pre-existing history for jobs already running
        if j["status"] == "running":
            j["step"] = 300
            for st in range(0, 300, 10):
                for key, v in sample(st).items():
                    METRICS.setdefault((j["id"], key), []).append([st, v])
    async with serve(handle, "localhost", PORT):
        print(f"{NAME}: ws://localhost:{PORT}", flush=True)
        await ticker()


if __name__ == "__main__":
    asyncio.run(main())
