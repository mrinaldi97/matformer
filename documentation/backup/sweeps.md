# Sweeps

One config file, many runs, and no run able to overwrite another.

A sweep is declared **inside the config**, on the value itself, by separating the candidates with
a pipe:

```yaml
training:
  lr: 2e-3 | 1e-4 | 5e-5
  accumulate_grad_batches: 1 | 2
model:
  pooling_type: attention | mean
task:
  - name: test
    topk: false | true
```

`|` was chosen because the YAML parser does not see it: inside a scalar it is an ordinary
character, so a swept value arrives as a string, gets split on its top-level pipes, and each
candidate is parsed back as YAML. That is why candidates keep their types (`true` is a bool, `8`
is an int) and why a candidate can be a whole list — `positional_encoding: alibi | [alibi, rope]`
sweeps between one strategy and two. Write `\|` for a pipe that means itself, and quote a
candidate that *starts* with a flow indicator: `"[1,2] | [3]"`.

Nothing else is a special syntax: the pipe works in any file a config is loaded from, in any
section, at any depth, and it also works on the command line and in the GUI's override dialog,
because an override value goes through the same parser:

```bash
python train_model.py --cfg exp.yaml --override 'training.lr=2e-3|1e-4' --sweeps
```

## What the configs become

`MatformerConfigs` collects the declarations while it imports (`configs.sweeps`, a dict of dotted
path → candidates) and **keeps the first candidate as its own value**, so a config that nobody
expands is still one perfectly valid run: the GUI can read it, `config.read` can edit it, and code
that knows nothing about sweeps keeps working.

The expansion is `configs.sweep_variants()`, which yields exactly one ready-to-run config per
point of the Cartesian product — and yields `self` when there is nothing to expand, so a driver
never has to branch:

```python
for configs in MatformerConfigs.create_configs(paths).sweep_variants():
    MatformerTrainer(configs).run()
```

Each variant is a detached copy: re-imported from its dict form (so no instantiated object and no
pending dimension rides along), with its point applied and its artifacts renamed.

## Artifacts that cannot collide

Renaming is the whole trick, and it is declared rather than hardcoded: a config class lists the
fields that name something it writes, in `_sweep_scoped`, exactly like `_ref_fields` and
`_merge_fields` declare theirs.

| who writes | declared scoped | one point of `lr=2e-3` produces |
|---|---|---|
| `TrainingConfig` | `run_name`, `checkpoint_name` | `ironita_lr-2e-3` → `{save_dir}/logs/ironita_lr-2e-3_v1/`, `ironita_lr-2e-3_v_1_s_5000.ckpt`, its `_plugins` folder, and the W&B run |
| `JSONLSaverConfig` | `filename`, `summary_file` | `predictions.lr-2e-3.jsonl`, `summary.lr-2e-3.jsonl` |
| `ActivationStoreBaseConfig` | `filename` | `activations.lr-2e-3.duckdb` — one store per point, never two writers in one file |
| `DlaStatsConfig`, `FuncLocalScanConfig` | `output_file`, `summary_file`, `ranking_file` | one projections/summary/ranking file per point |

A tag is inserted *before* the extension and appended when there is none, and it is a no-op on a
name that already carries it, so scoping twice does nothing. A `readonly` store is an input rather
than an artifact, so it keeps the name it was given: a sweep that replays one shared activation
store still shares it.

Nothing downstream changed: the checkpoint manager, the loggers, the savers and the stores keep
reading their own configs, and read a different shelf each time. Which is also what makes
resume work per point — the scan is by checkpoint name, and each point has its own.

## Pointing at models instead of hyperparameters

A `checkpoint` section says which saved model a run loads — the config answer to `run_task.py
--model` — and one entry can stand for a *series*:

```yaml
checkpoint:
  - name: probes
    path: "$CHECKPOINT_ASSETS/sequence_classification/ironita"   # a folder of checkpoints
    every_n_steps: 500
```

`CheckpointDirectoryManager.select()` reads the folder with the same naming convention it writes,
so the entry expands to `ironita_v_1_s_500.ckpt`, `..._s_1000`, and so on — newest version per run
unless `versions:` asks otherwise, `last: N` to keep only the newest few. A `path` with pipes is a
dimension like any other, and a `path` that is a plain file, a `hf://repo` or a folder of
checkpoints is simply itself, so the rule never guesses at what a path means.

Because the series becomes an axis of the same product, models cross-multiply with
hyperparameters for free, and each point is named after both: `probes-s1500_lr-1e-4`.

## Driving a sweep

```bash
python train_model.py --cfg exp.yaml --sweeps        # print the plan, run nothing
python train_model.py --cfg exp.yaml                 # every point, one after the other
python train_model.py --cfg exp.yaml --sweep 3       # one point (also: --sweep lr-2e-3*)
python train_model.py --cfg exp.yaml --sweep 1,5,9 --probe fill=0.9   # measure before launching
python run_task.py --cfg exp.yaml                    # every point, over every task
python run_task.py --cfg exp.yaml --sweep 2          # ... or just that one
```

`--sweep` takes 1-based numbers and name globs, comma separated; a clause matching nothing raises,
because quietly running *part* of a sweep is the one mistake nobody notices in a table of results.

A sweep runs in **one process**, sequentially: one CUDA context, one DDP launch, memory freed
between points. When the sweep reaches the sections a model is built from, each point gets its own
model; `run_task.py` detects that (`sweeps.crosses_model`) and rebuilds only then — a sweep over
tasks, data or decoding shares the loaded model and rebuilds just the run around it, which is what
makes evaluating ten checkpoints of the same architecture cheap.

To fan a sweep out over machines instead, select one point per job: the queue dialog has the same
*Sweep points* field, `train.sweep` answers what the configs declare, and the GUI's **Sweep…**
button on the Training page lists the points and queues either the whole product or one of them.

## Reading the results back

Every point is a normal run of its own, named after itself:

- `{save_dir}/logs/{run_name}_v{version}/meta.json` carries a `sweep` block — the point's name, its
  index, and the exact values it chose — next to `config.json` (the whole flattened config, so the
  swept field with its resolved value) and `run.jsonl` (one row per step).
- task summaries carry `"sweep": "<point name>"`.
- probe reports carry the same block, each in its own log folder.
- the Runs & Plots page lists the points side by side, has a `sweep` column, and its search box
  matches on the point name — so "which combination won on test" and "at which epoch" are the same
  comparison it already does for two hand-written runs.
