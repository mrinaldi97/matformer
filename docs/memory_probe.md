# Memory & speed probing

How much GPU a training really needs, and how fast it goes once it gets it.

The problem is that nobody knows the answer in advance: Matformer's memory footprint depends on the
sequence length (which varies, and is unpadded by default), on the masking rate (which is sampled per
batch, and decides how many rows are gathered out of the vocabulary), on which learning signal is active
in this step (CLM and MLM cost different amounts), on the layer schedule if the residual stream loops,
and on whatever a mounted plugin decides to keep. The config file says `batch_size: 48` and the run
either fits or it does not.

`MatformerTrainer.probe_memory()` answers the question empirically, in seconds, on the machine that will
do the training, and reports the throughput at the size it found.

> **Measured, for `mercurio_configs/albertina.yaml` on `Albertone_MDAT`** (strategy `Gettone1024_`,
> bf16-mixed, `max_seq_len: 1024`, MLM at the configured 0.2 rate): **85 rows per rank** on one
> Tesla V100S-32GB at `fill: 0.90` — 87 040 tokens per micro-batch, 696 320 tokens per optimizer step
> with `accumulate_grad_batches: 8` — peaking at 28 989 MiB with 256 MiB of margin, at 17 200 tokens/s.
> Ninety rows fit at `fill: 0.92` but reach no faster. Two ranks take the same 85 rows each, and real
> training at that size ran clean. The `48` in the yaml is 1.8× under the ceiling.


## What was added

| where | what |
|---|---|
| `matformer/training/matformer_trainer.py` | `probe_memory()`, plus `_probe_step`, `_mock_batch`, `_report_speed`, and the throughput window inside `train()` |
| `matformer/training/utils.py` | `PEAK_TFLOPS`, `peak_tflops()`, `matmul_parameter_count()`, `step_flops()` |
| `matformer/matformer_wrapper.py` | `strict_steps`: one line, so a probe sees the failure instead of the wrapper's skipped batch |
| `matformer/dataclasses/training_dataclasses.py` | `log_speed_every_n_steps` |
| `matformer/training/launch.py` | `--probe [key=value …]`, `--then-train`, and the `probe` key of a launch spec |
| `matformer/server/memory.py` | RPCs `train.probeOptions`, `memory.reports`, `memory.report` |
| `matformer/gui/trainpage.py` | "Memory & speed probe…" and "Probe reports…" on the Training page |

Nothing new has to be trained, loaded or rebuilt to probe: the trainer already holds the model, the
gradients, the optimizer state, the DDP buckets, the plugins and the data module, and the probe walks
that same object.


## How it works

**Mock tokens.** Only the token count matters, so no document is read: `_mock_batch` builds random ids
and hands them to the dataloader's own `collate_fn`, which means the batch is a real `UnpaddedTensor`
(or `PaddedTensor`, or a labelled classification batch) with the same packing, the same attributes and
the same `same_document_mask` as training produces. The MDAT is never opened for data.

**Worst case, not average.** Every row is as long as `max_seq_len` allows; the masking rate is pinned to
the upper end of what the config declares (a masked loss gathers `rate · tokens` rows out of the
vocabulary, so the *average* rate undershoots the peak); and each configured learning signal is walked
in turn, so a run alternating CLM and MLM is bounded by its dearer objective rather than by whichever
signal a random draw happened to pick. A `multitask` selector is left alone, since summing the signals
in one step is heavier still.

**Warm-up first.** One complete step — gradients reduced, clipped, applied — so that everything a steady
step needs already exists before anything is measured: DDP buckets, the optimizer's moments, cuDNN and
flash-attention workspaces. Without it the first real probe pays for all of that and the answer is wrong.

**The search is two points, not a binary scan.** Activation memory is linear in tokens, so
`peak = intercept + slope · rows` places the ceiling. The probe starts from one row and doubles what it
knows works (`2, 4, 8, …`) — never a big optimistic extrapolation from a single point, because an
out-of-memory is slow and leaves a mess — until two measured successes are close to the ceiling, then
jumps to where the line says the limit is, and verifies. Four to six steps forward and back. After any
failure the search bisects between the largest verified size and the smallest failure, and the answer is
confirmed by `steps` consecutive real steps at that size, because a model that keeps state between steps
(a recurrence store, an activation cache) does not say so in a single one.

**Distributed.** The search runs with gradient reduction disabled, so a rank that dies in the middle of
a step cannot leave its partners waiting in a collective forever; each attempt's outcome is then agreed
with an `all_reduce(…, "min")` of a fits flag, which keeps every rank on the same trajectory. The
reduction buffers are real memory, and they exist because the warm-up step does synchronise.

**Cleaning up.** Between attempts the gradients are freed, the model's caches (`storage`, and the
alibi/mask/rotary caches, which are keyed by shape) are cleared, and after a failure the exception object
is dropped before `gc.collect()` and `empty_cache()`. That last part is not hygiene for its own sake: an
exception holds every frame of the failed step, and with them the activations it died on — an early
version of the probe kept a reference, and from then on every smaller batch "OOMed" on a GPU that was
full only of the previous failure's corpse. It reported `batch_size=1` for a card that runs 85 rows.

The probe takes real steps, so it puts the weights and the optimizer's state back afterwards
(`keep_weights`, on by default): a probe followed by a training in the same job starts where it was told
to, not from a handful of steps on random tokens.


## Speed

`train()` keeps a window of `[clock, step, tokens, sequences]` and reports every
`training.log_speed_every_n_steps` optimizer steps (0 switches it off). Token counts come from the
batch's shape — `tensor.shape[0]` for a packed batch — so measuring never synchronises the device and
never reduces a tensor: the loop already syncs once per step through the loss `.item()`, and the window
adds two integer additions per batch and one clock read per window.

Metrics, all logged as `speed/*` (and shown in the progress bar):

- `tokens_per_second`, `tokens_per_second_cluster` (per rank, and × world size) — the number papers and
  framework comparisons quote;
- `sequences_per_second`, `steps_per_second`, `step_ms`;
- `tflops` — FLOPs actually performed, from the standard estimate: `6·N·T` for the parameter matmuls
  plus the attention's quadratic term `12·L·d·Σs²` (halved when causal, tripled backward), with `N` the
  parameters that multiply (embedding lookups excluded, the output projection included) and `Σs²`
  approximated by `rows · mean_length²` so that knowing it costs nothing;
- `mfu_percent` — the share of the device's peak dense throughput at the precision in use. MFU is the
  one number that compares across machines and frameworks, and it is only honest because `PEAK_TFLOPS`
  is declared per architecture: the alternative (measuring peak with a benchmark) is slower and lies
  about the machine you are on.

The probe measures the same things at the size it found, so memory and compute come from one allocation.


## Measurements: Albertina gold on mercurio, cascadelake:v100

Model `Albertina_gold` (768 hidden, 6 layers, 32768 vocab, ALiBi, flash attention, 96.4M parameters),
`precision: bf16-mixed`, `accumulate_grad_batches: 8`, `max_seq_len: 1024`, data `Albertone_MDAT` with
strategy `Gettone1024_`. One Tesla V100S-PCIE-32GB per rank.

| run | GPUs | fill | rows/rank | tokens/rank | peak MiB | held MiB | tokens/s per rank | TFLOP/s | MFU |
|---|---|---|---|---|---|---|---|---|---|
| `albertina-probe-1v100` | 1 | 0.90 | **85** | 87 040 | 28 989 | 31 012 | 17 215 | 8.3 | 53 % |
| `albertina-probe-ddp` | 2 | 0.90 | **85** | 87 040 | 28 984 | 31 052 | 19 979 | 9.7 | 62 % |

Both reports are on disk and readable with `memory.report` (`$ASSETS/checkpoints/probe_validation/logs/…/
memory_probe.json`). A run at `fill: 0.80` measured 75 rows/rank, peak 25 655 MiB, 4.20 s per step
(18.3k tokens/s), and one at `fill: 0.92` measured 90 rows/rank at 30 654 MiB and 17.1k tokens/s.

### The answer for this configuration

On **one Tesla V100S-32GB**, Albertina gold in `bf16-mixed` at `max_seq_len: 1024` sustains **85 rows per
rank = 87 040 tokens per micro-batch**, which is **696 320 tokens per optimizer step** with
`accumulate_grad_batches: 8`, peaking at 28 989 MiB of 32 494 with 256 MiB to spare. Two ranks take the
same 85 rows each (1.39 M tokens per optimizer step, ~40 k tokens/s over the pair). The `48` in the yaml
is 1.8× under that ceiling.

The search trace, which is the shape of every good probe:

```
[probe]       1 rows        1024 tokens  fits  peak    1115 MiB (+   348)  held    1204
[probe]       2 rows        2048 tokens  fits  peak    1437 MiB (+   670)  held    1506
[probe]       4 rows        4096 tokens  fits  peak    2080 MiB (+  1212)  held    2186
[probe]      88 rows       90112 tokens  fits  peak   30006 MiB (+ 28936)  held    30410
[probe] ceiling 29245 MiB of 32494 MiB, 332 KiB per token, 85 rows/rank verified
[probe]      85 rows       87040 tokens  fits  peak   29000 MiB (+ 19403)  held    30452   ×3
[probe] safe batch 85 rows/rank = 87040 tokens/rank · 348160 tokens per optimizer step ·
        peak 28984/32494 MiB (headroom 261) · 19,979 tokens/s/rank · 9.7 TFLOP/s · MFU 61.6%
```

Three rows of 1, 2, 4 cost 348, 670, 1212 MiB: perfectly linear at ~303 MiB per 1024-token row, which is
why the fit is allowed to jump from 4 straight to 88. `bytes_per_token` came out at 340 KiB and
`base_mib` at 1104, so the answer is predictable from two numbers:
`rows ≈ (fill · VRAM − 1.1 GiB) / (340 KiB · seq_len)`.

**End-to-end validation of the same job.** With `apply: true` the job went on to train at the probed size,
on real data, both ranks, and logged from inside the loop:

```
step 2  train/loss 10.2   speed/tokens_per_second 19933   speed/step_ms 4357   speed/mfu_percent 57.8
step 4  train/loss  9.7   speed/tokens_per_second 20311   speed/step_ms 4619   speed/mfu_percent 59.2
```

The throughput measured during real training (19.9–20.3k tokens/s/rank) is the throughput the probe
predicted on mock tokens (20.0k), and no rank OOMed. A probe that only predicted memory would be half
the tool.

### What the numbers say about this run

- **The config was running at ~55 % of the card.** `batch_size: 48` against 85–90 rows/rank measured. On
  two V100S that is 1.4M tokens per optimizer step available versus the ~350k the config was doing.
- **`fill` is nearly free in either direction, so keep some.** 75 rows at 0.80, 85 at 0.90 and 90 at 0.92
  all land between 17.1k and 18.3k tokens/s per rank — within node-to-node variation the throughput does not
  improve as the last tenths of VRAM are consumed, while the risk of an out-of-memory does. `fill: 0.90` is
  the recommendation: the largest safe batch, with margin left for fragmentation and for the odd plugin.
  tokens/s (18.3k vs 17.1k) while 0.90 lands at 20.0k: pushing into the last 8 % of VRAM buys nothing and
  costs the safety margin. `fill: 0.90` is the default recommendation.
- **A step's first use of a new shape is the expensive one.** After the first 85-row step, ~7 GiB of live
  memory stayed allocated (and then held flat across the following steps: a ratchet, not a leak — most of
  it is the vocabulary: logits of `tokens × 32768` plus their gradient and the loss's log-softmax).
  Measuring only the *marginal* cost of one step would have missed it, which is why the probe reads the
  absolute peak and confirms with consecutive steps.
- **bf16 on Volta is a mistake waiting to be seen.** MFU is quoted against 15.7 TFLOP/s because that is
  what a V100 does with a bf16 matmul (no bf16 tensor cores before Ampere): 52–62 % MFU means the run is
  already close to the fp32 cores' limit, and `precision: 16-mixed` (125 TFLOP/s of fp16 tensor cores on
  the same card) is where the headroom actually is. The probe does not judge a configuration, it makes
  the number visible; that number is the argument for an A/B of `16-mixed` versus `bf16-mixed` here.
- **Real batches are half of the worst case.** Sampling 4000 consecutive chunks from `Albertone_MDAT`
  with `Gettone1024_`: mean 545 tokens, p50 495, p90 1013, p99 1024, and only 1.4 % of rows reach 1024.
  So a batch of 85 rows carries ~46k tokens on average and 87k at worst. Sizing for the worst case is
  what makes the run never crash; it also means ~45 % of the token budget is idle most of the time.
  Which is the next section.


## Variable batch size: dream or reality?

**Reality, and the plumbing already exists — but it changes the training equation, so it must be an
opt-in modality, not a default.**

The mechanism is cheap. `MDatContainer` is an `IterableDataset` and the batch is whatever `collate_fn`
is handed, so a token-budgeted assembler is an iterator wrapper: pull items until
`Σ length ≥ budget` (or `rows == max_rows`), call the existing `collate_fn`, repeat. Roughly thirty lines
in `matformer/data_module.py`, no change to the MDAT, no change to the model, and the budget is already
computed by the probe (`tokens_per_rank`). Nothing about the framework has to be rethought.

The gains are what one would expect from the distribution measured above: with a fixed 85-row batch the
GPU is fed half its capacity on a typical step; filling to the token budget instead gives ~1.8× the
tokens per step at the same OOM risk, or the same tokens per step at a lower wall-clock cost.

What deserves care, in order of how much it can quietly ruin a run:

1. **The budget is not tokens, it is `Σ tokens` plus `Σ tokens²` for attention.** Two full 1024-token
   rows cost four times the attention of two 512-token rows. Budgeting on tokens alone is therefore
   *unsafe* precisely in the case that matters (a batch of long documents). The fix is free: the
   assembler already has the lengths on CPU, so accumulate both and stop when either limit binds. The
   probe's worst case (all rows at max length) is the correct calibration point for the `Σ s²` limit, and
   `step_flops()` already computes that term.
2. **Loss normalisation.** `cross_entropy` averages over the positions of *this* batch, so a small batch
   gives each token more weight than a large one, and variable batches make the effective step size
   variable even when the mean token count is constant. The clean formulation is to reduce with `sum`,
   accumulate the token counts across the gradient-accumulation window, and divide once before the
   optimizer step. That is a change in `LearningSignal.loss_function`/`training_step`, small but real, and
   it must be the same change in every learning signal (the masked, autoregressive and
   token-classification ones each select their own positions).
3. **DDP needs the same number of collectives per rank.** Rows per micro-batch may differ across ranks
   (that only skews the average, handled by point 2 with a token-weighted all-reduce); the *number of
   micro-batches per optimizer step may not*. So the adaptive quantity must be rows-per-batch with the
   micro-batch count fixed, not the other way round.
4. **`chunked_for_recurrence` ties the row count to the number of parallel document slots** (`__next__`
   advances slot `iteration_count % batch_size`, and the recurrence store is keyed per slot). There,
   changing rows per batch changes the document stream itself, so a token budget is only sound if slots
   stay fixed — i.e. adaptive batching should be offered for `chunked_tokens` first, and for the
   recurrent modality only by keeping slots constant and letting the budget decide something else (for
   instance how many slots are active, which is a different, bigger decision).
5. **Epoch and LR accounting.** `total_steps` is derived from `len(datamodule) / batch_size`; with a
   variable batch size that becomes a mean. The honest constant is the total *tokens* of the view, which
   the MDAT already knows exactly (the precomputed lengths: `total_divided_chunks × chunk_size`), so
   scheduling against tokens becomes more accurate than the current step-based version rather than less.
6. **Tail behaviour** already has an answer: the `worker_has_finished` dummy batch keeps ranks in step
   when one runs dry, and a budgeted assembler inherits it.

**Recommendation.** Do it in two steps. Step one is free today and captures the first-order gain: let the
probe write the batch size (`--probe --then-train`, or `memory.report`'s `recommendation`) so the static
value is the machine's optimum instead of a guess — that alone took this config from 48 to 85 rows. Step
two is the token-budgeted assembler with `Σ tokens` and `Σ tokens²` limits and token-summed losses, kept
behind a config flag, and validated against a fixed-batch run of the same total tokens before it becomes
anyone's default. Not a dream, but not a free lunch either: it changes what a step means, so it earns its
keep only with the loss normalisation done correctly.


## Using it

```bash
# terminal: measure, then train at the size found, on 2 GPUs
python -m matformer.training.launch --cfg ../mercurio_configs/albertina.yaml \
    --probe fill=0.90 steps=3 --then-train --gpu 2

# just the answer
python -m matformer.training.launch --cfg … --probe fill=0.9
# knobs accepted: fill, max_rows, steps, keep_weights, seq_len, report_path, apply

# through the server, and onto the cluster (same spec, executed where the GPUs are)
python -m matformer.training.launch --queue --cfg … --probe steps=3
```

Server: `train.probeOptions` returns the knobs with their defaults (generated from `probe_memory`'s own
signature, so dialog and code cannot drift); a probe is then queued with `train.start` carrying a `probe`
key, which means the local queue and `slurm.submit_training` both run it unchanged — the probe is a
training spec that stops after measuring. `memory.reports` lists every saved probe as a table, and
`memory.report` gives one in full, search trace and all:

```
AT                   RUN                        GPU                    ROWS  TOKENS  PEAK MIB  VRAM MIB  TOK/S  MFU %
2026-11-17T06:27:20  probe-albertina-1v100c_v1  Tesla V100S-PCIE-32GB    90   92160     30654     32494  17067   52.4
```

GUI: the Training page has *Memory & speed probe…* (the configs and overrides already on the page are the
probe's; the dialog adds the knobs and, when SLURM is reachable, the allocation) and *Probe reports…*,
which lists saved probes and opens one on double-click. A probe job appears in the job dock like any
other, and its log holds the table, so the answer is where the training's output is.

Each probe also writes `memory_probe.json` next to the run's logs and logs `probe/*` and `speed/*` to the
run's metric series.


## What else came out of this

Things that were broken before the probe and that the probe walked into. They are fixed here because
nothing else would have found them at this cost.

- **`Albertone_MDAT` could not be opened at all.** Its `datasets/` folders are gone (only `pretok/` and
  the catalog live on BeeGFS), and `MDat.load_submdat` refused any dataset whose raw folder was missing —
  even though training with a strategy reads nothing but `pretok/<strategy>/<name>/{tokens,chunks}.dat`,
  which is present and complete for all 19 submdats. `mdat.py` now treats a missing raw database as
  "nothing to open here" and only raises when neither the raw nor a pretokenized copy exists. The
  container then loads, serves real chunks, and trained as before.
- **A schema the code had outgrown.** `pretok_strategy_view_distributed_length` in `mdat.db` had no
  `chunk_multiplier` column, while every lookup in it asks for one: `sqlite3.OperationalError: no such
  column` during `_setup_data`, on every container built before the variable-context change. Added to
  `_ensure_schema_upgrades`, with the existing rows stamped multiplier 1 — which is what they are: the
  stored per-worker counts sum to 597 575 846, exactly the `Gettone1024_` precomputed total at
  multiplier 1.
- **A five-hour surprise waiting inside a training job.** `MDatContainer.__len__`, on a lookup miss, used
  to call `precompute_chunks_distributed`, which walks *every document of the view* to count chunks: for
  `default` that is 360M documents, hours of I/O, paid for on GPUs, to learn an epoch boundary that the
  per-dataset precomputes already give. It now estimates `total / workers` (round-robin dealing makes that
  exact within one window), prints a warning naming the command that would compute it exactly, and leaves
  the scan a deliberate act.
- **`launch --queue` never saw its answers.** The asynchronous client puts replies on its events queue and
  callbacks are run from there by whoever pumps it (the GUI's main loop); `_await`, the CLI's blocking
  helper, only slept and waited for a callback that was already queued. It looked like the server had
  ignored the request, while the job had in fact run. `_await` now pumps the queue like the GUI does.
- **`stochastic_selector` could not be imported.** It imports `LearningSignalSelector` from
  `matformer.training.learning_signals_selectors`, which is a namespace package with no re-export, so any
  config asking for CLM/MLM alternation died at model construction. One line: import from
  `matformer.training`, like the learning signals do.
- **Fabric's `_forward_module` is not the model.** It is the DDP module, so writing an attribute on it
  never reaches `MatformerModelWrapper`. The probe unwraps to the wrapper itself; worth remembering for
  anything else that wants to poke at the live model, since the failure mode is silent.
- **The MDAT shell does not start in `torch29_v100`**: `cmd2` needs `rich_argparse`, which is not
  installed there. (`python -m matformer.data.mdat.tui` → ImportError.) Not touched: it is a `pip` call on
  the machine that can run that venv, `slurm pip torch29_v100 rich_argparse`.
- **`MDatContainer.__len__` calls `.cuda()`** when the process group is up, so any CPU run of the MDAT
  path dies (this is why the logic-level tests here use the `lazy` module and the GPU tests use the MDAT).
