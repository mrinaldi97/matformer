# Pretraining on NVIDIA HPC systems

The supported runner is `train_model.py`: autoregressive pretraining with
Lightning Fabric DDP, cached MDAT data, and complete optimizer-boundary resumes.
Model blocks, collation, data selection, checkpoints, and logging remain separate
modules. No new post-training pipeline is included.

## Start here

Install with Python 3.10+ and a CUDA-compatible PyTorch environment:

```bash
pip install -e '.[test]'
python -m matformer.tokenizer_evaluation --tokenizer /path/to/local/tokenizer \
  --model-config configs/pretraining/italian.json --output tokenizer-report.json
python train_model.py --config configs/pretraining/italian.json --device cuda
```

Edit the example's data/tokenizer paths, domain names, vocabulary and special-token
IDs before running. Its model size, mixture, LR, and 100,000-update schedule are
illustrative settings, not a tuned Italian recipe. Pretokenized training does not
download or instantiate a tokenizer. Prepare MDAT/tokenizer files, Python packages,
and any NLTK resources on a machine with connectivity before submitting jobs.

`--override training.max_steps=2 data.batch_size=2` accepts dotted keys with JSON
values. `--max-steps` is an **absolute global optimizer-step ceiling**, including
previous jobs. A normal invocation resumes the latest complete checkpoint for
the configured name/version; `--start-from-scratch` creates a new version.
`--checkpoint /path/to/file.ckpt` selects an explicit checkpoint. Selecting an
older checkpoint branches into a new version rather than overwriting later saves.

## Looped models

`model_config.num_loops` repeats the same stack with shared weights inside each
forward pass. The standard example uses 12 distinct blocks once. The looped
example uses 4 distinct blocks 3 times: 12 block applications, fewer parameters,
and gradients through every repetition. Embeddings are applied once; the final
normalization and LM head are applied after all repetitions. There is no detach,
adaptive halting, or cache carried between batches. `num_loops=1` is the ordinary
model. Cross-document recurrence hooks are a separate experiment and are rejected
by this runner because their hidden-state caches are not yet resumable.

The examples use untied embeddings. The library still has a historical weight
tying gate: if enabling `tie_word_embeddings`, also set `ALLOW_WEIGHT_TYING=true`.
The runner rejects a conflicting setting instead of silently training untied
weights. Preserve this setting when resuming.

## Token counts and mixtures

These settings deliberately count different things:

| Setting | Meaning |
| --- | --- |
| `data.mdat_selection.max_tokens` | Maximum cached content tokens selected for one dataset stage, before injected BOS/EOS and padding |
| `data.mdat_selection.domain_tokens` | Independent content-token caps per named domain |
| `data.mdat_selection.composition` | Percentage shares of `max_tokens`; values must sum to `100%` |
| `training.max_tokens` | Exact global ceiling on next-token targets contributing to training loss, across jobs and stages |
| `training.stage_max_tokens` | Optional exact target ceiling for this stage; reaching it completes the stage |
| `training.schedule_steps` | Fixed LR horizon for the whole experiment, including future data stages |

Select only Wikipedia, up to 100 million cached tokens:

```json
{"max_tokens": 100000000, "composition": {"wikipedia": "100%", "twita": "0%"}, "seed": 27}
```

Or set explicit independent caps:

```json
{"domain_tokens": {"wikipedia": 80000000, "twita": 20000000}, "seed": 27}
```

Place either object under `data.mdat_selection`. A zero cap excludes the domain;
unlisted domains are excluded when a cap/composition map supplies the domain list.
Names must exist in the selected view. Do not combine `composition` and
`domain_tokens`. Percentages describe mixture shares, **not percentages of each
source's entire size**. For example, `70%` requires a total `max_tokens` budget.

Selection preserves whole documents and never oversamples: documents exceeding
the remaining cap are skipped, so achieved totals/ratios can underfill. The
printed selection summary reports the actual per-domain counts. There is no
automatic substitution from another domain when one source is exhausted. Existing
view membership is preserved; the optional seed defines a deterministic
constant-memory permutation, not a uniform random shuffle. The selected order is
reused each epoch. These are stage/epoch mixture quotas, not a balanced-mixture
guarantee for every small prefix of training.

Training counters count valid next-token targets (including EOS and any internal
special-token targets), not padding or a sequence's first token. At a token ceiling
the final microbatch's loss mask is trimmed in global-rank order, and its partial
accumulation window is stepped immediately. The input is still evaluated in full;
this is a target budget, not an exact FLOP/input-token budget. Unused targets in
that consumed final microbatch are intentionally discarded. If you later increase
that ceiling they are not replayed. For routine wall-time interruptions, use
signals/time limits, which do not trim batches.

## Packing, padding, and the last batch

`SmallSplitter` packs sentences within each document, splits overlong sentences,
and preserves trailing tokenizer tokens. The strategy reserves two slots for
BOS/EOS. Loading multiple cached chunks into a larger context keeps all content
and one outer BOS/EOS pair. Context length must be a positive multiple of the
stored strategy's chunk size. Oversized sequences raise an error instead of
silently losing tokens. Existing corrupt/previously truncated caches must be
regenerated; the loader cannot reconstruct tokens that were never stored.

Distinct documents/chunk samples remain independent attention sequences. There
is no cross-document concatenation into one causal attention context. Unpadding
stores sequences contiguously **with boundaries**, which variable-length attention
uses; loss shifting also excludes cross-sequence pairs. This is not arbitrary
cross-document bin packing.

The collator constructs padding masks from lengths, so a real EOS is retained
even when PAD and EOS share an ID. It rejects empty/overlong sequences and supports
both padding and unpadding. SDPA now respects padding and the selected positional
encoding. Prefer `sdpa` + `padding` for initial validation; FlashAttention needs a
matching CUDA installation and GPU validation, and `unpadding` is the appropriate
path for variable-length batches. Mixed attention backends and old experimental
hooks need their own validation.

Final short batches are kept, and partial accumulation windows receive an update.
Gradients are normalized by the actual number of targets across the entire
accumulation window and all ranks. A rank with no data contributes a zero-target
forward/backward pass while the other ranks continue; one dummy row no longer
discards real rows in the same batch. These shared fixes address padding/batch
shape hazards relevant to downstream use, but do not establish the correctness
of an unrelated fine-tuning pipeline.

MDAT owns prefetching. `data.num_workers` must be zero; use a modest
`data.prefetch_buffer` (documents, not batches). Fabric must not add a second
distributed sampler. Global rank partitions documents across **all GPUs on all
nodes**. Epoch lengths are padded with dummy rows to the longest rank.

## Checkpoints and 24-hour jobs

New format-v2 saves contain native model parameters/buffers, stable inference
weights, all optimizers and schedulers, precision/scaler state, Python/NumPy/CPU
and local CUDA RNG per rank, each rank's consumed data cursor, partial document,
step/epoch/batch/target counters, data-stage history, run ID and resolved configs.
DataLoader's private generator avoids changing training RNG on resume. Validation
restores training RNG. Saves occur only after complete optimizer updates, so there
are no pending gradients to serialize.

Only rank zero writes the payload. It writes and fsyncs a temporary file, atomically
renames it, then records the successful save and optionally applies retention.
Interrupted `.tmp` files are not resume candidates. A rank-zero write error is
reported to all ranks. This is synchronous DDP checkpointing, not FSDP/model
parallel checkpointing: each GPU must fit the model and optimizer, and rank zero
needs sufficient host memory and disk space for a full checkpoint.

Useful controls under `training`:

```json
{
  "save_every_n_steps": 250,
  "save_every_n_seconds": 1800,
  "snapshot_every_n_steps": 5000,
  "checkpoint_steps": [10, 100, 1000],
  "checkpoint_log_base": 2,
  "keep_last_checkpoints": null,
  "max_time_seconds": 82800,
  "schedule_steps": 100000
}
```

Step/time intervals save recovery checkpoints; time intervals are checked at
optimizer boundaries. Explicit milestones, periodic snapshots, and powers of the
log base mark retained snapshots. Logarithmic snapshots alone are a poor recovery
policy because late intervals become long. `keep_last_checkpoints=null` (default)
keeps everything. **Setting it to a positive integer explicitly enables deletion
of older non-snapshot checkpoints in the current name/version after a successful
save.** Retained snapshots are not pruned, so budget/archive those separately.
No existing checkpoints or datasets are removed by installing this change.

Plan checkpoint capacity separately from corpus capacity: an FP32 model with
AdamW moments is roughly 12 bytes/parameter before other state, with another full
checkpoint temporarily needed during publication. Measure actual files; BF16
mixed precision does not imply all saved parameters are two bytes. A nominal
1 TB project quota must cover data, checkpoints, environments and logs together.

Exact resume requires unchanged model/loop settings, precision, optimizer/LR
settings, accumulation, dataset identity and **total process count**. Data files
are checked by paths, sizes and modification times; this is a fast immutable-file
check, not a cryptographic verification of the entire corpus. Do not mutate active
datasets. Changing nodes while preserving total rank count and equivalent rank
mapping can work, but changing total GPUs is rejected. Bitwise GPU reproducibility
also depends on identical software/hardware and deterministic kernels; saving RNG
does not make nondeterministic CUDA kernels deterministic.

The saved LR horizon is reused when a job resumes. Set `schedule_steps` before
the first stage when future stages will follow. Integer warmup/hold values mean
steps; float values in `[0,1]` mean fractions of that horizon. An increased
`max_steps` does not silently stretch/restart the schedule. Legacy checkpoints
lack complete state: import weights into a new experiment with
`--checkpoint OLD.ckpt --weights-only`, rather than claiming an exact resume.

## Leonardo / HPC4AI launch

CINECA lists a 24-hour wall time for Leonardo's normal Booster production QoS;
other QoS limits differ. Check your allocation's current permissions. The example
script requests 2 nodes × 4 A100 GPUs and one Slurm task per GPU, consistent with
[CINECA's Leonardo guide](https://docs.hpc.cineca.it/hpc/leonardo.html) and
[Fabric's Slurm launch instructions](https://lightning.ai/docs/fabric/stable/reference/guide/multi_node/slurm).

```bash
mkdir -p logs
# Activate your environment first, or export MATFORMER_ACTIVATE=/path/to/activate.sh
sbatch scripts/leonardo/pretrain.sbatch configs/pretraining/italian.json
```

For the first two-node GPU check use a separate checkpoint directory:

```bash
sbatch --qos=boost_qos_dbg --time=00:30:00 scripts/leonardo/pretrain.sbatch \
  configs/pretraining/italian_looped.json --max-steps 2 --override \
  save_dir=checkpoints/distributed-smoke training.save_every_n_steps=1
```

Repeat with `--max-steps 3` and the same other settings to exercise resume. The
printed world size must be 8. Inspect every rank for NCCL errors and compare
counters before scaling. This example is not automatically submitted by tests.
Use one launcher: `srun` starts the processes and Fabric recognizes Slurm. Do not
run `torchrun` inside that `srun`. For a single node request `--nodes=1`; if changing
GPUs/node, change both `--ntasks-per-node` and `--gres=gpu:N`. The effective full
batch is GPUs × sequences/GPU × accumulation. No site-specific NCCL interface is
hard-coded; use the cluster-supported CUDA/NCCL/network environment.

`SIGUSR1` and `SIGTERM` request a coordinated stop after the current accumulation
window. The script requests a signal 15 minutes before wall time. During training
you can signal the step with `scancel --signal=USR1 JOBID.0`. Measure your slowest
update plus checkpoint write and leave more time than that; no handler can save
after SIGKILL/node failure. Signals arriving during initialization are not yet
handled. `max_time_seconds` measures training-loop time, not allocation/setup time;
the Slurm signal provides the allocation-relative guard.

Resubmit the same command after a clean stop. For a chain of jobs using unchanged
data/config, submit the successor with `sbatch --dependency=afterok:JOBID ...`.
Use the same job name plus `--dependency=singleton` if serializing independent
submissions, and never run two writers on one checkpoint name/version. Validate
the cluster's dependency/requeue behavior first. HPC4AI can use the same Python
runner and Slurm pattern; adapt its account, partition, GPUs/node and activation
script. Its GPU/network behavior has not been tested here.

## More data than local storage

Yes: stage immutable dataset parts between jobs while retaining training state.
Choose a fixed global LR horizon and unique `data.stage_id` for every part.

1. Train `part-000` to completion (`max_epochs`, normally 1, or an intentional
   `stage_max_tokens` ceiling). Time-limit stops mid-part resume the same part.
2. Wait for the final checkpoint and job exit. A completed-stage checkpoint has
   `stage_complete=true`. Keep it and a backup before removing any old input.
3. On a login/data-transfer host, archive/remove only the completed part and stage
   `part-001`. Never remove data while ranks/prefetchers still have it open.
4. Change `data_root`, `stage_id`, and optionally that part's selection, then run:

```bash
python train_model.py --config part-001.json --checkpoint /path/to/completed.ckpt \
  --next-data-stage --device cuda
```

The next stage keeps model, optimizers, LR progress, RNG, run ID, and global counters;
it starts its data cursor and local epoch/target counters at zero. The loader never
needs the previous part for this transition. Reusing a completed stage ID is
rejected. If you hit a global `max_tokens` limit mid-stage, that is not a completed
stage: resume it or use an intentional `stage_max_tokens` limit. A stage ceiling
permits intentionally skipping its unused tail. Build each part with the desired
mixture if you want balanced data throughout training, rather than all Wikipedia
followed by all tweets. The library does not download or delete corpus files for you.

## Offline logging and tokenizer checks

Leonardo compute-node connectivity should not be relied upon. The
[EPICURE Leonardo AI guide](https://docs-31e4a2.pages.it4i.eu/articles/best-practices/leonardo-ai-workloads/)
specifies offline W&B for its compute-node network restrictions. The runner and
Slurm templates default to `WANDB_MODE=offline`; `disabled` uses only local logs.
Explicit `WANDB_MODE=online` opts into live logging on a cluster with connectivity.

Rank zero always appends `save_dir/logs/RUN_ID/metrics.jsonl`. Offline W&B writes
one segment per job with a shared group/run identity and global optimizer steps;
offline mode does **not** behave like W&B online resume. Sync each completed
segment later from an approved network-connected host:

```bash
wandb sync /path/to/offline-run-YYYYMMDD_HHMMSS-SEGMENT_ID
```

See the [official W&B sync command](https://github.com/wandb/wandb/blob/main/wandb/cli/cli.py).
Separate segment runs remain grouped in W&B; local JSONL provides the continuous
record. Abrupt failures can leave metrics beyond the last durable checkpoint, so
replayed step numbers may appear after recovery.

The tokenizer module defaults to Italian accent/apostrophe/whitespace/Unicode
probes. Supply `--input corpus.txt corpus.jsonl --text-field text` for a corpus
report. It streams documents and reports tokens/word, UTF-8 bytes/token, unknown
tokens, round-trip differences and special-token IDs. Differences can reflect
intentional normalization. `--model-config` checks vocabulary capacity and
BOS/EOS/PAD agreement, exiting nonzero on a mismatch. It uses local files only.

## Checkpoint references and validation

[OLMo-core's checkpointer](https://olmo-core.readthedocs.io/en/latest/_modules/olmo_core/train/callbacks/checkpointer.html)
separates frequent ephemeral recovery saves, retained intervals and fixed-step
snapshots. This informed the two cadences here. The
[Apertus engineering report, IV-B.2](https://arxiv.org/html/2604.12973v1#S4.SS2.SSS2)
reports asynchronous saves every 250 iterations, using a failure/overhead tradeoff,
plus Slurm signals and serialized jobs. This library keeps synchronous atomic
writes for a smaller implementation; 250 steps is an example, not a universally
optimal interval.

Run the CPU tests with `python -m pytest tests`. The distributed regression uses
real Gloo collectives and two CPU processes:

```bash
tmp=$(mktemp -d)
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/distributed_pretraining_worker.py "$tmp"
```

These checks cover mathematical batch weighting and complete state restoration.
They do not verify CUDA kernels, NCCL/InfiniBand across physical nodes, checkpoint
latency under shared-filesystem load, or Slurm signal delivery. Run the GPU smoke
and resume checks above before a large allocation. See the work diary for actual
test results and unresolved limitations.
