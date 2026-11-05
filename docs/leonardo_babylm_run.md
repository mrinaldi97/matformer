# BabyLM on Leonardo

For the current checkpoint, data-stage, tokenizer, and multi-node contract see
[the pretraining guide](pretraining.md). This page retains the BabyLM-specific
paths and launch commands.

The launch configuration is `configs/babyLM/babyLM_piccolo_1epoch_leonardo.json`,
derived from `babyLM_piccolo_1epoch.json`. The original stays unchanged.

Preserved: 12 layers, hidden size 768, 12 heads, post-RMSNorm, ALiBi, SwiGLU
factor 3, untied embeddings, Muon, learning rate 8e-4, batch size 20,
accumulation 5, and one epoch. Adapted: local Gettone (32768 tokens,
BOS 5/EOS 6/PAD 0/MASK 4), cached 512-token sequences, SDPA with padding,
separate validation data, and W&B project `team-jerik/UniTO-LLM`.
No FlashAttention installation is needed for this initial run.

Training has 39728 sequences: 1987 batches and 398 optimizer updates,
including the final accumulation window. Validation uses all 1017 held-out
sequences and reports cross-entropy weighted by the number of next-token targets.
It runs every 50 updates and at the end. Checkpoints are saved every 50 updates
and at the end, with atomic replacement of the destination file. Failures stop
the job instead of being converted into zero training loss.

## First GPU check

On Leonardo, from the repository root, after applying the patch:

```bash
mkdir -p logs
sbatch scripts/leonardo/babylm.sbatch smoke
```

Record the printed job ID. This uses one GPU and the project's debug QoS for
at most 30 minutes. It performs two optimizer updates, validates two batches
after each update, and saves separate smoke checkpoints. This verifies the GPU,
bfloat16, W&B and Slurm integration, which CPU regression tests cannot verify.
The smoke test stops after two updates of the configured schedule; its checkpoint
is separate from the full experiment.

```bash
squeue -u "$USER"
tail -f logs/babylm-JOBID.out
sacct -j JOBID --format=JobID,JobName,State,Elapsed,ExitCode
```

Replace JOBID with the actual numeric job ID. `PD` means pending, `R` means
running. After a job disappears from squeue, use sacct. Ctrl-C exits tail only;
it does not cancel the job. Before the job starts, the log file may not exist.
A successful smoke run ends with `Training complete. Bye.` and a Slurm
`COMPLETED` state with exit code `0:0`. Check that validation losses are finite
and the log reports checkpoints at updates 1 and 2.

## Full epoch

Only after the smoke run succeeds:

```bash
sbatch --job-name=babylm-full --qos=normal --time=24:00:00 scripts/leonardo/babylm.sbatch full
```

Project/account: `IscrC_ALBERTIN`; partition: `boost_usr_prod`; one node and
one GPU. The current partition's normal time limit is 24 hours. This is a time
limit, not an estimate of how long training will take. Closing SSH does not stop
an sbatch job. Do not submit concurrent full runs to the same checkpoint folder.

Full checkpoints: `checkpoints_babylm_piccolo_gettone/`.
Smoke checkpoints: `checkpoints_babylm_smoke/`.
W&B entity/project: `team-jerik/UniTO-LLM`. The script defaults to offline logging.
All Hugging Face reads are offline from the transferred caches.

Offline mode can also be set explicitly:

```bash
WANDB_MODE=offline sbatch --job-name=babylm-full --qos=normal --time=24:00:00 scripts/leonardo/babylm.sbatch full
```

Later, from a node with connectivity, run `wandb sync` on the specific offline
run directory printed in that job's log. Offline logging does not produce a
live dashboard.

## Stop and resume

To request a checkpoint and stop after the current accumulation window:

```bash
scancel --signal=USR1 JOBID.0
```

Wait for `Checkpoint saved: ...` and for the job to finish.
The handler is active once the training loop starts. The job also requests
USR1 three minutes before the time limit. Saving depends on enough time remaining;
the previous complete checkpoint is retained if saving fails.

To cancel immediately (or cancel a pending job):

```bash
scancel JOBID
```

Immediate cancellation does not guarantee a new checkpoint. Resume the full
run by resubmitting the same `full` command after the old job has exited. The
entry point selects the latest completed full checkpoint. Keep model, data,
batch size, accumulation and GPU count unchanged. To intentionally start a new
experiment, use a distinct checkpoint name/folder rather than removing old files.

## Scope of validation

CPU tests cover sequence boundaries, smaller final batches, token-weighted
accumulation/validation, target budgets, and complete checkpoint restoration.
A real two-process Gloo test checks distributed gradient weighting and exact
resume. Physical multi-node CUDA/NCCL execution still requires a GPU smoke run.
This BabyLM script deliberately allocates one GPU; use `pretrain.sbatch` for the
multi-node template. Format-v2 checkpoints include per-rank RNG and precision
state. Older checkpoints can initialize weights with `--weights-only`, but cannot
provide a complete resume.

Slurm references: https://slurm.schedmd.com/sbatch.html and
https://slurm.schedmd.com/scancel.html.
