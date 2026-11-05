# Italian BabyLM on Leonardo: first experiment and initial review

Historical review: subsequent work and resolved findings are tracked in
[WORK_DIARY.md](../WORK_DIARY.md). Use [pretraining.md](pretraining.md) for the
current training and resume instructions.

Date: 2026-11-05. Branch inspected: `feat/flexible-data-loading`.

Status: initial static review and experiment specification. No training job has
been submitted; runtime behavior, checkpoint recovery, GPU kernels, and distributed
training have not been validated. This is not a completed whole-repository audit.

## Experiment

Use an autoregressive model and the official Italian BabyBabelLM tier:
[BabyLM-community/babylm-ita](https://huggingface.co/datasets/BabyLM-community/babylm-ita).
The dataset card reports 16,575 documents and 9,467,687 whitespace-counted words.
The tier is called 10M because BabyBabelLM calibrates budgets across languages;
it is not exactly 10 million Italian words. Access is gated: the downloading
account must have accepted the dataset's conditions. Access from Leonardo has
not been verified.

Treat 10M as the maximum unique training-word budget for this infrastructure
experiment. Do not set `mdat_selection.max_tokens=10000000` to enforce it:
that option counts cached subword IDs. Record actual raw word counts separately
from cached tokens, BOS/EOS, padding, and repeated-epoch exposure.

Preparation:

1. Pin the dataset revision and inspect available splits before downloading.
   Preserve document IDs, categories, source attribution, and licensing metadata.
2. Use official disjoint evaluation splits if supplied. Otherwise make a seeded,
   document-level holdout before tokenizer training or chunking. Check duplicate
   document IDs/content across splits. Record the resulting smaller train word
   count; do not count held-out words as training words.
3. For a self-contained baseline, train a 16K byte-level BPE tokenizer only on the
   training split. Save it locally with explicit PAD/BOS/EOS/MASK IDs. An existing
   tokenizer is also suitable for an infrastructure-only test, but record its
   provenance and do not claim a strict data-budget comparison without checking
   its training data. Vocabulary size must include added special tokens.
4. Import each category as an MDAT subdataset; preserve disjoint train/validation
   membership as saved views. Register a 512-token strategy using Italian sentence
   segmentation. Audit raw words, cached tokens, chunk counts, and maximum IDs.
5. Save dataset/tokenizer revisions, hashes, split membership, seed, and actual
   counts with the experiment. Use local caches on compute nodes.

Proposed first model (a starting configuration, not a tuned recipe):

| Setting | Initial value |
| --- | --- |
| Objective | Autoregressive |
| Transformer | 6 layers, hidden size 384, 6 attention heads |
| FFN | SwiGLU, factor 3 |
| Normalization / positions | Pre-RMSNorm / ALiBi |
| Attention | Existing FlashAttention variable-length path, after kernel smoke test |
| Sequence length | 512 |
| Optimizer | AdamW, learning rate 3e-4, weight decay 0.01 |
| Schedule | 5% warmup, cosine decay to 3e-5 |
| Precision | bf16 mixed |
| Initial hardware | 1 GPU on 1 node |
| Batch | 8 sequences per GPU, accumulation 4; adjust after memory measurement |
| Seed | 27 |
| First full-data pass | 1 epoch; more epochs are a separate quality experiment |

Before full-data training, run a bounded 20-update smoke test, then a checkpoint
interruption/resume comparison with the same data, batch size, scheduler horizon,
and world size. The current `--max-steps` does **not** bound execution; fix that
before using it for the smoke test. Compare the resumed document/chunk stream,
optimizer and scheduler state, progress counters, loss, and final weights against
an uninterrupted run. Disable stochasticity for the first deterministic comparison,
then test restoration of per-rank random-number-generator state.

Only after the single-GPU checks pass: test 2 GPUs on one node, then 2 nodes.
Compare token-weighted loss/gradients against a single-process reference with the
same effective batch. Include uneven sequence lengths, short final batches,
unequal rank exhaustion, and a rank with no real examples remaining.

## Leonardo constraints

[CINECA's Leonardo documentation](https://docs.hpc.cineca.it/hpc/leonardo.html)
lists `boost_usr_prod` normal jobs at 24 hours, `boost_qos_dbg` at 30 minutes,
and `boost_qos_lprod` at four days subject to its limits and account eligibility.
The 24-hour limit is therefore a partition/QoS limit, not a universal process limit.
The Booster nodes have four A100 GPUs with 64 GiB each.

Use the debug QoS for the first bounded GPU test. Confirm the project's actual
Slurm account and Python/CUDA environment before building the submission script.
For production, save periodically on persistent project storage and implement a
pre-time-limit signal that requests a checkpoint at an optimizer boundary across
all ranks. Write to a temporary checkpoint and rename after success; keep the
previous complete checkpoint. Verify restart before depending on it for a long job.
An `sbatch --signal` directive alone does not implement recovery.

## Blocking and high-priority findings

Findings below are based on source inspection, not successful runtime reproductions.

| Priority | Location | Finding and consequence |
| --- | --- | --- |
| P0 | `.gitignore`; `matformer/matformer_registry.py:1`; `matformer/models.py:9` | `matformer/utils` is absent, including registry implementation, cache, initialization, and masking. The broad `utils/` ignore rule matches it. Imports cannot succeed in this checkout. Recover the original files before attempting model execution; do not substitute guessed implementations for the ALBERTURIN code. |
| P1 | `matformer/data/mdat/preprocessing.py:233`; `matformer/matformer_tokenizers.py:21` | Preprocessing calls `MatformerTokenizer(tokenizer_type=..., tokenizer_name=...)`, but the constructor dispatches on `tokenizer`, which defaults to `None`. It falls through to `tokenizer.vocab_size`. Cached-data tests do not exercise creation of fresh token caches. Add a real tokenizer/preprocessing integration test. |
| P1 | `matformer/models.py:204` | Autoregressive shifting happens after flattening the batch. Unpadded sequences introduce EOS-to-next-sequence-BOS targets; padded rows also introduce cross-row targets. Shift within each sequence and exclude boundaries explicitly, including fused-loss paths. |
| P1 | `matformer/training/prepare_training.py:53`; `matformer/training/training_loop.py:20` | `--max-steps` changes the planned step count but the loop never enforces it. A purported short test can run through the whole corpus. |
| P1 | `matformer/training/training_loop.py:20` | Trailing incomplete accumulation windows are not stepped. Their gradients can spill into the next epoch. Flush with correct normalization and synchronization, or explicitly drop/clear the window and reflect that policy in step counts. |
| P1 | `matformer/training/training_loop.py:20,67` | The epoch loop starts at zero on resume. Final save uses estimated `total_steps` and the original epoch rather than actual progress. Repeated epochs and resumed runs can produce incorrect checkpoint progress/filenames. |
| P1 | `matformer/models.py:335` | Reset uses legacy keys (`optimizer_states`, `lr_schedulers`, `MatformerDataModule`, `global_step`), while save/load uses `optimizer`, `scheduler`, `datamodule`, `step`. Requested resets therefore leave current states intact. |
| P1 | `matformer/models.py:420` | Saving calls `scheduler.state_dict()` when scheduler is `None` (scheduling disabled), which raises. Random-number-generator states are not included. |
| P1 | `matformer/training/prepare_training.py:53` | Per-rank step counts are divided by `num_nodes` again. Scheduler horizon is also per epoch although the loop can run multiple epochs. |
| P1 | `matformer/data/matformer_data_module.py:109`; `matformer/models.py:132` | Collation ORs end-of-rank sentinels across the batch. One sentinel causes pretraining to return zero for the entire batch, discarding real examples alongside it. |
| P1 | `matformer/models.py:210,249`; `matformer/training/logger.py:12` | Local mean losses are not explicitly normalized by global valid-target counts. Logging does not reduce those counts/loss sums across ranks. Exception recovery can independently skip a rank's computation and is unsafe to assume correct under DDP. |
| P1 | `matformer/data/matformer_data_module.py:258` | World-size changes warn but still index per-rank saved states without a remapping algorithm. Exact resume should reject unsupported changes rather than silently duplicate/drop data or index beyond the saved ranks. |
| P2 | `matformer/training/ckpt_manager.py:78`; `matformer/models.py:418` | Checkpoint paths are logged before save success, and each rank can append to the same resume CSV. Review atomicity, single-writer bookkeeping, and recovery from truncated latest files. |
| P2 | `pyproject.toml` | README filename case does not match `readme.md`; declared Python >=3.6 is incompatible with newer syntax/features; `lightning.fabric` is imported but only `pytorch-lightning` is declared; optional `transfomer_engine` is misspelled. Reproducible installation needs its own check. |

## Remaining requested work

- **Pretokenized data:** retain the branch's domain/view/token-budget selection.
  Add tokenizer identity and vocabulary checks, explicit imported-token formats,
  and raw-word accounting. Keep saved split membership independent of selection
  seeds. Exercise both fresh preprocessing and cached loading.
- **Sentence packing:** `SmallSplitter._process` already groups consecutive
  sentences within each document and slices oversized sentences. The training
  collator concatenates sequences for variable-length attention and preserves
  `cu_seqlens`. This is distinct from filling fixed-length blocks across multiple
  documents. Measure utilization and token coverage before introducing a new packer;
  specify document-boundary attention and label masking together.
- **General model review:** complete after the missing utilities are restored.
  Prioritize padding/attention masks, positional encodings, causal target alignment,
  fused/unfused parity, registry weight mappings, and save/load round trips.
- **Instruction tuning:** add a separate supervised dataset/collator with chat
  templates, prompt/padding labels set to -100, assistant-only target masks, and
  sequence-boundary-safe shifting. Preserve those masks through unpadding and
  fused losses. Do not silently reuse the current pretraining targets.
- **Looped models:** existing recurrence hooks carry state across chunks; that is
  not the same as repeatedly applying shared transformer layers to one sequence.
  Define shared weights, iteration count, state initialization/reset, gradient
  truncation policy, and inference behavior before implementing a looped model.
- **Annealing:** `training/lr_scheduler.py` already has cosine/linear/polynomial
  schedules and `annealing_start_step`/`annealing_lr`. Audit continuity, the step-zero
  case, optimizer-group ratios, and resumed scheduling before adding another API.
- **Hugging Face final batch:** not reproduced. The reviewed collators derive
  tensor batch size from their inputs; the missing attention cache could be relevant,
  but the root cause is unknown. Reproduce using full batch -> short batch -> batch
  of one, with varying lengths, both padding modes, and inference/training wrappers.
  Do not use `drop_last=True` as the fix. Also inspect attention-mask forwarding in
  HF wrappers and discarded `padded_sequence.unpad()` return values in the
  classification collator; these are separate observations, not a proven explanation
  for the reported crash.

## Verification and information needed

- All 66 Python files under `matformer` plus the training entry point parsed under
  Leonardo's `python/3.11.7` module (66 files total).
- Existing MDAT regression tests were attempted but failed during import because
  that environment lacks NumPy; the default shell also has no PyTorch, Lightning,
  LMDB, Transformers, or datasets installation. This is not a passing test run.
- The default system Python is 3.6.8. The Python 3.11.7 module is available.
- Required next inputs: location of the original `matformer/utils` sources and the
  working Leonardo Python environment. Dataset access and Slurm account also need
  verification before a download/job can be run.

Implementation order: restore runnable source/environment -> fresh MDAT pipeline
and causal loss -> bounded loop/checkpoint/resume -> one-GPU BabyLM -> distributed
validation -> downstream HF regression -> instruction tuning / looped experiments.
