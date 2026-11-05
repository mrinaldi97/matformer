# Pretraining work diary

Scope: minimal, modular Italian LLM pretraining on NVIDIA HPC systems. No new
post-training implementation. Preserve pre-existing uncommitted work.

## Checklist

- [x] Audit code, existing changes, dependencies, and tests.
- [x] Add a standalone tokenizer evaluation module and CLI.
- [x] Expose and test looped, weight-sharing transformer settings.
- [x] Restore model, optimizers, schedulers, precision/scaler, per-rank RNG,
      data position, counters, and configuration at safe optimizer boundaries.
- [x] Add atomic checkpoint publication, retention, regular/milestone saves,
      and coordinated signal/wall-clock stopping for 24-hour jobs.
- [x] Verify distributed rank partitioning, equal update counts, launch scripts,
      and strict resume invariants.
- [x] Support pre-tokenized dataset selection by tokens and composition;
      distinguish dataset quotas from lifetime training token limits.
- [x] Audit packing, padding, next-token labels, and incomplete final batches.
- [x] Support bounded-storage dataset stages without resetting training state.
- [x] Make offline W&B and local logging explicit; document later synchronization.
- [x] Remove demonstrably obsolete files while preserving research assets and
      active/user-edited code.
- [x] Run targeted regression tests and document remaining cluster-only checks.

## Findings and actions

### 2026-11-06 — Initial inspection

- Repository has pre-existing changes in data loading, model/loss code, tensor
  transfers, training, cache handling, and BabyLM scripts/tests/docs. These are
  input to this work and will not be reverted.
- No applicable AGENTS.md was found in the repository or ancestor directories.
- Shell sandbox cannot start: `bwrap ... namespace ... ENOSPC`. Read commands
  work through the reviewed escalation path; this is a host sandbox issue,
  not a dataset storage error.
- Existing training has basic optimizer-boundary signal handling and atomic
  model checkpoint writes, but complete state restoration and distributed
  invariants still need audit.
- Existing data selection already supports some domain token quotas; extend
  this rather than introduce an unrelated training stack.
- Research started with official OLMo checkpoint code and CINECA HPC docs.

## Validation record

Pending baseline test run and implementation.

### Audit and first implementation

- Baseline: 42 tests, 39 passed; 3 old Maskerator tests expect two return values
  while the current API returns three. Update the tests to the current contract.
- Fixed mixed real/exhausted batches being entirely discarded; dummy rows now
  participate in DDP forward/backward with zero targets.
- Added a standalone collator with length-based padding masks: EOS remains a
  target when EOS and PAD share an ID. Oversized/empty input fails explicitly.
- Fixed SDPA always applying ALiBi and ignoring padding masks.
- Existing preprocessing already reserves BOS/EOS slots. Added preservation of
  trailing tokens and replaced silent overlength truncation with an error.
- Added num_loops for shared-weight depth loops; cross-document memory hooks are
  distinct and rejected by the exact-resume runner until their caches can be saved.
- Added percentage composition on top of existing whole-document token quotas.
- Added complete checkpoint payloads, atomic publication, snapshot schedules,
  strict configuration checks, per-rank RNG, and explicit data-stage transitions.
- Automatic approval review rejected default deletion of old checkpoints. Revised
  retention to preserve all saves unless keep_last_checkpoints is explicitly set.
  The revised patch was approved; no existing checkpoints were deleted.
- Added offline-by-default W&B segments grouped by persistent run ID plus local
  JSONL logs; W&B offline does not provide online resume semantics.
- Added a standalone offline tokenizer evaluation module with Italian probes.
- Research: Apertus engineering report (arXiv:2604.12973, section IV-B.2)
  reports 250-iteration asynchronous checkpoints and Slurm signals/job chaining.
  This implementation uses synchronous writes for simplicity and durability.

### Distributed validation and cleanup

- The 10 existing BabyLM tests passed after the training changes. The first 10
  new runtime tests also passed, including bit-for-bit next-update equivalence
  after a full save/load and exact stage/global target ceilings.
- The first two-process Gloo comparison differed in one Adam-updated weight by
  ~1.2e-6 because global-batch and distributed reductions sum in different orders.
  Added a direct gradient comparison and a small roundoff tolerance for the
  global-batch comparison; kept exact equality for uninterrupted-vs-resumed runs.
  The revised real DDP test passed, including uneven rank exhaustion and per-rank
  RNG/data payloads. All ranks ended with identical weights and counters.
- Added two-node/four-GPU-per-node Slurm template, offline defaults that survive
  the existing activation script, standard/looped example configs, and a detailed
  pretraining guide citing OLMo, Apertus, Fabric, CINECA and EPICURE sources.
- Removed 51 tracked, unreferenced legacy/archive/backup files from
  `appoggio/legacy`, `configs/legacy`, and old MDAT/attention/debug copies.
  Preserved current configs, notebooks, assets, and existing untracked user files.
- Cleanup helper initially used subprocess's newer `text=` keyword with the
  host Python 3.6; it failed before any deletion. Retried successfully with
  `universal_newlines=True`. Training/tests use the installed Python 3.11 env.
- Source identities now guard against swapping datasets in place on resume.
  Merged chunks keep a single BOS/EOS pair; configured RoPE theta now reaches
  the attention cache rather than silently using its default.
- All retained JSON configs parse, and both Slurm scripts pass `bash -n`.
- Full pytest suite running; physical multi-node NVIDIA/NCCL and signal/timing
  tests remain cluster validation steps, not claimed as completed.

### Full-suite findings

- First full pytest pass: 60 passed / 3 failed. Two previously uncollected
  Maskerator tests used an incomplete UnpaddedTensor constructor and an unreliable
  Bernoulli-rate assertion on only 20 draws. Corrected test inputs, without
  changing masked-model behavior.
- The third failure exposed a new source-identity audit bug: excluded domains
  can have no pretok_db. Skip absent caches when collecting file identities.
- Added failure-injection coverage for checkpoint write errors, world-size/data
  identity rejection, real percentage selection, and merged-chunk token retention.
- Extended the DDP regression to test an exact global target cap and a real
  SIGUSR1 sent to only one rank, followed by a coordinated final checkpoint.

### CLI integration findings

- Full suite after fixes: 65 tests and 13 subtests passed. Extended two-process
  Gloo test passed, including global token caps and coordinated SIGUSR1 saves.
- A separate CLI/offline integration check found Lightning's WandbLogger.finalize
  does not finish the W&B run. Added explicit experiment.finish() so offline
  segments flush and repeated runs in one process do not reuse the prior segment.
- Zero-token/zero-percent domains no longer require a pretokenized cache. Added
  a public-CLI integration test using temporary MDAT files and a local byte-level
  tokenizer, without external downloads or online logging.


### Final verification

- Full CPU suite: 65 tests + 13 subtests passed. After the final CLI/logger and
  selection changes, the focused runtime/data/CLI suite passed 37 tests + 10
  subtests, including the added public-CLI test (66 total distinct tests now).
- Real two-process Gloo validation passed: correct global token-weighted gradients,
  exact uninterrupted/resumed equality, identical ranks, per-rank RNG/data state,
  exact global target ceiling, and SIGUSR1 on one rank causing a shared save/stop.
- Public CLI starts, saves, resumes, and creates separate flushed offline W&B
  segments with a shared run group. A locally trained byte-level tokenizer passed
  all Italian Unicode/whitespace round trips through the tokenizer CLI.
- Final audit fixed a prefetch shutdown race: the reader must finish before its
  queue/event/LMDB is replaced or closed; the old two-second timeout was unsafe
  on a slow shared filesystem. Final dataset regressions passed: 24 tests and
  10 subtests, including zero-percent exclusion without a token cache.
- `git diff --check`, JSON parsing, and Slurm shell syntax checks passed.
- No GPU training jobs, external uploads, Git commits, or existing user
  dataset/checkpoint deletions were performed. Temporary test fixtures were
  cleaned up. Existing user edits remain in the working tree.

## Remaining deployment checks (require an allocated GPU job)

- [ ] Run the documented two-node NVIDIA/NCCL smoke and resume job.
- [ ] Measure maximum update + checkpoint time on shared storage and tune the
      advance signal / wall-clock margin accordingly.
- [ ] Verify Slurm signal delivery and chaining with the project's actual QoS.
- [ ] Validate optional compilation/FlashAttention/fused CUDA kernels and HPC4AI's environment
      before using them for production. CPU DDP tests do not certify these paths.

Intentional limits: fixed total process count for exact resume; synchronous DDP
full checkpoints; whole-document corpus selection can underfill quotas; target
ceilings discard unused targets in the final consumed microbatch; no automatic
corpus transfer/deletion; no new post-training code. The guide explains these
contracts and the legacy checkpoint migration path.
