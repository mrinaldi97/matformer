"""Primary training charts, independent of W&B's transport history counter.

Aliases leave all original metrics intact and allow both axes simultaneously.
The completed-run backfill appends only this namespace, never old metric keys.
"""
import argparse
from collections import Counter
import fcntl
import json
import math
from pathlib import Path
import subprocess
import uuid


TOKENS = 'primary/tokens_seen'
STEP = 'primary/trainer_global_step'
# (chart metric, original metric, horizontal axis)
CHARTS = (
    ('primary/train_loss_vs_tokens', 'train/token_loss', TOKENS),
    ('primary/train_loss_vs_step', 'train/token_loss', STEP),
    ('primary/val_loss_vs_tokens', 'val/loss', TOKENS),
    ('primary/val_loss_vs_step', 'val/loss', STEP),
    ('primary/val_ppl_vs_tokens', 'val/perplexity', TOKENS),
)


def define_charts(run):
    for axis in (TOKENS, STEP):
        run.define_metric(axis, hidden=True)
    for name, _, axis in CHARTS:
        run.define_metric(name, step_metric=axis, step_sync=False)


def chart_values(metrics, step, tokens_seen):
    """Pair each observation with its exact optimizer step and trained targets."""
    values = {name: metrics[source] for name, source, _ in CHARTS if source in metrics}
    if not values:
        return {}
    if step is None or tokens_seen is None:
        raise ValueError('Primary loss charts require the matching step and token count')
    return {**values, STEP: step, TOKENS: tokens_seen}


def completed_history(path):
    """Join scalar JSONL records by contiguous optimizer step, without sampling.

    Legacy JSONL has no segment IDs. Refuse rollbacks instead of guessing which
    attempted update an observation belongs to. Live logging preserves them.
    """
    rows, pending, step = [], {}, None

    def flush():
        values = chart_values(pending, step, pending.get('train/tokens_seen'))
        if values:
            rows.append(values)

    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            current = record.pop('step')
            if step is not None and current < step:
                raise ValueError('Legacy JSONL contains a rollback; segment-aware migration is required')
            if step is not None and current != step:
                flush()
                pending = {}
            step = current
            pending.update(record)
    flush()
    if not rows:
        raise ValueError('No token-weighted training or validation metrics found')
    return rows


def cloud_chart_counts(run):
    """Count actual paired observations, including sparse/empty API rows.

    Scan once without key filtering: exported history and the live tail can
    handle missing columns differently. Neither nulls nor absent keys are data.
    """
    counts = Counter()
    for row in run.scan_history(page_size=1000, use_cache=False):
        for metric, _, axis in CHARTS:
            values = (row.get(metric), row.get(axis))
            if all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                counts[metric] += 1
    return dict(counts)


def backfill(directory):
    """Add the primary charts to one completed cloud run, with recoverable sync.

    Both locks exclude live producers/uploaders. Reserve transport rows through
    the existing cursor before writing; a receipt makes repeated sync idempotent.
    A crash during file creation fails closed and needs explicit inspection.
    """
    import wandb
    from matformer.training.wandb_logging import write_json

    directory = Path(directory).resolve()
    with (directory.parent.parent / '.wandb-uploader.lock').open('a') as uploader_lock, \
            (directory / '.wandb-producer.lock').open('a') as producer_lock:
        for lock in (uploader_lock, producer_lock):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Training or its uploader is still active; retry after completion') from None
        status = json.loads((directory / 'training_status.json').read_text())
        if status.get('phase') != 'completed':
            raise ValueError('Chart backfill requires completed training')
        if status.get('primary_charts_version') == 1:
            print('This run already logged primary charts during training.', flush=True)
            return
        metadata = json.loads((directory / 'wandb_run.json').read_text())
        if metadata.get('format_version') != 1 or metadata['run_id'] != directory.name:
            raise ValueError('Invalid experiment manifest')
        destination = '/'.join(metadata[key] for key in ('entity', 'project', 'run_id'))
        api = wandb.Api(timeout=30)
        remote = api.run(destination)
        receipt_path = directory / 'wandb_charts_backfill.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        metrics_path = directory / 'metrics.jsonl'
        size = metrics_path.stat().st_size
        if receipt and receipt['metrics_size'] != size:
            raise ValueError('Training history changed after backfill; do not duplicate old charts')
        if receipt.get('state') == 'synced':
            print(f'Primary charts already verified: {destination}', flush=True)
            return receipt
        if receipt.get('state') == 'writing':
            raise RuntimeError('Interrupted chart creation: inspect the recorded fragment before retrying')
        if not receipt:
            rows = completed_history(metrics_path)
            counts = dict(Counter(key for row in rows for key, _, _ in CHARTS if key in row))
            cursor_path = directory / 'wandb_cursor.json'
            start = json.loads(cursor_path.read_text())['reserved_until']
            if isinstance(start, bool) or not isinstance(start, int) or start < 0:
                raise ValueError('Invalid W&B history cursor')
            # Original history must have reached the cloud before derived rows.
            if remote.summary.get('trainer/global_step') != status['global_step']:
                raise ValueError('Upload the completed original history before adding charts')
            source_id = f'{metadata["run_id"]}-charts-{uuid.uuid4().hex[:12]}'
            receipt = dict(state='writing', source_id=source_id, metrics_size=size,
                           first_row=start, rows=len(rows), counts=counts,
                           original_name=remote.name, original_config=remote.config,
                           original_summary={key: value for key, value in remote.summary.items()
                                             if not key.startswith('_')},
                           original_step=status['global_step'])
            write_json(receipt_path, receipt, durable=True)
            write_json(cursor_path, {'reserved_until': start + len(rows)}, durable=True)
            run = wandb.init(id=source_id, group=metadata['run_id'], name=remote.name,
                entity=metadata['entity'], project=metadata['project'], mode='offline',
                dir=str(directory), config=remote.config,
                settings=wandb.Settings(x_disable_stats=True, disable_code=True, console='off'))
            try:
                # Preserve the existing default axes as well as their values.
                run.define_metric('trainer/global_step')
                run.define_metric('*', step_metric='trainer/global_step', step_sync=True)
                define_charts(run)
                for index, row in enumerate(rows, start):
                    run.log(row, step=index, commit=True)
                # A synced fragment replaces the run summary. Carry original
                # user fields forward without appending them to history again.
                run.summary.update(receipt['original_summary'])
                offline_file = str(Path(run.dir).parent / f'run-{source_id}.wandb')
            finally:
                run.finish()
            receipt.update(state='ready', offline_file=offline_file)
            write_json(receipt_path, receipt, durable=True)
        subprocess.run(['wandb', 'beta', 'sync', '--yes', '--entity', metadata['entity'],
            '--project', metadata['project'], '--id', metadata['run_id'], receipt['offline_file']],
            check=True, timeout=300)
        api.flush()
        remote = api.run(destination)
        counts = cloud_chart_counts(remote)
        for metric, _, _ in CHARTS:
            count = counts.get(metric, 0)
            if count != receipt['counts'].get(metric, 0):
                raise RuntimeError(f'Cloud chart count mismatch for {metric}: {count}')
        if remote.name != receipt['original_name'] or remote.config != receipt['original_config']:
            raise RuntimeError('Unexpected run metadata change after chart sync')
        if remote.summary.get('trainer/global_step') != receipt['original_step']:
            raise RuntimeError('Unexpected original training-step change after chart sync')
        if any(remote.summary.get(key) != value for key, value in receipt['original_summary'].items()):
            raise RuntimeError('Unexpected original summary change after chart sync')
        receipt['state'] = 'synced'
        write_json(receipt_path, receipt, durable=True)
        print(f'Primary charts verified in {destination}: {receipt["counts"]}', flush=True)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log_directory', type=Path, help='Completed experiment logs/RUN_ID on WORK')
    args = parser.parse_args()
    if not args.log_directory.resolve().is_relative_to(Path.cwd().resolve()):
        parser.error('Keep experiment logs under the project checkout')
    backfill(args.log_directory)


if __name__ == '__main__':
    main()
