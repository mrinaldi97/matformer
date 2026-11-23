"""W&B metrics and local progress for one experiment across process restarts.

The training process never contacts W&B in offline mode. Each invocation keeps
its own file; the uploader maps those files to the checkpoint's persistent ID.
"""
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from matformer.monitoring.wandb_charts import chart_values, define_charts


def write_json(path, value, *, durable=False):
    """Publish metadata atomically beside its destination, including on WORK."""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w') as stream:
            stream.write(json.dumps(value, sort_keys=True) + '\n')
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        if durable:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class WandbLogger:
    """Collect metrics by training step; keep W&B's transport steps independent.

    A checkpoint rollback may replay optimizer steps. Keep that attempted work
    with its segment ID instead of dropping it or pretending the model advanced.
    Charts use trainer/global_step. Explicit W&B row numbers remain unique even
    after a producer crash or re-upload; reserving small blocks avoids a disk
    sync on every update. Unused row numbers are harmless gaps after a restart.
    """
    def __init__(self, log_dir, run_id, config, *, mode, step=0, tokens_seen=0,
                 resumed=False, settings=None):
        import wandb

        self.directory = Path(log_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / '.wandb-producer.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError('Another training process is writing this experiment') from None
        self.segment_id = uuid.uuid4().hex[:12]
        self.pending, self.pending_step = {}, None
        self.closed = False
        self.cursor_path = self.directory / 'wandb_cursor.json'
        if resumed and not self.cursor_path.exists():
            self.lock.close()
            raise FileNotFoundError('Single-run W&B resume requires its logs directory, including '
                                    'wandb_cursor.json, alongside the checkpoint. Restore those '
                                    'files; legacy grouped histories need explicit migration.')
        cursor = json.loads(self.cursor_path.read_text()) if self.cursor_path.exists() else {}
        self.next_row = self.reserved_until = cursor.get('reserved_until', 0)
        if isinstance(self.next_row, bool) or not isinstance(self.next_row, int) or self.next_row < 0:
            self.lock.close()
            raise ValueError('Invalid W&B history cursor')
        if not self.cursor_path.exists():
            write_json(self.cursor_path, {'reserved_until': 0}, durable=True)
        previous_path = self.directory / 'training_status.json'
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        previous_step = previous.get('global_step', step)
        self.status = dict(run_id=run_id, segment_id=self.segment_id,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'), phase='starting',
            global_step=step, tokens_seen=tokens_seen, checkpoint_step=step if resumed else None,
            resumed_from_step=step if resumed else None,
            rollback_from_step=previous_step if resumed and previous_step > step else None,
            last_training_update=None, updated_at=time.time(), primary_charts_version=1)
        self.last_status_write = 0.0
        project = config.get('wandb_project', 'matformer')
        entity = config.get('wandb_entity') or os.environ.get('WANDB_ENTITY')
        manifest_path = self.directory / 'wandb_run.json'
        manifest = dict(format_version=1, run_id=run_id, project=project, entity=entity, mode=mode)
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            self.lock.close()
            raise ValueError('The existing W&B experiment identity/destination differs')
        # Unique source IDs also allow retaining individual fragments for audit.
        source_id = run_id if mode == 'online' else f'{run_id}-{self.segment_id}'
        self.run = wandb.init(id=source_id, group=run_id,
            name=config.get('wandb_run_name', 'matformer'), project=project, entity=entity,
            mode=mode, resume='allow' if mode == 'online' else None,
            dir=str(self.directory), config=config, settings=settings)
        self.run.define_metric('trainer/global_step')
        self.run.define_metric('*', step_metric='trainer/global_step', step_sync=True)
        define_charts(self.run)
        if mode == 'offline':
            self.status['offline_file'] = str((Path(self.run.dir).parent / f'run-{source_id}.wandb').resolve())
        write_json(manifest_path, manifest)
        self.update_status()

    def log(self, metrics, step):
        step = self.status['global_step'] if step is None else step
        if self.pending_step is not None and step != self.pending_step:
            self.flush()
        self.pending_step = step
        self.pending.update(metrics)

    def flush(self):
        if not self.pending:
            return
        if self.next_row >= self.reserved_until:
            self.reserved_until = self.next_row + 256
            # Reserve before logging: a killed producer must never reuse rows
            # which may already be visible in the cloud beyond its checkpoint.
            write_json(self.cursor_path, {'reserved_until': self.reserved_until}, durable=True)
        tokens = self.pending.get('train/tokens_seen')
        if tokens is None and self.pending_step == self.status['global_step']:
            tokens = self.status['tokens_seen']
        primary = chart_values(self.pending, self.pending_step, tokens)
        payload = {**self.pending, **primary, 'trainer/global_step': self.pending_step,
            'training/segment_id': self.segment_id,
            'training/resumed_from_step': self.status['resumed_from_step'],
            'training/slurm_job_id': self.status['slurm_job_id']}
        if primary:
            # Validation is a separate history row: pair it with the unchanged
            # count of training targets, not val/tokens or padded token slots.
            payload.setdefault('train/tokens_seen', tokens)
        self.run.log(payload, step=self.next_row, commit=True)
        self.next_row += 1
        self.pending, self.pending_step = {}, None

    def advance(self):
        """Called only after a successful optimizer update."""
        self.status.update(global_step=self.pending_step if self.pending_step is not None
                           else self.status['global_step'], phase='training',
                           last_training_update=time.time())
        if 'train/tokens_seen' in self.pending:
            self.status['tokens_seen'] = self.pending['train/tokens_seen']
        self.flush()
        if time.monotonic() - self.last_status_write >= 10:
            self.update_status()

    def update_status(self, **fields):
        self.status.update(fields, updated_at=time.time())
        write_json(self.directory / 'training_status.json', self.status)
        self.last_status_write = time.monotonic()

    def close(self):
        if self.closed:
            return
        try:
            self.flush()
            self.update_status()
            self.run.finish(exit_code=1 if self.status['phase'] == 'failed' else 0)
        finally:
            self.lock.close()
            self.closed = True
