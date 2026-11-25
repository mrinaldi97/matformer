# matformer/server/task_jobs.py
"""Queue a task job and expose the same thing over RPC.

One entrypoint (matformer.tasks.task_job) covers annotation, a dataset pass, a text and a collect
run; the mode in the spec picks the branch. Wizards and clients go through `enqueue_task`, so a
new caller never invents a second job shape."""
from datetime import datetime
from pathlib import Path

from matformer.server.core import rpc
from matformer.server.jobs import enqueue


def enqueue_task(spec: dict, name=None, project=None) -> str:
    mode = spec.get("mode") or "annotate"
    if mode not in ("annotate", "run", "text"):
        raise ValueError(f"unknown task mode '{mode}' (annotate, run, text)")
    if mode == "annotate" and not spec.get("input"):
        raise ValueError("annotation needs an input file or MDAT")
    if mode == "text" and not (spec.get("text") or spec.get("texts")):
        raise ValueError("text mode needs text or texts")
    if not spec.get("cfg") and not spec.get("model"):
        raise ValueError("a task job needs a model or config files")
    default_name = (Path(str(spec.get("input") or "task")).stem if mode == "annotate"
                    else {"run": "task-run", "text": "task-text"}[mode])
    label = name or spec.get("name") or default_name or "task"
    if not label:
        label = f"task-{datetime.now():%H%M%S}"
    return enqueue("task", label, project or mode, spec, "matformer.tasks.task_job:execute")


@rpc("task.enqueue")
def task_enqueue(conn, p):
    """Queue any task over data or text: see matformer.tasks.task_job for the modes.
    Returns the job id; follow it with jobs.log/jobs.watch like every other job."""
    spec = {key: value for key, value in p.items() if key not in ("name", "project")}
    job = enqueue_task(spec, name=p.get("name"), project=p.get("project"))
    return {"job": job, "name": p.get("name"), "mode": spec.get("mode") or "annotate"}
