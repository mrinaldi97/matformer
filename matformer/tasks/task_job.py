# matformer/tasks/task_job.py
"""The queue entry for *any* task: one implementation, several modes.

An annotation was only one way to run a task over data. The same job now also runs the task as a
plain dataset pass (`run`), one or more texts (`text`), or a run that additionally pools its
activation store into vectors. Every mode loads its own model and config, so the console can
close; the queue, the log and abort are the same for all of them.

Spec:
  mode       annotate (default) | run | text
  model, cfg, tokenizer, device, overrides, task, presentation
  annotate:  input, output, format, column, label, dataset, keys_only, limit, batch_size
  run:       vectorize {...} (optional), label_key
  text:      text or texts, output (optional JSONL of rendered answers)
"""
from __future__ import annotations


def _configs(spec):
    from matformer.matformer_configs import MatformerConfigs
    files = spec.get("cfg") or []
    files = [files] if isinstance(files, str) else list(files)
    configs = MatformerConfigs.create_configs(files)
    if spec.get("tokenizer"):
        configs.import_config({"tokenizer": {"pretrained_name": spec["tokenizer"]}})
    return configs


def _overrides(spec, task):
    overrides = list(spec.get("overrides") or [])
    if task == "generation":
        if spec.get("max_new_tokens") is not None:
            overrides.append(f"task.generation.max_new_tokens={int(spec['max_new_tokens'])}")
        if spec.get("temperature") is not None:
            overrides.append(f"task.generation.temperature={float(spec['temperature'])}")
        if spec.get("mode") != "text":
            overrides.append("task.generation.stream=false")
    return overrides


def _finish(loaded, spec):
    """Flush and close the stores the config mounted, then pool them into vectors when asked.
    Every mode that ran a forward shares this tail."""
    mounted = list((getattr(loaded["model"], "plugins", {}) or {}).values())
    stores = [plugin for plugin in mounted
              if hasattr(plugin, "databank") and hasattr(plugin, "available_addresses")
              and hasattr(plugin, "flush")]
    for plugin in stores:
        plugin.flush()
    vector = spec.get("vectorize") or {}
    print(f"stores: {len(stores)}; vectorize: {vector.get('enabled')} -> {vector.get('filename')}",
          flush=True)
    if vector.get("enabled") and vector.get("filename") and stores:
        from matformer.vectors.pooling import pool_activation_store
        vector.setdefault("source", getattr(getattr(stores[0], "config", None), "filename", None))
        saved = pool_activation_store(stores[0], vector, spec.get("label_key") or "label",
                                      spec.get("model"))
        print(f"vectors: {saved} rows -> {vector['filename']}", flush=True)
    for plugin in stores:
        plugin.close()


def execute(spec: dict) -> None:
    import torch
    from matformer.matformer_wrapper import load_model
    from matformer.tasks.session import TaskSession, format_rows

    device_name = spec.get("device")
    if not device_name or str(device_name).lower() == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    configs = _configs(spec)
    loaded = load_model(path=spec.get("model"), map_location=torch.device(device_name),
                        dtype=torch.bfloat16, inference=True, configs=configs)
    task = spec.get("task") or ("forward" if spec.get("mode") == "run" else "evaluation")
    session = TaskSession(loaded["model"], loaded["configs"], task=task,
                          overrides=_overrides(spec, task), source=spec.get("model"),
                          presentation=spec.get("presentation"))
    mode = spec.get("mode") or "annotate"

    if mode == "run":
        print(f"task {task} over the dataset", flush=True)
        summary = session.run()
        print(f"run complete: {summary}", flush=True)
        _finish(loaded, spec)
        return

    if mode == "text":
        texts = spec.get("texts") or ([spec["text"]] if spec.get("text") else [])
        if not texts:
            raise ValueError("mode 'text' needs text or texts")
        rendered = []
        for text in texts:
            rows = session.feed(text, stream=False, save=bool(spec.get("save")))
            line = session.task.present(rows)
            if line is None:
                line = format_rows(rows, False)
            print(f"\n=== {text!r} ===\n{line}\n{session.status()}", flush=True)
            rendered.append({"text": text, "rendered": line, "status": session.status()})
        if spec.get("output"):
            import json
            with open(spec["output"], "w") as handle:
                for row in rendered:
                    handle.write(json.dumps(row) + "\n")
            print(f"answers written to {spec['output']}", flush=True)
        _finish(loaded, spec)
        return

    # annotate: predictions over a file, a CSV or an MDAT dataset
    summary = session.annotate(
        spec["input"], output_path=spec.get("output") or None,
        column=spec.get("column") or "text", label=spec.get("label") or None,
        limit=spec.get("limit") or None, batch_size=spec.get("batch_size") or None,
        output_format=spec.get("format") or None, keys_only=bool(spec.get("keys_only")),
        mdat_dataset=spec.get("dataset") or None)
    print(f"annotation complete: {summary.get('rows')} rows -> {summary.get('output')}", flush=True)
    for key, value in summary.items():
        if key not in ("rows", "output"):
            print(f"  {key}: {value}", flush=True)
    _finish(loaded, spec)
