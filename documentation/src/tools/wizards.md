---
title: Wizards
sources:
  - matformer/wizards/engine.py
  - matformer/server/wizard_cmd.py
  - matformer/gui/wizard.py
---
# Wizards

A wizard is a TOML document that describes a multi-step interface for a long-running operation.
The definition is renderer independent: the desktop console draws it, and a terminal, a web page or
the documentation build can read the same steps. The server executes the action the answers
describe, using the same job queue as every other long operation.

The definitions live in `matformer/wizards/definitions`. The engine is in
`matformer/wizards/engine.py`; the RPC layer is `matformer/server/wizard_cmd.py`.

## Document

```toml
id = "train"
title = "Train a model"
description = "Build a training configuration and queue it as a job."
category = "Training"

[[steps]]
id = "kind"
title = "Kind of model"
[[steps.fields]]
name = "model_kind"
kind = "radio"
label = "Model"
default = "autoregressive"
options = [
  { value = "autoregressive", label = "Autoregressive", doc = "Predicts the next token." },
  { value = "masked", label = "Masked", doc = "Reconstructs masked tokens." },
]
```

A step is shown when its `when` condition holds. A field is shown when its own `when` holds.
Conditions are structured, not evaluated as code:

```toml
when = { field = "model_kind", is = "masked" }
when = { field = "model_kind", in = ["masked", "autoregressive"] }
when = { field = "validation_enabled", truthy = true }
when = { all = [{ field = "layout", is = "container" }, { field = "mode", is = "new" }] }
```

Field kinds are the field language used by every form in the project: `text`, `path`, `int`,
`float`, `bool`, `choice`, `radio`, `multi`, `json`, `paths` and `mapping`. `advanced = true` moves
a field behind one expander. `required = true` blocks the step until the field has a value.
`source = "mdat.datasets"` fills a choice from live data; `depends_on` names the answers the source
reads.

## Generated configuration steps

A model has dozens of fields. Writing them in TOML would duplicate the dataclasses, so a step can
generate its fields from the registry:

```toml
[[steps]]
id = "model"
generator = "config"
preset = "${mode}_${model_kind}"
sections = ["model", "tokenizer"]
```

The preset names the sections, the default values, the paths that stay focused and the paths that
are required:

```toml
[presets.scratch_autoregressive]
sections = ["model", "tokenizer", "data", "training"]
focus = ["model.hidden_size", "model.num_hidden_layers", "training.lr"]
required = ["tokenizer.pretrained_name", "data.data_root"]

[presets.scratch_autoregressive.variants]
model = "default"
data = "lazy"

[presets.scratch_autoregressive.values]
"model.hidden_size" = 512
"data.batch_size" = 8
"training.learning_signals" = ["autoregressive"]
```

Every other field of the section is still present under Advanced. `extends` merges one preset into
another. `variant_by` selects a variant from an answer, which is how one data step covers both the
`lazy` and `mdat` data modules.

## Actions

The action is declarative. `train` writes the generated YAML and queues it through `train.start`.
`mdat` queues a sequence of MDatCommands calls. `annotate` queues an annotation job.

```toml
[action]
kind = "train"
name = "${training.run_name}"
project = "${training.project_name}"

[[action.commands]]
cmd = "open"
args = { path = "${container}" }

[[action.commands]]
cmd = "pretok_all"
when = { field = "target", is = "all" }
args = { strategy = "${strategy}" }
```

`${answers.NAME}` is substituted with an answer. A command or file with a `when` that does not hold
is dropped. A file can be materialized for a command and addressed as `${files.NAME}`.

## RPC

| Command | Purpose |
|---------|---------|
| `wizard.list` | Every definition with title, category and step count. |
| `wizard.schema` | Visible steps and fields for the answers so far. |
| `wizard.plan` | Validation, generated YAML and the command list. Nothing is queued. |
| `wizard.submit` | Execute the action and return the job id. |
| `wizard.context` | The model loaded on this server, for prefill. |

## Built-in wizards

| Wizard | Work |
|--------|------|
| `train` | Training from scratch or a fine-tune, with data, validation and launch options. |
| `annotate` | Annotation of a file or an MDAT with its own model and job. |
| `mdat_create` | Create an MDAT container. |
| `mdat_dataset` | Create a linked dataset and optionally link it into a container. |
| `mdat_link` | Attach an existing linked dataset to a container. |
| `mdat_ingest` | Import or append JSONL, CSV, LMDB, HuggingFace or Atlas documents. |
| `mdat_pretokenize` | Register a strategy and pretokenize one dataset or every dataset. |
| `mdat_view` | Create a head, range or selection view and shuffle it. |
| `mdat_prepare` | Select a strategy and view, then precompute the distributed chunk layout. |

Every long action becomes a job in the server queue: it keeps running when the console closes, and
its log, status and abort are the queue's usual controls.

## Generated reference

The catalog below is rendered from the same TOML files the GUI loads.

{{ wizard_catalog }}
