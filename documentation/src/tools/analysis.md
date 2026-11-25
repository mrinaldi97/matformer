---
title: Analysis, vectors and steering
sources:
  - matformer/vectors/__init__.py
  - matformer/vectors/pooling.py
  - matformer/server/store_cmd.py
  - matformer/server/vector_cmd.py
  - matformer/server/collect_cmd.py
  - matformer/server/plugin_cmd.py
  - matformer/server/task_jobs.py
  - matformer/gui/analysis.py
  - matformer/gui/activationpage.py
  - matformer/gui/steeringpage.py
  - matformer/gui/internalspage.py
  - matformer/tasks/analysis.py
---
# Analysis, vectors and steering

Three views share one idea: a flat or tabular result is only useful if it can be read as a table,
a distribution, a relation and an outlier list without changing tool.

## The analysis view

`matformer/gui/analysis.py` parses the flat keys the analysis tasks produce
(`weights/.../kurtosis`, `activations/.../std`) and the time series a run stores
(`train/loss` over steps). It offers:

* **Table**: the values, filterable by kind, metric and subject path;
* **Histogram**: the selected metric over every subject;
* **Scatter**: the metric across layers (weight/activation analysis) or steps (run metrics);
* **Outliers**: the same values ranked by distance from the mean, with their full key.

The task page shows it after **Run** (the *Analysis* tab), and *Runs & Plots → Analyze metrics…*
opens exactly the same view over a run's metric history. Static weight analysis therefore becomes
a distribution of kurtosis per layer, not a wall of JSON.

## Collecting activations and vectors

The [[tools/wizards | collect wizard]] queues one forward over the dataset and writes two things:

1. an **activation store** (LMDB or DuckDB) with every selected socket, all positions, padding
   removed, and the dataset's own metadata attached to each sequence;
2. optionally, a **vector store**: one pooled vector per sequence (mean, last, first, max or
   selected token positions), labelled with the metadata key the dataset provides. A contrastive
   dataset therefore produces a labelled prototype set in the same run.

`collect.start` queues the job from an already-mounted config and `collect.vectorize` re-pools an
existing store without another forward. A store can be opened read-only for inspection with
`store.inspect` and then visualised as a summary, histogram or PCA scatter in
*Stores · activations*.

## Vectors are not only steering

`matformer.vectors` stores a named vector with a kind (`steering`, `embedding`, `prototype`,
`probe`, `vector`), an optional label, its address, a metric, free tags and metadata. The same
physical store keeps all of them in one databank; the RPCs are `vectors.save`, `vectors.list`,
`vectors.get`, `vectors.delete`, `vectors.nearest`, `vectors.classify`, `vectors.compare`,
`vectors.export` and `vectors.import`.

* a **steering** record is mounted by the `steering` plugin on a socket (positive `alpha` pushes
  towards it, negative ablates it); the plugin reads the direction from the store itself;
* an **embedding** or **prototype** is classified by cosine similarity, top-k vote and the
  per-label best score — no second model is trained to read the store;
* `vectors.embed` turns texts into pooled vectors at a socket, optionally saving them with a
  label and name.

The *Stores · vectors* page lists, saves, applies and classifies records; both store pages share
the same file/backend/databank bar and the last opened store.

## Plugins and interventions

`plugin.*` mounts and unmounts any registered plugin; `param.*` edits, freezes, ablates and
restores stable weights. A plugin mounted by the model's own config is marked `origin=model`;
anything attached later is `origin=runtime`, and task pages show a banner because the answer is
no longer only the model's. `steer.apply` is a one-call steering mount from a stored vector.

## Jobs

`task.enqueue` is one entry point for every task mode (`annotate`, `run`, `text`); the collect
wizard and annotation wizard both go through it. Job progress, logs and abort remain the queue's
business (`jobs.*`).
