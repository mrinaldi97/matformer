---
title: Configuration
sources:
  - matformer/matformer_configs.py
  - matformer/dataclasses/matformer_config.py
  - matformer/utils/matformer_assets.py
  - matformer/sweeps.py
---
# Configuration

Matformer contains a complex configuration system that is able to control in an integrated manner all the framework's components.
Classes that requires a configuration must have their dataclass. This dataclass is automatically discovered by the [[components/registry | MatformerRegistry]] with the decorator `@register_dataclass`.
The `MatformerConfigs` class is then able to load YAML/JSON config file(s) into a tree of typed `MatformerConfig` dataclasses.
Because dataclasses are very flexible and able to configure almost every aspect of the framework, you can use the [[configuration/index | configuration reference]] to explore all the configuration's options for anything registered in the [[components/registry | MatformerRegistry]].

For example, let's assume the YAML contains:

```yaml
data:
  - name: train
    type: HFDataset
  - name: test
    type: HFDataset
```

This means that `MatformerConfigs` will look in the registry for a dataclass for the category `data` and of variant `HFDataset`, which is addressed by the special `type` key. If the `type` key is missing, `MatformerConfigs` will use the category's default (`default=True` in the registry).

Another special, optional, key is `name`, as it is specialized for giving a unique alias to a specific object in the Configs system: this is necessary when the objects present in the configuration compose with each other, for example when a specific evaluation task refers to a specific dataset on which the task is evaluated.

When a `MatformerConfigs` is created or updated, it is possible to provide several files (ex. a list of YAMLs). In this case, the order may be important: if a section is declared more than once, its existing fields are patched from left to right, the latest imported file wins. Files may also include other files by declaring a special section `includes`. Includes are the first element resolved, so they are lowest priority. Highest priority are `overrides`, which can be set at any moment and give the user a final decision over the `MatformerConfigs` object.

An important distinction is that some categories only allow for a single configuration object; for example, it is possible to have only a single `model` section and so if more than one `model` section is present, their content get merged as explained above. Some categories, however, may allow for more than one configured object, as it is the case for the `data` and `task` sections. They are distinguished because their dataclass inherit from `MatformerMultiConfig` instead of `MatformerConfig`. When it is initialized, `MatformerConfigs` automatically checks if there are inconsistencies (clearly, all the variants of a category must either be all grouped or all listed).

It is particularly for `MultiConfigs` dataclasses that the `name` field exists, in order to avoid ambiguities. In this way, it is possible to have named object in the configuration tree, and those objects can refer to each other. This behaviour is governed by the dataclass itself, thanks to three `ClassVar`:

```python
_ref_fields: ClassVar[dict] = {"ref_section": "section_name"}
```

`_ref_fields` in this case is saying to `MatformerConfigs` that the configuration's key `ref_section` must contain the reference to an object registered in the section `section_name`. For example, `_ref_fields: ClassVar[dict] = {"ref_data": "data"}` means that the configuration of this object should contain a field called `ref_data` containing the name of a (named) object insterted in the `data` section.
If this is the data section:

```yaml
data:
  - name: train
    type: HFDataset
  - name: test
    type: HFDataset
```

this means that this config should contain a `ref_data` key with either `train` or `test` as a value.
The referenced object is automatically looked for by the `MatformerConfigs` systems. If it missing, no errors are raised if the key doesn't appear in the `_mandatory_references: ClassVar[list] = []` classvar; in the counter case, an exception is raised. After the object is looked for, it is automatically linked: this means that calling `my_section.ref_data` will resolve to `data.train`.

```yaml
my_section:
  my_value: 1
  foo_value: bar
  ref_data: train
```

Because this `linking` capability can be useful in some cases, the `linked_fields` classvar acts a syntactic sugar to directly have a specific section linked inside the configuration object.

A `MatformerConfig` object can also be used to rapidly instantiate the configured object. For example, if we have loaded a tokenizer's configuration:

```yaml
tokenizer:
  type: huggingface
  pretrained_name: mrinaldi/Gettone
```

we can get an instance of the actual tokenizer with `configs.tokenizer.instantiate()` (see `instantiate` in [[file:matformer/matformer_configs.py:instantiate | matformer_configs.py]]).

## Generated reference

The reference is rendered from `registry.get_definition(category, variant)` with `matformer.utils.schema`, the same helper used by the GUI and the task console. `fields_schema` gives the name, type, default and options; `field_docs` gives the inline comment; `field_sections` gives the standalone comment above a group of fields. A new field appears in the reference as soon as it is added to the dataclass.

## Validation

The configuration is typed. `from_dict` rejects unknown fields unless `strict=False`, coerces `5e-4` from YAML's string form to a float, and checks that a mandatory reference points to a real entry.

## Sweeps

A value can declare several candidates separated by a pipe, and the file then describes a sweep:

```yaml
training:
  lr: 2e-3 | 1e-4 | 5e-5
model:
  pooling_type: attention | mean
task:
  - name: test
    topk: false | true
```

The YAML parser keeps the pipe inside the scalar. `MatformerConfigs` splits the scalar on top-level pipes and parses each candidate as YAML, so `true` is a bool, `8` is an int, and `positional_encoding: alibi | [alibi, rope]` sweeps between one strategy and two. `\|` is a pipe that means itself.

The configs keep the first candidate, so a config that is not expanded is still one valid run. `sweep_variants()` returns one detached config per combination. Each point gets its own run name, checkpoint name and output names, and its `meta.json` records the values it chose. `check_sweeps()` raises when a swept path does not address a real field.

```bash
python train_model.py --cfg exp.yaml --sweeps     # print the plan and run nothing
python train_model.py --cfg exp.yaml              # run every point
python run_task.py --cfg exp.yaml --sweep 3       # one point
```

The implementation is in `matformer/sweeps.py`: `harvest` records the candidates during import, `plan` builds the product, `select` applies a selection, and `apply` produces the config of one point. `_sweep_scoped` names the fields that must be renamed per point.

## Assets

`MATFORMER_ASSETS` sets one folder outside the repository for configs, checkpoints, datasets, logs, virtual environments, experiments and extra plugins. The folder can be declared in `configuration.yaml` in the repository root, or with the environment variable.

```yaml
# configuration.yaml
assets: /data/matformer_assets
```

The folder is created with `python -m matformer.utils.matformer_assets`. Its area folders are `configs`, `checkpoints`, `venvs`, `logs`, `dataset`, `experiments` and `extra_plugins`.

In YAML values, `includes` and overrides, the tokens `$ASSETS`, `$CONFIG_ASSETS`, `$CHECKPOINT_ASSETS`, `$DATA_ASSETS`, `$LOG_ASSETS`, `$VENV_ASSETS`, `$EXPERIMENT_ASSETS` and `$PLUGIN_ASSETS` expand to the corresponding folders. If no folder is configured, the tokens are not used and relative paths work as usual. See [[intro/env-vars | Environment variables]].
