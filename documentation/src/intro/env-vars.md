---
title: Environment variables
sources:
  - matformer/utils/env_variables.py
---
# Environment variables

Matformer reads a small set of environment variables. They are optional and keep their defaults when unset.

Two families are used often:

* `MATFORMER_<CATEGORY>_<VARIANT>` forces a registry implementation. The name is the registry path upper-cased, with non-alphanumeric characters replaced by underscores. For example, `MATFORMER_ATTENTION_NORMAL=sdpa`.
* `MATFORMER_ASSETS` sets the root of the shared folder outside the repository. In configs, `$ASSETS`, `$DATA_ASSETS`, `$CHECKPOINT_ASSETS` and the other area tokens expand under that root.

The table is generated from `matformer/utils/env_variables.py` and from the variables read in the code.

{{ env_vars }}

The source column links to the file that reads the variable. A variable with no source link is documented in `env_variables.py` and read by a generated or optional component, or by the registry itself.
