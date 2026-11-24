---
title: Documentation system
sources:
  - documentation/scripts/build.py
  - documentation/scripts/import_manual.py
  - documentation/templates/header.html
  - documentation/templates/footer.html
  - documentation/nav.yaml
  - documentation/links.yaml
  - documentation/repo.yaml
---
# Documentation system

The documentation is a static site generated from Markdown and from the code. The output uses one stylesheet and one Pygments stylesheet.

## Build

```bash
.venv/bin/python documentation/scripts/build.py
.venv/bin/python documentation/scripts/build.py --clean
```

The script finds the repository root, sets `MATFORMER_SERVER=0` before importing the registry and writes the site to `documentation/site`. The `docs` optional dependency group contains `markdown` and `Pygments`.

## Sources

```
documentation/
  src/                      Markdown pages by section
  templates/                header.html, footer.html, style.css
  nav.yaml                  sections and pages
  links.yaml                short names for the [[...]] linker
  repo.yaml                 GitHub URL and branch
  scripts/build.py          generator
  scripts/import_manual.py  migration of the original manual
  backup/                   original documents
  site/                     generated site
```

A page has optional front matter:

```yaml
---
title: Registry
sources:
  - matformer/matformer_registry.py
registry_docs: plugin.classification
category_docs: plugin
---
```

`sources` becomes the Python source row. `registry_docs` attaches the page to a registry variant. `category_docs` attaches it to a category. `readme: true` makes the homepage read `readme.md` at build time.

## Links

The custom syntax is `[[target | text]]`.

| Target | Destination |
|--------|-------------|
| `page:components/registry` | a Markdown page |
| `components/registry` | the same page, by key |
| `tutorial:add-a-task` | a page under `tutorials/` |
| `file:matformer/wrapper.py` | the source page of a file |
| `file:matformer/wrapper.py:Class.method` | the source page at a line |
| `dir:matformer/transformer_blocks` | a folder listing |
| `registry` | the registry index |
| `registry:model.classification` | a registry variant |
| `registry_category:task` | a registry category |
| `dataclass:model.default` | a configuration page |
| `configuration:task.generation` | the same page |
| `env_vars:MATFORMER_ASSETS` | an entry in the environment variable table |
| `https://...` | an external link |

A bare name is resolved through `links.yaml` and then through the page titles. An unresolved target is rendered as a broken link and reported by the build.

## Generated pages

* **Registry**: index, category pages and variant pages. A variant page shows its dataclass, implementations, priority, `requires`, constraints, metadata and source class.
* **Configuration**: index, category pages and variant pages. The pages use `registry.get_definition()` and `matformer.utils.schema`. `fields_schema` provides name, type and default; `field_docs` provides the inline comment; `field_sections` provides the section comment above a field group.
* **Files and folders**: every text file the documentation references is rendered with Pygments. The page contains an index of classes and functions, their bases and MRO, the internal imports, and a GitHub link.
* **Environment variables**: the table is generated from the comments in `matformer/utils/env_variables.py` and from every environment variable read in the code.
* **Commands**: the tables in [[tools/commands | Commands]] are generated from the server RPC registry, the `argparse` parsers, the MDAT shell and `SlurmCommands`.

## Maintenance

The registry provides the categories, variants and implementations. The dataclasses provide the fields and their documentation. The build script combines these sources, so a new field or a new registered class appears without a second list to update.

The `backup/` folder contains the original manual and the old README. The pages under `src/` preserve the hand-written text; generated pages are rebuilt on every run.

## Adding a page

1. Create `documentation/src/<section>/<name>.md` with a `title` and, when relevant, `sources`.
2. Add the page key to `nav.yaml`.
3. Add `registry_docs: category.variant` or `category_docs: category` when the page describes a registry entry.
4. Run the build and fix the warnings.

The sidebar, breadcrumbs, top navigation and source links are derived from the front matter and the page location.
