# Matformer documentation

The documentation is a static site generated from Markdown and from the code.

## Build

```bash
.venv/bin/python documentation/scripts/build.py
.venv/bin/python documentation/scripts/build.py --clean
```

`markdown` and `Pygments` are the only extra dependencies; they are declared in the `docs` optional dependency group:

```bash
uv pip install -e '.[docs]'
```

The output is written to `documentation/site`. Open `documentation/site/index.html` directly, or serve the folder with any static file server.

## Layout

```
documentation/
  src/                  Markdown sources, one folder per section
  templates/            header.html, footer.html, style.css
  nav.yaml              the sections and the pages they contain
  links.yaml            short names accepted by the [[...]] linker
  repo.yaml             GitHub URL and branch for source links
  scripts/build.py      the generator
  scripts/import_manual.py
                        the one-off migration of the original manual
  backup/               the original documents, untouched
  site/                 the generated site
  ../DOCUMENTATION_HANDOFF.md  what is done, what is missing
```

## Writing a page

A page is a Markdown file with optional front matter:

```yaml
---
title: The registry
sources:
  - matformer/matformer_registry.py
category_docs: [model]
registry_docs: [plugin.classification]
---
```

* `sources` becomes the "Source" row at the top of the page.
* `category_docs` attaches the page to a registry category.
* `registry_docs` attaches the page to a registry variant.
* `readme: true` makes the homepage the repository README, rewritten on every build.

Add the page key to `nav.yaml` to put it in the sidebar and the breadcrumbs. No other file needs to change.

## Links

The custom syntax is `[[target | text]]`. The target is one of:

| Target | Destination |
|--------|-------------|
| `page:components/registry` | a Markdown page |
| `components/registry` | the same page, by key |
| `tutorial:add-a-task` | a tutorial |
| `file:matformer/wrapper.py` | the rendered source |
| `file:matformer/wrapper.py:load_model` | the source at the symbol's line |
| `dir:matformer/transformer_blocks` | a folder listing |
| `registry` / `registry:model.classification` | the registry reference |
| `registry_category:task` | a registry category |
| `dataclass:model.default` | a configuration page |
| `configuration:task.generation` | the same page |
| `env_vars:MATFORMER_ASSETS` | an environment variable entry |
| `https://...` | an external link |

A bare name is resolved through `links.yaml`, then through the page titles. A target that cannot be resolved is rendered as a visibly broken link and reported by the build. The build also checks every `href` and every anchor after writing the site.

## Generated content

* The registry pages come from `registry.get_registry_info()`.
* The configuration pages come from `registry.get_definition(category, variant)` and `matformer.utils.schema` (`fields_schema`, `field_docs`, `field_sections`).
* The environment variable tables come from the `# NAME: description` lines in `matformer/utils/env_variables.py` and from every `os.environ.get(...)` in the code.
* The file and folder pages come from the repository itself, with Pygments highlighting, an index of classes and functions, class bases resolved through the MRO and internal import links.
* The command reference comes from the RPC registry, the `argparse` parsers, the MDAT shell commands and the SLURM commands.
* The repository explorer lists the tracked folders; `repo.yaml` sets the GitHub URL and branch used by every source page.

The generated parts are deliberately not maintained by hand. If a field, a variant or a variable is missing, the fix belongs in the code (a comment, a decorator, a registration), not in a separate documentation file that would drift.

## The original manual

`backup/documentazione.txt`, `backup/todo_documentazione.txt`, `backup/readme.md` and the files from the old `docs/` folder, now in `backup/legacy-docs/`, are the author's originals. They are kept for reference; the pages under `src/` preserve the hand-written prose. `scripts/import_manual.py` is the migration script and can be re-run with `--force` only if a page needs to be restored from the original.
