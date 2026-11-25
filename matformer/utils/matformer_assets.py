"""MATFORMER_ASSETS: one folder, outside the repo, holding everything a run touches.

A tiny YAML in the repo root (``configuration.yaml``) points the checkout at an absolute folder;
the ``MATFORMER_ASSETS`` environment variable, if set, wins over it:

    # configuration.yaml
    assets: /home/matteo/matformer_assets

Under that root every area has its fixed place, created on demand by ``bootstrap()`` (called once,
at ``matformer_registry`` import time) if the folder is configured but absent:

    <assets>/configs         shared configs        <assets>/logs          run logs
    <assets>/checkpoints     weights               <assets>/dataset       datasets
    <assets>/venvs           virtualenvs           <assets>/experiments   experiment outputs
    <assets>/extra_plugins   modules the registry also scans for @register / @register_dataclass

Configs then address those places with magic tokens instead of machine-specific absolute paths.
``MatformerConfigs`` expands them at load time, in every string of the document (``includes``
included), and CLI/GUI/server paths go through the same expansion:

    includes:    ["$CONFIG_ASSETS/classification/shared/base_sequence_classification.yaml"]
    data_root:   "$DATA_ASSETS/IRONITA/train.jsonl"
    save_dir:    "$CHECKPOINT_ASSETS/sequence_classification/ironita"

``$ASSETS`` is the root itself; ``$<AREA>_ASSETS`` is one area folder. Moving or syncing a machine
is then a matter of copying the assets folder and editing one line, with no config surgery.
From a shell the token must be quoted ('--cfg "$CONFIG_ASSETS/..."'), or the expansion happens in
the wrong process. ``python -m matformer.utils.matformer_assets`` declares the folder and builds it.

Everything here is optional: with no configuration.yaml and no environment variable the tokens do
not exist, the registry scans nothing, and the framework behaves exactly as before. A config that
*uses* a token while no root is configured raises, rather than quietly opening '$DATA_ASSETS/...'.
"""
import os
import re
from pathlib import Path

CONFIG_FILENAME = "configuration.yaml"

# Area folders under the assets root: area name -> folder name (they can be remapped, ex. to
# 'datasets', without touching anything else). Scanned/created in this order.
AREA_FOLDER = {"configs": "configs", "checkpoints": "checkpoints", "venvs": "venvs",
               "logs": "logs", "dataset": "dataset", "experiments": "experiments",
               "extra_plugins": "extra_plugins"}
AREAS = tuple(AREA_FOLDER)

# Magic token -> area (None means the assets root itself).
TOKENS = {
    "ASSETS": None,
    "CONFIG_ASSETS": "configs", "CONFIGS_ASSETS": "configs",
    "CHECKPOINT_ASSETS": "checkpoints", "CHECKPOINTS_ASSETS": "checkpoints",
    "VENV_ASSETS": "venvs", "VENVS_ASSETS": "venvs",
    "LOG_ASSETS": "logs", "LOGS_ASSETS": "logs",
    "DATA_ASSETS": "dataset", "DATASET_ASSETS": "dataset", "DATASETS_ASSETS": "dataset",
    "EXPERIMENT_ASSETS": "experiments", "EXPERIMENTS_ASSETS": "experiments",
    "PLUGIN_ASSETS": "extra_plugins", "PLUGINS_ASSETS": "extra_plugins",
    "EXTRA_PLUGIN_ASSETS": "extra_plugins", "EXTRA_PLUGINS_ASSETS": "extra_plugins",
}

_TOKEN_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")

# The spelling shrink() writes back: one token per area, deepest folder wins.
CANONICAL = (("$CONFIG_ASSETS", "configs"), ("$CHECKPOINT_ASSETS", "checkpoints"),
             ("$VENV_ASSETS", "venvs"), ("$LOG_ASSETS", "logs"), ("$DATA_ASSETS", "dataset"),
             ("$EXPERIMENT_ASSETS", "experiments"), ("$PLUGIN_ASSETS", "extra_plugins"),
             ("$ASSETS", None))

_ROOT_KEYS = ("assets", "matformer_assets", "MATFORMER_ASSETS", "root")

_cache = {}


def repo_root() -> Path:
    """Root of the checkout this matformer package lives in."""
    return Path(__file__).resolve().parents[2]


def config_file() -> Path:
    """The tiny YAML at the repo root declaring the assets folder of this checkout."""
    return repo_root() / CONFIG_FILENAME


def configured_root() -> Path | None:
    """Assets root as declared by the environment or configuration.yaml, None if not configured.
    Nothing is created and nothing is required to exist here: this is only what was asked for."""
    env = os.environ.get("MATFORMER_ASSETS", "").strip()
    if env:
        return Path(os.path.expandvars(env)).expanduser()
    file = config_file()
    if not file.is_file():
        return None
    try:
        import yaml
        declared = yaml.safe_load(file.read_text("utf-8")) or {}
    except Exception as problem:
        print(f"WARNING: {file} cannot be read ({type(problem).__name__}: {problem}); "
              f"MATFORMER_ASSETS is disabled.")
        return None
    if isinstance(declared, str):
        return Path(os.path.expandvars(declared)).expanduser()
    if not isinstance(declared, dict):
        print(f"WARNING: {file} must hold a mapping (assets: /absolute/path); MATFORMER_ASSETS is disabled.")
        return None
    for key in _ROOT_KEYS:
        value = declared.get(key)
        if isinstance(value, str) and value.strip():
            return Path(os.path.expandvars(value)).expanduser()
    print(f"WARNING: {file} declares no assets path (expected '{_ROOT_KEYS[0]}: /absolute/path').")
    return None


def assets_root(create: bool = False) -> Path | None:
    """The configured assets folder, or None when MATFORMER_ASSETS is not in use.
    With create=True the folder and its area subfolders are made, if missing."""
    if "root" not in _cache:
        _cache["root"] = configured_root()
    root = _cache["root"]
    if root is None:
        return None
    if create:
        ensure_layout(root)
    return root


def area(name: str) -> Path | None:
    """One area folder (ex. 'dataset'), or None if MATFORMER_ASSETS is not configured.
    Does not require the folder to exist."""
    root = assets_root()
    if root is None:
        return None
    if name not in AREAS:
        raise ValueError(f"Unknown assets area '{name}'. Available: {', '.join(AREAS)}")
    return root / AREA_FOLDER[name]


def path(*parts: str) -> Path:
    """Join onto the assets root (ex. path('dataset', 'IRONITA')); raises if not configured."""
    root = assets_root()
    if root is None:
        raise ValueError("MATFORMER_ASSETS is not configured: add "
                         f"'{_ROOT_KEYS[0]}: /absolute/path' to {config_file()} "
                         "or export the MATFORMER_ASSETS environment variable.")
    return root.joinpath(*parts)


def enabled() -> bool:
    return assets_root() is not None


def expand(value):
    """Replace magic assets tokens in a string ($ASSETS, $DATA_ASSETS, ...). Anything else
    (numbers, Path, containers) is returned untouched; see expand_tree for nested documents."""
    if not isinstance(value, str) or "$" not in value:
        return value

    def substitute(match):
        token = match.group(1).upper()
        if token in TOKENS:
            target = TOKENS[token]
            root = assets_root(create=False)
            if root is None:
                raise ValueError(f"'${match.group(1)}' needs an assets folder: add "
                                 f"'{_ROOT_KEYS[0]}: /absolute/path' to {config_file()} "
                                 "or export MATFORMER_ASSETS.")
            return str(root if target is None else root / AREA_FOLDER[target])
        if token.endswith("ASSETS"):
            raise ValueError(f"Unknown assets token '${match.group(1)}'. "
                             f"Available: {', '.join(sorted(TOKENS))}")
        return match.group(0)                 # not ours ($HOME, $id, ...): leave it alone

    return _TOKEN_RE.sub(substitute, value)


def shrink(value):
    """Inverse of expand, for anything writing a config back: a path that lives inside the assets
    folder goes back to its token, so a file edited by the GUI stays machine-independent."""
    if not isinstance(value, str) or not value:
        return value
    root = assets_root()
    if root is None:
        return value
    prefixes = []
    for token, area_name in CANONICAL:
        for base in (root, root.resolve()):
            folder = base if area_name is None else base / AREA_FOLDER[area_name]
            prefixes.append((str(folder), token))
    for folder, token in sorted(prefixes, key=lambda pair: -len(pair[0])):   # deepest first
        if value == folder or value.startswith(folder + os.sep):
            return token + value[len(folder):]
    return value


def expand_tree(node):
    """Recursively expand every string of a parsed config document (dicts, lists, tuples)."""
    if isinstance(node, str):
        return expand(node)
    if isinstance(node, dict):
        return {key: expand_tree(value) for key, value in node.items()}
    if isinstance(node, list):
        return [expand_tree(item) for item in node]
    if isinstance(node, tuple):
        return tuple(expand_tree(item) for item in node)
    return node


def ensure_layout(root: Path | None = None) -> list:
    """Create the assets root and its empty area folders; returns the folders created."""
    root = Path(root) if root is not None else assets_root()
    if root is None:
        return []
    created = []
    for folder in [root, *(root / sub for sub in AREA_FOLDER.values())]:
        if not folder.is_dir():
            folder.mkdir(parents=True, exist_ok=True)
            created.append(folder)
    return created


def bootstrap() -> Path | None:
    """Called once at registry import: resolve the assets folder, build the tree if configured
    and missing. Never raises, never prints on the happy path."""
    try:
        root = assets_root()
        if root is None:
            return None
        created = ensure_layout(root)
        if created:
            print(f"MATFORMER_ASSETS: created {len(created)} folder(s) under {root}")
        return root
    except Exception as problem:
        print(f"WARNING: MATFORMER_ASSETS bootstrap failed "
              f"({type(problem).__name__}: {problem}); continuing without it.")
        return None


def write_config(root: str | Path) -> Path:
    """Declare `root` as the assets folder of this checkout (configuration.yaml), creating the tree."""
    folder = Path(root).expanduser()
    if not folder.is_absolute():
        raise ValueError(f"The assets folder must be an absolute path, got '{folder}'.")
    file = config_file()
    file.write_text(
        "# Where this checkout keeps datasets, checkpoints, configs, logs and venvs.\n"
        "# Every config may address it with $ASSETS and $<AREA>_ASSETS.\n"
        "# Override it for a single process with the MATFORMER_ASSETS environment variable.\n"
        f"{_ROOT_KEYS[0]}: {folder}\n", encoding="utf-8")
    _cache.clear()
    ensure_layout(assets_root())
    return file


def reload():
    """Forget the cached root: next access re-reads environment and configuration.yaml."""
    _cache.clear()


def main(argv=None):
    """CLI: `python -m matformer.utils.matformer_assets [/abs/assets/folder]`. With a path, declares
    it in configuration.yaml and builds the folder tree; without, reports the current one."""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m matformer.utils.matformer_assets",
                                     description="Point this checkout at a MATFORMER_ASSETS folder.")
    parser.add_argument("root", nargs="?", help="absolute path of the assets folder")
    args = parser.parse_args(argv)
    if args.root:
        write_config(args.root)
    root = assets_root(create=True)
    if root is None:
        print(f"No assets folder configured. Give one: python -m matformer.utils.matformer_assets "
              f"/abs/path (writes {CONFIG_FILENAME}).")
        return 1
    print(f"MATFORMER_ASSETS: {root}")
    for name in AREAS:
        mark = "" if (root / AREA_FOLDER[name]).is_dir() else " MISSING"
        print(f"  {name:<14} {root / AREA_FOLDER[name]}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
