"""Sweeps: how one config file becomes many runs, and how each run stays on its own shelf.

A sweep is declared inside an ordinary YAML config, by separating the candidates of a value with
a pipe::

    training:
      lr: 2e-3 | 1e-4 | 5e-5
      accumulate_grad_batches: 1 | 2
    model:
      positional_encoding: alibi | [alibi, rope]

The pipe is invisible to the YAML parser (it is only special as the *first* character of a value),
so nothing here has to work around the format: a swept value arrives as a string, is split, and
each candidate is parsed back as YAML — which is why `true`, `2e-3` and `[alibi, rope]` keep the
type they would have had on their own line. Write `\\|` for a pipe that means itself, and quote a
candidate that starts with `[` or `{`: `"[[a], [b]] | [c]"`.

Every pipe is one dimension and the runs are their Cartesian product. The configs that were loaded
are never among them: `MatformerConfigs` keeps the candidates in `.sweeps` (dotted path -> values)
and holds the *first* candidate as its own value, so a config that nobody expands is still a valid,
runnable run. `sweep_variants()` is the expansion, and it returns these very configs when there is
nothing to expand — every caller therefore iterates unconditionally.

Each variant is a detached copy with its point applied and its artifacts renamed to carry the
point's short name: `run_name` and `checkpoint_name` gain a suffix, `predictions.jsonl` becomes
`predictions.lr-2e-3.jsonl`. Which names get rewritten is declared by the config classes
themselves (`_sweep_scoped`), so a component joins the scheme by declaring what it writes rather
than by knowing that sweeps exist. Nothing downstream has to change: the trainer, the loggers
and the stores simply read their configs, and read a different shelf each time.
"""
import copy
import fnmatch
import hashlib
import itertools
import json
import os
import re

import yaml

MARK = '|'
ESCAPED = '\\|'
NAME_LIMIT = 56
TOKEN_LIMIT = 16
KEY_LIMIT = 28
EXT_LIMIT = 8                       # .duckdb and friends still count as an extension

_SPLIT = re.compile(r'(?<!\\)\|')
_UNSAFE = re.compile(r'[^A-Za-z0-9.+_-]+')
_STEP = re.compile(r'_s_?(\d+)$')
# A field whose name says what it holds rather than what it means: named after the entry instead,
# so `checkpoint.probes.path` contributes `probes-s1500` to a point name, not `path-s1500`.
GENERIC_FIELDS = ('path', 'file', 'filename', 'dir', 'location', 'value')


def candidates(value):
    """The candidates a swept value declares, or None when the value is a plain one.

    Each candidate is parsed back as YAML, so it means exactly what it would have meant written
    alone on its line. Repeated candidates are dropped: two identical points of a sweep would only
    produce two identical runs."""
    if not isinstance(value, str) or MARK not in value:
        return None
    fields = _SPLIT.split(value)
    if len(fields) < 2:
        return None
    found, seen = [], set()
    for field in fields:
        text = field.strip().replace(ESCAPED, MARK)
        if not text:
            raise ValueError(f"'{value}': an empty candidate. Write '{ESCAPED}' for a pipe that "
                             f"means itself.")
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError:
            parsed = text
        key = json.dumps(parsed, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            found.append(parsed)
    return found


def declare(path, value, declared):
    """Register one value written over an override string, which may itself be a declaration:
    'training.lr=2e-3|1e-4' sweeps, 'training.lr=2e-3' pins. Returns the value to set now -- the
    first candidate, so the configs stay a runnable run whatever they are used for."""
    found = candidates(value)
    if found is None:
        declared.pop(path, None)
        return value
    declared[path] = found
    return found[0]


def harvest(document, prefix=''):
    """Split a raw config document into (base, candidates, touched).

    `base` is the document with every swept value replaced by its first candidate, `candidates`
    maps each swept dotted path to its values, and `touched` lists every path the document writes
    — so that a later file which re-declares a field *without* a pipe also closes that dimension."""
    found, touched = {}, set()

    def walk(node, path):
        if isinstance(node, dict):
            return {key: walk(value, f"{path}.{key}") for key, value in node.items()}
        if isinstance(node, (list, tuple)):
            return [walk(value, f"{path}.{index}") for index, value in enumerate(node)]
        swept = candidates(node)
        touched.add(path)
        if not swept:
            return node
        found[path] = swept
        return swept[0]

    return walk(document, prefix), found, touched


def token(value):
    """One candidate rendered short and readable: `on`, `2e-3`, `alibi+rope`, `s1500`."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'on' if value else 'off'
    if isinstance(value, (list, tuple)):
        return '+'.join(token(one) for one in value) or 'empty'
    if isinstance(value, dict):
        return '+'.join(f"{key}{token(one)}" for key, one in value.items()) or 'empty'
    if isinstance(value, float):
        return f"{value:g}"
    text = str(value)
    if os.sep in text or '/' in text:          # a path says nothing but its name, and its step
        text = os.path.splitext(os.path.basename(text.rstrip(os.sep + '/')))[0] or text
        step = _STEP.search(text)
        if step:
            return f"s{step.group(1)}"
    return _UNSAFE.sub('-', text).strip('-') or 'empty'


def _digest(values):
    return hashlib.sha1(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:6]


def point_name(values, full=False, forced=False):
    """The short name of one combination: `lr-2e-3_topk-on`.

    Built from each field's leaf so it stays readable, and kept under `NAME_LIMIT`: dimensions that
    do not fit are dropped in favour of the digest of the whole combination, which is what makes
    the name strictly unique even when two points share a prefix. `full` addresses the fields by
    their whole path (two swept fields can share a leaf), `forced` asks for the digest even when
    everything fit."""
    parts = []
    for path, value in values.items():
        trail = path.split('.')
        leaf = trail[-1] if trail[-1] not in GENERIC_FIELDS or len(trail) < 2 else trail[-2]
        parts.append(f"{_UNSAFE.sub('-', '.'.join(trail) if full else leaf)[:KEY_LIMIT]}"
                     f"-{token(value)[:TOKEN_LIMIT]}")
    readable = '_'.join(parts).strip('_-') or 'point'
    if not forced and len(readable) <= NAME_LIMIT:
        return readable
    digest = _digest(values)
    budget = NAME_LIMIT - len(digest) - len(f"+{len(parts)}") - 3
    kept, used = [], 0
    for part in parts:
        if used + len(part) + 1 > budget:
            break
        kept, used = kept + [part], used + len(part) + 1
    if len(kept) < len(parts):
        kept.append(f"+{len(parts) - len(kept)}")     # how many dimensions the name had to drop
    return f"{'_'.join(kept).strip('_-')}-{digest}"


class SweepPoint:
    """One combination of the sweep's candidates: what to run, and what its artifacts are called."""

    __slots__ = ('index', 'values', 'name')

    def __init__(self, index, values, name=None):
        self.index, self.values, self.name = index, values, name

    def describe(self):
        """The combination as printable text: one line, `path=value` pairs, a file shown by its
        name rather than by the shelf it sits on."""
        return '  '.join(f"{path}={os.path.basename(str(value)) if isinstance(value, str) else value}"
                         for path, value in self.values.items())

    def asdict(self):
        return {'name': self.name, 'index': self.index, 'values': dict(self.values)}

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<sweep {self.index}: {self.name}>"


def plan(sweeps):
    """Every point of the Cartesian product of `sweeps`, each with a name no other point shares.

    Readable names -- field leaf, candidate -- and the digest only when the points need it to tell
    themselves apart. A name is an address rather than a summary: the whole combination travels
    with every run (the run database row and its parameters, the checkpoint), so nothing is lost
to brevity."""
    if not sweeps:
        return []
    keys = list(sweeps)
    points = [SweepPoint(index, dict(zip(keys, combo)))
              for index, combo in enumerate(itertools.product(*(sweeps[key] for key in keys)), 1)]
    for forced in (False, True):
        for point in points:
            point.name = point_name(point.values, forced=forced)
        if len({point.name for point in points}) == len(points):
            break
    return points


def select(points, selection):
    """The points a `selection` asks for: 1-based numbers and name globs, comma separated.

    Empty means the whole sweep. A clause matching nothing raises instead: quietly running part of
    a sweep is the one mistake nobody notices in a table of results."""
    if selection is None or str(selection).strip() in ('', 'all', '*'):
        return list(points)
    if not points:
        raise ValueError(f"'{selection}' selects sweep points, but these configs declare no sweep "
                         f"(write 'value | other' as a value, or a 'checkpoint' section standing "
                         f"for several models).")
    wanted = []
    for clause in str(selection).replace(' ', ',').split(','):
        if not clause:
            continue
        hits = [point for point in points
                if (clause.isdigit() and point.index == int(clause))
                or fnmatch.fnmatch(point.name, clause)]
        if not hits:
            raise ValueError(f"'{clause}' is not a point of this sweep: "
                             f"{', '.join(point.name for point in points)}")
        wanted += [point for point in hits if point not in wanted]
    return wanted


def apply(configs, point):
    """Materialize one sweep point onto a copy of `configs`: its values applied, its artifacts
    renamed, the point recorded. Any configs will do -- the ones a file gave, the ones a checkpoint
    came back with -- which is what lets a task run share one loaded model across points that do
    not touch the model, and rebuild it across the ones that do."""
    twin = detached(configs)
    for path, value in point.values.items():
        twin.set_value(path, value, declare=False, resolve=False)
    twin.sweeps, twin.active_sweep = {}, point
    twin.resolve_references()
    return scope_artifacts(twin, point.name)


def scope(value, tag):
    """Give one artifact name the tag of its sweep point: before the extension when it has one, as
    a suffix when it has none. A name already ending in the tag passes through, so scoping the same
    configs twice -- a variant of a variant -- does nothing."""
    if value in (None, ''):
        return value
    text, suffix = os.path.splitext(str(value))
    if text == tag or text.endswith((f"_{tag}", f".{tag}")):
        return text + suffix
    if suffix and len(suffix) <= EXT_LIMIT and '/' not in suffix:
        return f"{text}.{tag}{suffix}"
    return f"{text}_{tag}"


def entries(configs):
    """(holder, key, dataclass) for every config instance: what to setattr to replace one entry."""
    for name in configs.objects:
        node = getattr(configs, name)
        if hasattr(node, 'objects'):                 # MatformerConfigs / GroupConfig
            for entry in node.objects:
                child = getattr(node, entry)
                if hasattr(child, 'objects'):
                    yield from entries(child)
                else:
                    yield node, entry, child
        else:
            yield configs, name, node


def scope_artifacts(configs, tag):
    """Rename, in place, every artifact these configs point at.

    A config class declares the fields that name something it writes (`_sweep_scoped`); this walks
    the tree and tags them, so trainers, loggers and stores keep reading their own configs
    and still land on a shelf of their own. A config that is only *read* (a replay-only activation
    store) is an input, not an artifact: it keeps the name it was given."""
    touched = False
    for holder, key, instance in entries(configs):
        fields = getattr(type(instance), '_sweep_scoped', None)
        if not fields or instance.get('readonly'):
            continue
        patch = {}
        for field in fields:
            value = instance.get(field)
            if isinstance(value, str) and value:
                patch[field] = scope(value, tag)
        if patch:
            setattr(holder, key, instance.update(patch))
            touched = True
    if touched:
        configs.resolve_references()
    return configs


def detached(configs):
    """A copy of `configs` sharing nothing with them: the same tree, re-imported from its own dict
    form with the declarations already resolved, so no instantiated object and no pending dimension
    rides along into a variant."""
    twin = type(configs)()
    twin.import_config(copy.deepcopy(configs.asdict(include_defaults=False)), sweeps_on=False)
    return twin


# The sections a built model depends on: sweeping any of them means each point needs its own
# model, while a sweep of anything else can share one and rebuild only the run around it.
MODEL_SECTIONS = ('model', 'tokenizer', 'checkpoint', 'weights', 'plugin',
                  'activation_store', 'virtual_model')


def crosses_model(configs):
    """Whether what `configs` sweeps touches the sections a model is built from."""
    sections = {path.split('.')[0] for path in configs.sweeps} | {path.split('.')[0]
                                                                  for path in configs.model_axes()}
    return bool(sections & set(MODEL_SECTIONS))


def summarize(points, out=print):
    """The plan as a terminal shows it: the dimensions, then one line per run."""
    if not points:
        out("No sweep declared: these configs are one run.")
        return
    out(f"Sweep: {len(points)} runs over {', '.join(points[0].values)}")
    for point in points:
        out(f"  [{point.index}] {point.name}")
        out(f"      {point.describe()}")
