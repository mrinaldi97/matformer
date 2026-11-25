# matformer/utils/schema.py
"""Config introspection: the bridge between a registered dataclass and a rendered form.

    field_docs(cls)      -> {field: the # comment on its annotation}, read from the source
    fields_schema(cls)   -> [{name, label, kind, default, doc, options?}] in the GUI field language
    to_yamlable(v)       -> a dataclass-friendly plain value for JSON transport

Used by the task console (docs), the server (`task.fields`, `config.schema`) and any future
editor: one definition of "what a parameter is, what it means, which values it accepts".
"""
import ast
import dataclasses
import inspect
import re
import textwrap

_FIELD_DOC = re.compile(r'^\s+([A-Za-z_]\w*)\s*:[^#\n]*(?:#\s*(.*\S))?\s*$', re.MULTILINE)
_LITERAL = re.compile(r"Literal\[(.+?)\]")


def field_docs(config_class) -> dict:
    """'# comment' after each annotated field, across the whole MRO."""
    docs = {}
    for klass in reversed(inspect.getmro(config_class)):
        try:
            source = inspect.getsource(klass)
        except (OSError, TypeError):
            continue
        for name, doc in _FIELD_DOC.findall(source):
            if doc:
                docs[name] = doc
    return docs


def field_sections(config_class) -> dict:
    """{field: the standalone '# Section' comment above it}, across the whole MRO.

    Comments that follow a field on the same line are field documentation (field_docs);
    a comment on a line of its own labels the group of fields that starts below it. The two
    together are everything the configuration reference needs to render a dataclass.
    """
    sections = {}
    for klass in reversed(inspect.getmro(config_class)):
        try:
            source = textwrap.dedent(inspect.getsource(klass))
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            continue
        node = next((child for child in tree.body if isinstance(child, ast.ClassDef)), None)
        if node is None:
            continue
        lines = source.splitlines()
        body_indent = node.body[0].col_offset if node.body else 4
        current = None
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                annotation = ast.unparse(statement.annotation) if statement.annotation else ""
                if statement.target.id.startswith("_") or annotation.startswith("ClassVar"):
                    continue
                comments = []
                index = statement.lineno - 2
                while index >= 0:
                    stripped = lines[index].strip()
                    indent = len(lines[index]) - len(lines[index].lstrip())
                    if not stripped and not comments:
                        index -= 1
                        continue
                    if stripped.startswith("#") and indent == body_indent:
                        comments.append(stripped.lstrip("#").strip())
                        index -= 1
                        continue
                    break
                if len(comments) == 1:
                    current = comments[0]
                elif len(comments) > 1:
                    current = None
                if current:
                    sections[statement.target.id] = current
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                current = None
    return sections


PATH_HINTS = ("path", "dir", "file", "root", "source", "model", "checkpoint", "output",
              "input", "tokenizer", "filename")


def is_path_field(field) -> bool:
    """Heuristic: a str-ish field whose name smells like a filesystem location. It earns the
    '...' browse button everywhere forms are rendered (config editor, task parameters)."""
    t = str(field.type)
    if "str" not in t or any(k in t for k in ("list", "dict", "tuple")):
        return False
    name = field.name.lower()
    return any(h in name for h in PATH_HINTS) and not name.endswith("class")


def kind_of(field) -> str:
    """Map a dataclass annotation (string, thanks to PEP 563) to a GUI field kind."""
    t = str(field.type)
    if "Literal" in t:
        return "choice"
    if "bool" in t:
        return "bool"
    if "float" in t:
        return "float"
    if "int" in t:
        return "int"
    if is_path_field(field):
        return "path"
    if t in ("str", "Optional[str]") or ("str" in t and not any(k in t for k in ("list", "dict", "tuple"))):
        return "text"
    return "json"                                   # lists, dicts, tuples, unions: JSON text


def literal_options(field) -> list:
    match = _LITERAL.search(str(field.type))
    return [o.strip().strip("'\"") for o in match.group(1).split(",")] if match else []


def default_of(field):
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:                    # type: ignore[misc]
        try:
            return field.default_factory()                                  # type: ignore[misc]
        except Exception:
            return None
    return None


def fields_schema(config_class, skip=(), base_only=False) -> list:
    """A GUI-ready description of a config dataclass: kinds, defaults, docs, literal options.
    Unknown/unsettable fields (private, references) are skipped by the caller's `skip`."""
    docs = field_docs(config_class)
    out = []
    for f in dataclasses.fields(config_class):
        if f.name in skip or f.name.startswith("_"):
            continue
        kind = kind_of(f)
        default = default_of(f)
        raw_type = f.type if isinstance(f.type, str) else str(f.type)
        if raw_type.startswith("<class ") and not isinstance(f.type, str):
            raw_type = getattr(f.type, "__name__", raw_type)
        annotation = raw_type.removeprefix("typing.")
        entry = {"name": f.name, "label": f.name.replace("_", " ").capitalize(),
                 "type": annotation, "kind": kind, "doc": docs.get(f.name, "")}
        if kind == "choice":
            entry["options"] = literal_options(f)
        if isinstance(default, (tuple, set, list, dict)) and kind != "choice":
            kind = entry["kind"] = "json"
        entry["default"] = list(default) if isinstance(default, tuple) else default
        out.append(entry)
    return out
