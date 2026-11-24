#!/usr/bin/env python3
"""Build the Matformer documentation.

The sources are Markdown files in documentation/src, plus pages generated from the registry,
the configuration dataclasses, the environment variables and the Python files themselves.

Usage:
    python documentation/scripts/build.py [--clean]

Everything is written to documentation/site. The repository root is discovered from this file,
so the script can be run from anywhere.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import html
import importlib
import inspect
import io
import json
import os
import re
import shutil
import sys
import textwrap
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import markdown
import yaml
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_for_filename

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = REPO_ROOT / "documentation"
SRC = DOC_ROOT / "src"
SITE = DOC_ROOT / "site"
TEMPLATES = DOC_ROOT / "templates"
NAV_FILE = DOC_ROOT / "nav.yaml"
LINKS_FILE = DOC_ROOT / "links.yaml"
REPO_FILE = DOC_ROOT / "repo.yaml"

IGNORED_DIRS = {"gui", "server", "remote", "__pycache__", ".git", ".venv"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".json", ".cfg", ".ini",
                 ".sh", ".html", ".css", ".js", ".jinja", ".j2"}

CATEGORY_ORDER = [
    "model", "layer", "sequence_mixer", "attention", "mlp", "norm", "norm_style",
    "embedding", "linear", "dropout", "positional_encoding", "loss", "loss_fn",
    "learning_signal", "learning_signal_selector", "stream", "plugin", "hooks",
    "tokenizer", "task", "data", "logger", "checkpoint", "checkpoint_holder",
    "weights", "training", "validation", "masking", "virtual_model", "activation_store",
]

MD_EXTENSIONS = [
    "markdown.extensions.toc", "markdown.extensions.tables",
    "markdown.extensions.fenced_code", "markdown.extensions.codehilite",
    "markdown.extensions.sane_lists", "markdown.extensions.attr_list",
    "markdown.extensions.def_list", "markdown.extensions.md_in_html",
]
MD_EXTENSION_CONFIGS = {
    "markdown.extensions.toc": {"permalink": ""},
    "markdown.extensions.codehilite": {"guess_lang": False, "css_class": "highlight"},
}


@dataclass
class SourcePage:
    source: Path
    out: Path
    key: str
    meta: dict
    body: str
    title: str


@dataclass
class RenderedPage:
    out: Path
    title: str
    body: str
    sources: list[str]
    section: Optional[str] = None
    sidebar: Optional[str] = None
    breadcrumbs: Optional[str] = None
    body_class: str = ""


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return text or "page"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def rel_url(from_out: Path, to_out: Path) -> str:
    start = (SITE / from_out).parent
    target = SITE / to_out
    relative = os.path.relpath(target, start).replace(os.sep, "/")
    return urllib.parse.quote(relative, safe="/#~:@!$&'()*+,;=-._")


def is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class Documentation:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.pages: dict[str, SourcePage] = {}
        self.variant_docs: dict[str, str] = {}
        self.category_docs: dict[str, str] = {}
        self.alias: dict[str, str] = {}
        self.registry_info: dict[str, Any] = {}
        self.registry_categories: list[str] = []
        self.symbols: dict[str, dict[str, int]] = {}
        self.class_info: dict[str, dict[str, tuple[int, list[str]]]] = {}
        self.file_imports: dict[str, list[dict]] = {}
        self.import_map: dict[str, dict[str, tuple[str, Optional[str]]]] = {}
        self.class_locations: dict[str, list[tuple[str, int, list[str]]]] = {}
        self.repo: dict = {}
        self.file_pages: dict[str, Path] = {}
        self.dir_pages: dict[str, Path] = {}
        self.generated: list[RenderedPage] = []
        self.nav: list[dict] = []

    # ------------------------------------------------------------------ loading

    def load_registry(self) -> None:
        os.environ.setdefault("MATFORMER_SERVER", "0")
        sys.path.insert(0, str(REPO_ROOT))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            from matformer.matformer_registry import registry
            self.registry_info = registry.get_registry_info(relative_filepath=True)
            self.registry_info.pop("hooks", None)
            self.registry = registry
        self.registry_categories = sorted(
            self.registry_info,
            key=lambda name: (CATEGORY_ORDER.index(name) if name in CATEGORY_ORDER else 999, name),
        )
        for line in buffer.getvalue().splitlines():
            if line.strip() and "not loaded" not in line:
                self.warnings.append(f"registry: {line.strip()}")

    def load_repo(self) -> None:
        self.repo = yaml.safe_load(read_text(REPO_FILE)) if REPO_FILE.exists() else {}
        if not isinstance(self.repo, dict):
            self.repo = {}

    def load_commands(self) -> None:
        self.rpc_commands: list[dict] = []
        self.cli_commands: list[dict] = []
        self.mdat_commands: list[dict] = []
        self.slurm_commands: list[dict] = []
        try:
            from matformer.server import core
            for module in ("config_cmd", "fs", "jobs", "mdat_host", "memory", "runs",
                           "slurm_cmd", "sysinfo", "tasks_host"):
                importlib.import_module(f"matformer.server.{module}")
            for name, handler in sorted(core.RPC.items()):
                source = inspect.getsourcefile(handler)
                line = inspect.getsourcelines(handler)[1] if source else None
                self.rpc_commands.append({
                    "name": name,
                    "doc": (handler.__doc__ or "").strip().splitlines()[0] if handler.__doc__ else "",
                    "file": repo_relative(Path(source)) if source else None,
                    "line": line,
                })
        except Exception as problem:
            self.warnings.append(f"command reference: server RPC introspection failed: {problem}")
        for module, label in (("run_task", "run_task"), ("matformer.training.launch", "train")):
            try:
                parser = getattr(importlib.import_module(module), "parser")()
                options = []
                for action in parser._actions:
                    if action.dest in ("help", "version"):
                        continue
                    options.append({
                        "flags": ", ".join(action.option_strings) or action.dest,
                        "help": action.help or "",
                    })
                self.cli_commands.append({"name": label, "usage": parser.format_usage().strip(),
                                           "options": options})
            except Exception as problem:
                self.warnings.append(f"command reference: {label} CLI introspection failed: {problem}")
        try:
            from matformer.data.mdat.tui import MDatShell
            for name in sorted(dir(MDatShell)):
                if not name.startswith("do_"):
                    continue
                method = getattr(MDatShell, name)
                doc = (method.__doc__ or "").strip().splitlines()
                self.mdat_commands.append({"name": name[3:], "usage": doc[0] if doc else "",
                                           "doc": " ".join(doc[1:]).strip()})
        except Exception as problem:
            self.warnings.append(f"command reference: MDAT shell introspection failed: {problem}")
        try:
            from matformer.slurm import SlurmCommands
            for name, method in inspect.getmembers(SlurmCommands, inspect.isfunction):
                if name.startswith("_"):
                    continue
                doc = (method.__doc__ or "").strip().splitlines()
                self.slurm_commands.append({"name": name, "doc": doc[0] if doc else ""})
        except Exception as problem:
            self.warnings.append(f"command reference: SLURM introspection failed: {problem}")

    def load_aliases(self) -> None:
        if LINKS_FILE.exists():
            self.alias = yaml.safe_load(read_text(LINKS_FILE)) or {}
        self.alias = {str(k): str(v) for k, v in self.alias.items()}

    def load_nav(self) -> None:
        raw = yaml.safe_load(read_text(NAV_FILE)) if NAV_FILE.exists() else None
        if raw is None:
            self.nav = []
            return
        for section in raw.get("sections", []):
            title = section["title"]
            pages = []
            for entry in section.get("pages", []):
                if isinstance(entry, str):
                    pages.append({"key": entry, "title": None})
                else:
                    pages.append({"key": entry["page"], "title": entry.get("title")})
            self.nav.append({"title": title, "pages": pages})

    def split_front_matter(self, text: str) -> tuple[dict, str]:
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end != -1:
                meta = yaml.safe_load(text[4:end]) or {}
                return meta, text[end + 5:]
        return {}, text

    def load_sources(self) -> None:
        for path in sorted(SRC.rglob("*.md")):
            relative = path.relative_to(SRC).with_suffix("")
            key = relative.as_posix()
            meta, body = self.split_front_matter(read_text(path))
            title = meta.get("title")
            if title is None:
                match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
                title = match.group(1).strip() if match else key.split("/")[-1].replace("-", " ").title()
            out = Path("index.html") if key == "index" else Path(f"{key}.html")
            self.pages[key] = SourcePage(path, out, key, meta, body, title)

    def map_variant_docs(self) -> None:
        for page in self.pages.values():
            registry_docs = page.meta.get("registry_docs") or []
            if isinstance(registry_docs, str):
                registry_docs = [registry_docs]
            for target in registry_docs:
                self.variant_docs[target] = page.key
            category_docs = page.meta.get("category_docs") or []
            if isinstance(category_docs, str):
                category_docs = [category_docs]
            for target in category_docs:
                self.category_docs[target] = page.key

    # ------------------------------------------------------------------ inventory

    def collect_linked_paths(self) -> tuple[set[str], set[str]]:
        files, dirs = set(), set()
        pattern = re.compile(r"\[\[([^\]]+)\]\]")
        for page in list(self.pages.values()) + [self.readme_page()]:
            for target in pattern.findall(page.body):
                target = target.split("|", 1)[0].strip()
                if target.startswith(("file:", "file://", "py_file:")):
                    value = re.sub(r"^(file://|file:|py_file:)", "", target).lstrip("/")
                    path = value.split(":", 1)[0]
                    files.add(path)
                elif target.startswith("dir:"):
                    dirs.add(target[4:].strip("/"))
        return files, dirs

    def candidates(self) -> list[Path]:
        roots = [REPO_ROOT]
        found: list[Path] = []
        for path in (REPO_ROOT / "matformer").rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES and not is_ignored(path.relative_to(REPO_ROOT)):
                found.append(path)
        for name in ("configuration.yaml", "pyproject.toml", "train_model.py", "run_task.py"):
            path = REPO_ROOT / name
            if path.is_file():
                found.append(path)
        for folder in ("configs", "scripts", "tests"):
            for path in (REPO_ROOT / folder).rglob("*"):
                if path.is_file() and path.suffix in TEXT_SUFFIXES and not is_ignored(path.relative_to(REPO_ROOT)):
                    found.append(path)
        for path in roots:
            for name in ("readme.md",):
                if (path / name).is_file():
                    found.append(path / name)
        return sorted(set(found))

    def build_inventory(self) -> None:
        linked_files, linked_dirs = self.collect_linked_paths()
        for page in self.pages.values():
            sources = page.meta.get("sources") or []
            if isinstance(sources, str):
                sources = [sources]
            linked_files.update(str(source) for source in sources)
        files = {repo_relative(path) for path in self.candidates()}
        files.update(linked_files)
        for relative in sorted(files):
            path = REPO_ROOT / relative
            if not path.is_file():
                continue
            self.file_pages[relative] = Path("file") / (relative + ".html")
        directories = {"."}
        for relative in list(self.file_pages):
            parts = Path(relative).parts[:-1]
            for i in range(len(parts) + 1):
                directories.add(Path(*parts[:i]).as_posix() if i else ".")
        for relative in linked_dirs:
            path = REPO_ROOT / relative
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_dir() and not is_ignored(child.relative_to(REPO_ROOT)):
                        directories.add(repo_relative(child))
        hidden = {"assets", "experiments", "checkpoints_NEW", "interfacciaclaude",
                  "matformer.egg-info", "wandb", ".aid", ".crush", ".ipynb_checkpoints",
                  ".matformer_server", ".pytest_cache", ".git", ".venv"}
        for child in REPO_ROOT.rglob("*"):
            if not child.is_dir():
                continue
            relative_child = child.relative_to(REPO_ROOT)
            if any(part in hidden or part.startswith(".") for part in relative_child.parts):
                continue
            if is_ignored(relative_child) or relative_child.parts[:2] == ("documentation", "site"):
                continue
            directories.add(repo_relative(child))
        for relative in sorted(directories):
            if relative == ".":
                self.dir_pages[relative] = Path("file/index.html")
            else:
                self.dir_pages[relative] = Path("file") / relative / "index.html"
        for relative in list(self.file_pages):
            symbols, classes, imports = self.index_file(REPO_ROOT / relative)
            self.symbols[relative] = symbols
            self.class_info[relative] = classes
            self.file_imports[relative] = imports
            for name, (line, bases) in classes.items():
                self.class_locations.setdefault(name.split(".")[-1], []).append((relative, line, bases))
        for relative, imports in self.file_imports.items():
            resolved = {}
            for item in imports:
                target = self.resolve_module(item["module"])
                if target is None:
                    continue
                for name in item["names"]:
                    resolved[name] = (target, name if name != "*" else None)
            self.import_map[relative] = resolved

    def index_file(self, path: Path) -> tuple[dict[str, int], dict[str, tuple[int, list[str]]], list[dict]]:
        if path.suffix != ".py":
            return {}, {}, []
        try:
            tree = ast.parse(read_text(path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            return {}, {}, []
        symbols: dict[str, int] = {}
        classes: dict[str, tuple[int, list[str]]] = {}
        imports: list[dict] = []

        def walk(node: ast.AST, prefix: str = "") -> None:
            for child in getattr(node, "body", []):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = f"{prefix}.{child.name}" if prefix else child.name
                    symbols[name] = child.lineno
                    if isinstance(child, ast.ClassDef):
                        bases = [ast.unparse(base).split(".")[-1] for base in child.bases]
                        classes[name] = (child.lineno, bases)
                        walk(child, name)
        walk(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("matformer"):
                imports.append({"module": node.module, "names": [alias.name for alias in node.names],
                                "line": node.lineno})
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("matformer"):
                        imports.append({"module": alias.name, "names": [alias.asname or alias.name.rsplit(".", 1)[-1]],
                                        "line": node.lineno})
        return symbols, classes, imports

    def resolve_module(self, module: str) -> Optional[str]:
        base = module.replace(".", "/")
        for candidate in (base + ".py", base + "/__init__.py"):
            if candidate in self.file_pages:
                return candidate
        return None

    def class_location(self, name: str, current: str) -> Optional[tuple[str, int, list[str]]]:
        local = self.class_info.get(current, {}).get(name)
        if local is not None:
            return current, local[0], local[1]
        imported = self.import_map.get(current, {}).get(name)
        if imported:
            target, symbol = imported
            if symbol and symbol in self.class_info.get(target, {}):
                line, bases = self.class_info[target][symbol]
                return target, line, bases
            for qualified, (line, bases) in self.class_info.get(target, {}).items():
                if qualified.split(".")[-1] == name:
                    return target, line, bases
        candidates = self.class_locations.get(name, [])
        if len(candidates) == 1:
            return candidates[0]
        return None

    def mro_chain(self, name: str, current: str, seen: Optional[set] = None) -> list[tuple[str, str, int]]:
        seen = seen or set()
        location = self.class_location(name, current)
        if location is None:
            return []
        file, line, bases = location
        key = (file, name)
        if key in seen:
            return []
        seen.add(key)
        chain = [(file, name, line)]
        for base in bases:
            chain.extend(self.mro_chain(base, file, seen))
        return chain

    # ------------------------------------------------------------------ links

    def page_out(self, key: str) -> Optional[Path]:
        page = self.pages.get(key)
        return page.out if page else None

    def resolve_target(self, target: str, from_out: Path) -> tuple[Optional[str], Optional[str]]:
        target = target.strip()
        if not target:
            return None, None
        if target.startswith(("http://", "https://", "mailto:")):
            return target, None
        if target in self.alias:
            return self.resolve_target(self.alias[target], from_out)
        for prefix, scheme in (("registry.", "registry:"), ("dataclass.", "dataclass:"),
                               ("configuration.", "configuration:")):
            if target.startswith(prefix):
                target = scheme + target[len(prefix):]
                break
        anchor = None
        if "#" in target and not target.startswith(("file:", "py_file:")):
            target, anchor = target.split("#", 1)
        if target.startswith(("http://", "https://", "mailto:")):
            return target, anchor

        if target.startswith("page:"):
            key = target[5:].strip("/")
            out = self.page_out(key)
            if out:
                return rel_url(from_out, out), anchor
            return None, None
        if target.startswith(("tutorial:", "tutorials:")):
            name = target.split(":", 1)[1].strip("/").replace("_", "-")
            key = f"tutorials/{name}"
            out = self.page_out(key)
            if out:
                return rel_url(from_out, out), anchor
            return None, None
        if target.startswith(("file://", "file:", "py_file:")):
            value = re.sub(r"^(file://|file:|py_file:)", "", target).lstrip("/")
            if value.endswith("/") or (REPO_ROOT / value).is_dir():
                out = self.dir_pages.get(value.rstrip("/") or ".")
                return (rel_url(from_out, out), None) if out else (None, None)
            symbol = None
            if ":" in value:
                value, symbol = value.rsplit(":", 1)
                symbol = symbol.rstrip("()")
            out = self.file_pages.get(value)
            if out:
                if symbol:
                    line = self.symbols.get(value, {}).get(symbol) or self.symbols.get(value, {}).get(symbol.split(".")[-1])
                    if line:
                        return rel_url(from_out, out) + f"#L{line}", None
                return rel_url(from_out, out), None
            return None, None
        if target.startswith("dir:"):
            value = target[4:].strip("/") or "."
            out = self.dir_pages.get(value)
            return (rel_url(from_out, out), None) if out else (None, None)
        if target.startswith("env_vars"):
            suffix = target[len("env_vars"):].lstrip(":").strip("/")
            out = self.page_out("intro/env-vars")
            if out:
                return rel_url(from_out, out) + (f"#{slugify(suffix)}" if suffix else ""), None
            return None, None
        if target == "registry":
            return rel_url(from_out, Path("registry/index.html")), anchor
        if target.startswith(("registry:", "registry_category:")):
            value = target.split(":", 1)[1].strip("/")
            parts = value.split(".")
            category = self.singular(value) if len(parts) == 1 else None
            if len(parts) == 1:
                if value in self.registry_info:
                    return rel_url(from_out, Path("registry") / f"{slugify(value)}.html"), anchor
                if category in self.registry_info:
                    return rel_url(from_out, Path("registry") / f"{slugify(category)}.html"), anchor
            elif len(parts) >= 2:
                category = self.singular(parts[0])
                variant = ".".join(parts[1:])
                if category in self.registry_info and variant in self.registry_info[category]["variants"]:
                    url = rel_url(from_out, Path("registry") / slugify(category) / f"{slugify(variant)}.html")
                    return url, anchor
            return None, None
        if target.startswith(("dataclass:", "configuration:", "config:")):
            value = target.split(":", 1)[1].strip("/")
            resolved = self.resolve_dataclass(value)
            if resolved:
                category, variant = resolved
                return rel_url(from_out, Path("configuration") / slugify(category) / f"{slugify(variant)}.html"), anchor
            return None, None
        special = {
            "registry/index": Path("registry/index.html"),
            "configuration/index": Path("configuration/index.html"),
        }.get(target.strip("/"))
        if special:
            return rel_url(from_out, special), anchor
        direct = self.page_out(target.strip("/"))
        if direct:
            return rel_url(from_out, direct), anchor
        if target in self.page_out_by_title():
            return rel_url(from_out, self.page_out_by_title()[target]), anchor
        if target in self.alias:
            return self.resolve_target(self.alias[target], from_out)
        return None, None

    def singular(self, value: str) -> str:
        if value in self.registry_info:
            return value
        if value.endswith("s") and value[:-1] in self.registry_info:
            return value[:-1]
        return value

    def resolve_dataclass(self, value: str) -> Optional[tuple[str, str]]:
        parts = value.split(".")
        if len(parts) == 2:
            category = self.singular(parts[0])
            variant = parts[1]
            if category in self.registry_info and variant in self.registry_info[category]["variants"]:
                return category, variant
        if len(parts) == 1:
            category = self.singular(value)
            if category in self.registry_info:
                default = self.registry_info[category].get("default_variant")
                if default:
                    return category, default
            for cat, info in self.registry_info.items():
                for variant, variant_info in info["variants"].items():
                    definition = variant_info.get("definition")
                    if definition and definition.get("pkgname") == value:
                        return cat, variant
        return None

    def page_out_by_title(self) -> dict[str, Path]:
        return {page.title: page.out for page in self.pages.values()}

    def default_label(self, target: str) -> str:
        for prefix in ("registry_category:", "registry:", "dataclass:", "configuration:",
                       "config:", "page:", "tutorial:", "tutorials:", "env_vars:"):
            if target.startswith(prefix):
                value = target.split(":", 1)[1].strip("/")
                return value.replace("_", " ") or target
        if target.startswith(("file://", "file:", "py_file:")):
            value = re.sub(r"^(file://|file:|py_file:)", "", target).lstrip("/")
            return Path(value.split(":", 1)[0]).name or value
        if target.startswith("dir:"):
            return target[4:].strip("/") or "."
        return target

    def link(self, target: str, text: Optional[str], from_out: Path) -> str:
        href, anchor = self.resolve_target(target, from_out)
        if href is None:
            self.warnings.append(f"{from_out}: cannot resolve link target '{target}'")
            return f'<span class="broken-link" title="missing target: {html.escape(target)}">{html.escape(text or target)}</span>'
        if anchor:
            href += "#" + anchor
        label = text or self.default_label(target)
        return f'<a href="{html.escape(href, quote=True)}">{label}</a>'

    def rewrite_links(self, body: str, from_out: Path) -> str:
        def replace(match: re.Match) -> str:
            inside = match.group(1)
            target, _, text = inside.partition("|")
            target, text = target.strip(), text.strip()
            return self.link(target, text or None, from_out)

        parts = re.split(r"(```.*?```|~~~.*?~~~)", body, flags=re.DOTALL)
        for index, part in enumerate(parts):
            if part.startswith(("```", "~~~")):
                continue
            chunks = re.split(r"(`[^`]*`)", part)
            for chunk_index, chunk in enumerate(chunks):
                if chunk.startswith("`") and chunk.endswith("`"):
                    continue
                chunk = re.sub(r"\[\[([^\[\]]+?)\]\]", replace, chunk)
                chunk = self.rewrite_markdown_links(chunk, from_out)
                chunks[chunk_index] = chunk
            parts[index] = "".join(chunks)
        return "".join(parts)

    def rewrite_markdown_links(self, body: str, from_out: Path) -> str:
        pattern = re.compile(r"\]\(([^)\s]+?\.md)(#[^)]*)?\)")

        def replace(match: re.Match) -> str:
            target, anchor = match.group(1), match.group(2) or ""
            cleaned = target[2:] if target.startswith("./") else target
            candidates = [cleaned, cleaned[2:] if cleaned.startswith("../") else cleaned,
                          "about/" + Path(cleaned).name, Path(cleaned).name[:-3]]
            if cleaned.startswith("documentation/src/"):
                candidates.insert(0, cleaned[len("documentation/src/"):])
            for candidate in candidates:
                key = candidate[:-3] if candidate.endswith(".md") else candidate
                if key in self.pages:
                    return f"]({rel_url(from_out, self.pages[key].out)}{anchor})"
            return match.group(0)

        return pattern.sub(replace, body)

    # ------------------------------------------------------------------ rendering

    def banner(self, text: str, kind: str = "warning") -> str:
        return f'<div class="callout {kind}">{text}</div>'

    def preprocess(self, body: str) -> str:
        lines = body.splitlines()
        out: list[str] = []
        open_callout = False
        for line in lines:
            match = re.match(r"^>?\s*\[!(\w+)\]\s*(.*)$", line)
            if match:
                if open_callout:
                    out.append("</div>")
                kind = "warning" if "warn" in match.group(1).lower() else "note"
                out.append(f'<div class="callout {kind}" markdown="1">{match.group(2)}')
                open_callout = True
                continue
            if open_callout and not line.strip():
                out.append("</div>")
                open_callout = False
                out.append("")
                continue
            section = re.match(r"^!Section\s+(.+)$", line.strip())
            if section:
                out.append(f'<a id="{slugify(section.group(1))}"></a>')
                continue
            out.append(line)
        if open_callout:
            out.append("</div>")
        return "\n".join(out)

    def render_markdown(self, body: str) -> str:
        body = self.preprocess(body)
        return markdown.markdown(body, extensions=MD_EXTENSIONS, extension_configs=MD_EXTENSION_CONFIGS)

    def readme_page(self) -> SourcePage:
        body = read_text(REPO_ROOT / "readme.md")
        return SourcePage(REPO_ROOT / "readme.md", Path("index.html"), "index", {}, body, "Matformer")

    def source_block(self, paths: list[str], from_out: Path) -> str:
        if not paths:
            return ""
        links = []
        for path in paths:
            out = self.file_pages.get(path)
            label = f"<code>{html.escape(path)}</code>"
            if out:
                links.append(f'<a href="{html.escape(rel_url(from_out, out), quote=True)}">{label}</a>')
            else:
                links.append(label)
        return '<div class="page-sources"><span class="label">Source</span>' + " ".join(links) + "</div>"

    def render_source_page(self, page: SourcePage) -> RenderedPage:
        body = page.body
        if page.meta.get("readme"):
            body = read_text(REPO_ROOT / "readme.md")
        body = self.rewrite_links(body, page.out)
        body = self.expand_directives(body, page.out)
        rendered = self.render_markdown(body)
        sources = page.meta.get("sources") or []
        if isinstance(sources, str):
            sources = [sources]
        return RenderedPage(
            out=page.out,
            title=page.title,
            body=rendered,
            sources=sources,
            section=self.section_of(page.key),
            breadcrumbs=self.breadcrumbs_for_source(page),
            body_class="doc-page",
        )

    def section_of(self, key: str) -> Optional[str]:
        for section in self.nav:
            for entry in section["pages"]:
                if entry["key"] == key:
                    return section["title"]
        return key.split("/")[0].capitalize() if "/" in key else None

    def breadcrumbs_for_source(self, page: SourcePage) -> str:
        crumbs = ['<a href="' + html.escape(rel_url(page.out, Path("index.html")), quote=True) + '">Docs</a>']
        section = self.section_of(page.key)
        if section:
            crumbs.append(html.escape(section))
        crumbs.append(html.escape(page.title))
        return ' <span class="sep">/</span> '.join(crumbs)

    def expand_directives(self, body: str, from_out: Path) -> str:
        def replace(match: re.Match) -> str:
            directive = match.group(1).strip()
            name, _, argument = directive.partition(":")
            argument = argument.strip()
            if name == "registry_index":
                return self.registry_index_html(from_out)
            if name == "registry_category":
                return self.registry_category_html(argument, from_out)
            if name == "dataclass_index":
                return self.dataclass_index_html(from_out)
            if name == "env_vars":
                return self.env_vars_html(from_out)
            if name == "file_tree":
                return self.dir_listing_html(argument or ".", from_out)
            if name == "command_reference":
                return self.command_reference_html(from_out)
            return match.group(0)

        return re.sub(r"\{\{\s*([^{}]+?)\s*\}\}", replace, body)

    # ------------------------------------------------------------------ shared fragments

    def category_url(self, category: str, from_out: Path) -> str:
        return rel_url(from_out, Path("registry") / f"{slugify(category)}.html")

    def variant_url(self, category: str, variant: str, from_out: Path) -> str:
        return rel_url(from_out, Path("registry") / slugify(category) / f"{slugify(variant)}.html")

    def dataclass_url(self, category: str, variant: str, from_out: Path) -> str:
        return rel_url(from_out, Path("configuration") / slugify(category) / f"{slugify(variant)}.html")

    def category_default(self, category: str) -> Optional[str]:
        return self.registry_info.get(category, {}).get("default_variant")

    def category_is_multi(self, category: str) -> bool:
        try:
            category_object = self.registry.registry.get(category)
            return bool(getattr(category_object, "is_multi", False))
        except Exception:
            return False

    def registry_index_html(self, from_out: Path) -> str:
        rows = []
        for category in self.registry_categories:
            info = self.registry_info[category]
            variants = info["variants"]
            definitions = sum(1 for v in variants.values() if v.get("definition"))
            implementations = sum(len(v.get("implementations", {})) for v in variants.values())
            default = info.get("default_variant") or "-"
            rows.append(
                "<tr>"
                f'<td><a href="{html.escape(self.category_url(category, from_out), quote=True)}"><code>{html.escape(category)}</code></a></td>'
                f"<td>{len(variants)}</td><td>{definitions}</td><td>{implementations}</td>"
                f"<td><code>{html.escape(str(default))}</code></td>"
                "</tr>"
            )
        return (
            '<table class="registry-table"><thead><tr><th>Category</th><th>Variants</th>'
            "<th>Dataclasses</th><th>Implementations</th><th>Default</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    def registry_category_html(self, category: str, from_out: Path) -> str:
        category = self.singular(category)
        if category not in self.registry_info:
            return f'<p class="warning">Unknown registry category <code>{html.escape(category)}</code>.</p>'
        info = self.registry_info[category]
        rows = []
        for variant, variant_info in sorted(info["variants"].items()):
            definition = variant_info.get("definition", {})
            impls = variant_info.get("implementations", {})
            default = "yes" if info.get("default_variant") == variant else ""
            rows.append(
                "<tr>"
                f'<td><a href="{html.escape(self.variant_url(category, variant, from_out), quote=True)}"><code>{html.escape(variant)}</code></a></td>'
                f'<td><code>{html.escape(definition.get("pkgname", "-"))}</code></td>'
                f"<td>{len(impls)}</td><td>{default}</td>"
                "</tr>"
            )
        return (
            '<table class="registry-table"><thead><tr><th>Variant</th><th>Dataclass</th>'
            "<th>Implementations</th><th>Default</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    def dataclass_index_html(self, from_out: Path) -> str:
        rows = []
        for category in self.registry_categories:
            info = self.registry_info[category]
            variants = [v for v, vi in info["variants"].items() if vi.get("definition")]
            if not variants:
                continue
            links = ", ".join(
                f'<a href="{html.escape(self.dataclass_url(category, variant, from_out), quote=True)}"><code>{html.escape(variant)}</code></a>'
                for variant in sorted(variants)
            )
            rows.append(
                "<tr><td><a href=\"" + html.escape(self.category_url(category, from_out), quote=True) + "\"><code>" + html.escape(category) + "</code></a></td>"
                f"<td>{links}</td><td>{'multi' if self.category_is_multi(category) else 'single'}</td></tr>"
            )
        return (
            '<table class="registry-table"><thead><tr><th>Section</th><th>Variants</th><th>Kind</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table>"
        )

    def env_entries(self) -> list[dict]:
        path = REPO_ROOT / "matformer" / "utils" / "env_variables.py"
        entries: list[dict] = []
        group = "General"
        if path.exists():
            for line in read_text(path).splitlines():
                heading = re.match(r"^\s*#\s*---+\s*(.+?)\s*---+\s*$", line)
                if heading:
                    group = heading.group(1)
                    continue
                match = re.match(r"^\s*#\s*([A-Z][A-Z0-9_<>{}-]*)\s*:\s*(.+?)\s*$", line)
                if match:
                    entries.append({"name": match.group(1), "doc": match.group(2), "group": group, "files": set()})
        known = {entry["name"]: entry for entry in entries}
        env_pattern = re.compile(r"""(?:os\.environ(?:\.get)?\s*[\(\[]|os\.getenv\s*\()\s*['"]([A-Z][A-Z0-9_]*)['"]""")
        for source in (REPO_ROOT / "matformer").rglob("*.py"):
            if is_ignored(source.relative_to(REPO_ROOT)):
                continue
            try:
                text = read_text(source)
            except (UnicodeDecodeError, OSError):
                continue
            for name in env_pattern.findall(text):
                entry = known.get(name)
                if entry is None:
                    entry = {"name": name, "doc": "", "group": "Used in the code", "files": set()}
                    entries.append(entry)
                    known[name] = entry
                entry["files"].add(repo_relative(source))
        return entries

    def env_vars_html(self, from_out: Path) -> str:
        entries = self.env_entries()
        groups: dict[str, list[dict]] = {}
        for entry in entries:
            groups.setdefault(entry["group"], []).append(entry)
        blocks = []
        for group, items in groups.items():
            rows = []
            for item in sorted(items, key=lambda one: one["name"]):
                files = " ".join(
                    f'<a href="{html.escape(rel_url(from_out, self.file_pages[path]), quote=True)}"><code>{html.escape(Path(path).name)}</code></a>'
                    for path in sorted(item["files"]) if path in self.file_pages
                )
                doc = item["doc"] or "Used in the code below; no inline description yet."
                rows.append(
                    f'<tr id="{slugify(item["name"])}"><td><code>{html.escape(item["name"])}</code></td>'
                    f"<td>{doc}</td><td>{files}</td></tr>"
                )
            blocks.append(
                f'<h3>{html.escape(group)}</h3><table class="registry-table"><thead><tr>'
                "<th>Variable</th><th>Meaning</th><th>Source</th></tr></thead><tbody>"
                + "".join(rows) + "</tbody></table>"
            )
        return "".join(blocks)

    def command_reference_html(self, from_out: Path) -> str:
        blocks = []
        if self.rpc_commands:
            rows = []
            for item in self.rpc_commands:
                source = ""
                if item.get("file") and item["file"] in self.file_pages:
                    url = rel_url(from_out, self.file_pages[item["file"]])
                    if item.get("line"):
                        url += f"#L{item['line']}"
                    source = f'<a href="{html.escape(url, quote=True)}"><code>{html.escape(Path(item["file"]).name)}</code></a>'
                rows.append(f'<tr><td><code>{html.escape(item["name"])}</code></td><td>{html.escape(item["doc"])}</td><td>{source}</td></tr>')
            blocks.append(
                '<h2>Server RPC commands</h2><table class="registry-table"><thead><tr>'
                '<th>Method</th><th>Description</th><th>Source</th></tr></thead><tbody>'
                + "".join(rows) + "</tbody></table>"
            )
        if self.cli_commands:
            rows = []
            for item in self.cli_commands:
                options = "".join(
                    f'<tr><td><code>{html.escape(option["flags"])}</code></td><td>{html.escape(option["help"])}</td></tr>'
                    for option in item["options"]
                )
                rows.append(
                    f'<tr><td><code>{html.escape(item["name"])}</code></td><td><code>{html.escape(item["usage"])}</code>'
                    f'<table class="registry-table"><tbody>{options}</tbody></table></td></tr>'
                )
            blocks.append(
                '<h2>Command line entry points</h2><table class="registry-table"><thead><tr>'
                '<th>Entry point</th><th>Options</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>"
            )
        if self.mdat_commands:
            rows = "".join(
                f'<tr><td><code>{html.escape(item["name"])}</code></td><td><code>{html.escape(item["usage"])}</code></td>'
                f'<td>{html.escape(item["doc"])}</td></tr>' for item in self.mdat_commands
            )
            blocks.append(
                '<h2>MDAT shell commands</h2><table class="registry-table"><thead><tr>'
                '<th>Command</th><th>Usage</th><th>Description</th></tr></thead><tbody>'
                + rows + "</tbody></table>"
            )
        if self.slurm_commands:
            rows = "".join(
                f'<tr><td><code>{html.escape(item["name"])}</code></td><td>{html.escape(item["doc"])}</td></tr>'
                for item in self.slurm_commands
            )
            blocks.append(
                '<h2>SLURM commands</h2><table class="registry-table"><thead><tr>'
                '<th>Method</th><th>Description</th></tr></thead><tbody>' + rows + "</tbody></table>"
            )
        return "".join(blocks)

    def dir_listing_html(self, relative: str, from_out: Path) -> str:
        relative = relative.strip("/") or "."
        path = REPO_ROOT if relative == "." else REPO_ROOT / relative
        if not path.is_dir():
            return f'<p class="warning">No such folder: <code>{html.escape(relative)}</code></p>'
        rows = []
        for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            child_relative = repo_relative(child)
            if child.is_dir():
                if child_relative not in self.dir_pages:
                    continue
                href = rel_url(from_out, self.dir_pages[child_relative])
                label = f"{child.name}/"
                kind = "folder"
            else:
                if child_relative not in self.file_pages:
                    continue
                href = rel_url(from_out, self.file_pages[child_relative])
                label = child.name
                kind = child.suffix.lstrip(".") or "file"
            rows.append(
                f'<tr><td><a href="{html.escape(href, quote=True)}"><code>{html.escape(label)}</code></a></td>'
                f"<td>{html.escape(kind)}</td></tr>"
            )
        return (
            '<table class="registry-table"><thead><tr><th>Name</th><th>Kind</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>"
        )

    def sidebar_html(self, current: Optional[str], extra: Optional[str] = None) -> str:
        blocks = []
        for section in self.nav:
            items = []
            open_section = False
            for entry in section["pages"]:
                page = self.pages.get(entry["key"])
                if page is None:
                    continue
                label = entry["title"] or page.title
                if entry["key"] == current:
                    open_section = True
                    items.append(f'<li><a class="current" href="{html.escape(rel_url(self._current_out, page.out), quote=True)}">{html.escape(label)}</a></li>')
                else:
                    items.append(f'<li><a href="{html.escape(rel_url(self._current_out, page.out), quote=True)}">{html.escape(label)}</a></li>')
            flags = " open" if open_section else ""
            blocks.append(
                f'<details class="nav-section"{flags}><summary>{html.escape(section["title"])}</summary>'
                f'<ul>{"".join(items)}</ul></details>'
            )
        if extra:
            blocks.append(extra)
        return "".join(blocks)

    # ------------------------------------------------------------------ special pages

    def category_sidebar(self, category: str, current_out: Path) -> str:
        items = []
        for name in self.registry_categories:
            cls = "current" if name == category else ""
            items.append(
                f'<li><a class="{cls}" href="{html.escape(self.category_url(name, current_out), quote=True)}">'
                f"<code>{html.escape(name)}</code></a></li>"
            )
        return f'<details class="nav-section" open><summary>Registry categories</summary><ul>{"".join(items)}</ul></details>'

    def breadcrumbs(self, current_out: Path, parts: list[tuple[str, Optional[Path]]]) -> str:
        crumbs = [f'<a href="{html.escape(rel_url(current_out, Path("index.html")), quote=True)}">Docs</a>']
        for label, target in parts:
            if target is None:
                crumbs.append(html.escape(label))
            else:
                crumbs.append(f'<a href="{html.escape(rel_url(current_out, target), quote=True)}">{html.escape(label)}</a>')
        return ' <span class="sep">/</span> '.join(crumbs)

    def render_registry_index(self) -> RenderedPage:
        out = Path("registry/index.html")
        body = (
            "<h1>The Registry</h1>"
            "<p>Everything the framework can build is registered here. Pick a category to see its "
            "variants, their configuration dataclasses and the implementations available on this machine.</p>"
            + self.registry_index_html(out)
        )
        return RenderedPage(out, "Registry", body, [], "Components",
                            sidebar=self.category_sidebar("", out),
                            breadcrumbs=self.breadcrumbs(out, [("Registry", None)]),
                            body_class="reference-page")

    def render_registry_category(self, category: str) -> RenderedPage:
        out = Path("registry") / f"{slugify(category)}.html"
        info = self.registry_info[category]
        default = info.get("default_variant") or "none"
        doc_key = self.category_docs.get(category)
        doc_link = ""
        if doc_key and doc_key in self.pages:
            doc_link = f'<p><a href="{html.escape(rel_url(out, self.pages[doc_key].out), quote=True)}">Read the description</a></p>'
        kind = "multi" if self.category_is_multi(category) else "single"
        body = (
            f"<h1>Registry: <code>{html.escape(category)}</code></h1>"
            f'<p>Default variant: <code>{html.escape(str(default))}</code>. This section is <strong>{kind}</strong>.</p>'
            + doc_link
            + self.registry_category_html(category, out)
        )
        return RenderedPage(
            out, f"Registry · {category}", body, [],
            "Components",
            sidebar=self.category_sidebar(category, out),
            breadcrumbs=self.breadcrumbs(out, [("Registry", Path("registry/index.html")), (category, None)]),
            body_class="reference-page",
        )

    def render_registry_variant(self, category: str, variant: str) -> RenderedPage:
        out = Path("registry") / slugify(category) / f"{slugify(variant)}.html"
        info = self.registry_info[category]["variants"][variant]
        definition = info.get("definition")
        impls = info.get("implementations", {})
        parts = [f"<h1>Registry: <code>{html.escape(category)}.{html.escape(variant)}</code></h1>"]

        doc_key = self.variant_docs.get(f"{category}.{variant}") or self.variant_docs.get(variant)
        if doc_key and doc_key in self.pages:
            parts.append(
                f'<p class="doc-chip"><a href="{html.escape(rel_url(out, self.pages[doc_key].out), quote=True)}">'
                "Read the description of this variant</a></p>"
            )

        if definition:
            file_out = self.file_pages.get("matformer/" + definition["filepath"].replace("matformer/", ""))
            file_link = ""
            if file_out:
                source_url = rel_url(out, file_out) + f"#L{definition.get('line_start')}"
                file_link = f' · <a href="{html.escape(source_url, quote=True)}">source</a>'
            parts.append(
                "<h2>Configuration</h2>"
                f'<p>Dataclass <a href="{html.escape(self.dataclass_url(category, variant, out), quote=True)}">'
                f'<code>{html.escape(definition["pkgname"])}</code></a>'
                f" in <code>{html.escape(definition['filepath'])}</code>{file_link}</p>"
            )
        else:
            parts.append("<h2>Configuration</h2><p>This variant has no configuration dataclass.</p>")

        parts.append("<h2>Implementations</h2>")
        if impls:
            rows = []
            for name, impl in sorted(impls.items(), key=lambda item: (-item[1].get("priority", 0), item[0])):
                impl_file = self.file_pages.get("matformer/" + impl["filepath"].replace("matformer/", ""))
                impl_link = f'<code>{html.escape(impl["pkgname"])}</code>'
                if impl_file:
                    impl_url = rel_url(out, impl_file) + f"#L{impl.get('line_start')}"
                    impl_link = f'<a href="{html.escape(impl_url, quote=True)}"><code>{html.escape(impl["pkgname"])}</code></a>'
                requires = ", ".join(impl.get("requires") or []) or "-"
                constraints = impl.get("constraints") or {}
                constraints_text = ", ".join(f"{k}={v}" for k, v in constraints.items()) or "-"
                metadata = impl.get("metadata") or {}
                metadata_text = ", ".join(f"{k}={v}" for k, v in metadata.items()) or "-"
                rows.append(
                    f'<tr id="impl-{slugify(name)}"><td><code>{html.escape(name)}</code></td>'
                    f"<td>{impl_link}</td><td>{impl.get('priority', 0)}</td>"
                    f"<td>{html.escape(requires)}</td><td>{html.escape(constraints_text)}</td>"
                    f"<td>{html.escape(metadata_text)}</td></tr>"
                )
            parts.append(
                '<table class="registry-table"><thead><tr><th>Implementation</th><th>Class</th>'
                "<th>Priority</th><th>Requires</th><th>Constraints</th><th>Metadata</th></tr></thead><tbody>"
                + "".join(rows) + "</tbody></table>"
            )
        else:
            parts.append("<p>No implementation is registered for this variant.</p>")

        return RenderedPage(
            out, f"Registry · {category}.{variant}", "".join(parts), [],
            "Components",
            sidebar=self.category_sidebar(category, out),
            breadcrumbs=self.breadcrumbs(
                out,
                [("Registry", Path("registry/index.html")),
                 (category, Path("registry") / f"{slugify(category)}.html"),
                 (variant, None)],
            ),
            body_class="reference-page",
        )

    def render_configuration_index(self) -> RenderedPage:
        out = Path("configuration/index.html")
        body = (
            "<h1>Configuration reference</h1>"
            "<p>Every registered dataclass, grouped the way a YAML file addresses it. "
            "This is generated from the dataclasses themselves, so it cannot drift from the code.</p>"
            + self.dataclass_index_html(out)
        )
        return RenderedPage(out, "Configuration", body, [], "Components",
                            sidebar=self.category_sidebar("", out),
                            breadcrumbs=self.breadcrumbs(out, [("Configuration", None)]),
                            body_class="reference-page")

    def render_configuration_category(self, category: str) -> RenderedPage:
        out = Path("configuration") / f"{slugify(category)}.html"
        variants = [v for v, vi in self.registry_info[category]["variants"].items() if vi.get("definition")]
        parts = [f"<h1>Configuration: <code>{html.escape(category)}</code></h1>",
                 f'<p><a href="{html.escape(self.category_url(category, out), quote=True)}">Back to the registry category</a></p>',
                 '<table class="registry-table"><thead><tr><th>Variant</th><th>Dataclass</th></tr></thead><tbody>']
        for variant in sorted(variants):
            definition = self.registry_info[category]["variants"][variant]["definition"]
            parts.append(
                f'<tr><td><a href="{html.escape(self.dataclass_url(category, variant, out), quote=True)}"><code>{html.escape(variant)}</code></a></td>'
                f'<td><code>{html.escape(definition["pkgname"])}</code></td></tr>'
            )
        parts.append("</tbody></table>")
        return RenderedPage(
            out, f"Configuration · {category}", "".join(parts), [], "Components",
            sidebar=self.category_sidebar(category, out),
            breadcrumbs=self.breadcrumbs(
                out, [("Configuration", Path("configuration/index.html")), (category, None)]
            ),
            body_class="reference-page",
        )

    def render_configuration_variant(self, category: str, variant: str) -> RenderedPage:
        out = Path("configuration") / slugify(category) / f"{slugify(variant)}.html"
        definition = self.registry_info[category]["variants"][variant].get("definition")
        if not definition:
            return self.render_registry_variant(category, variant)
        parts = [f"<h1>Configuration: <code>{html.escape(category)}.{html.escape(variant)}</code></h1>"]

        try:
            from matformer.utils.schema import fields_schema, field_sections

            cls = self.registry.get_definition(category, variant)
            fields = fields_schema(cls)
            sections = field_sections(cls)
            doc_name = cls.__name__
        except Exception as problem:
            self.warnings.append(f"cannot introspect {category}.{variant}: {problem}")
            cls, fields, sections, doc_name = None, [], {}, definition["pkgname"]

        registry_link = self.variant_url(category, variant, out)
        parts.append(
            f'<p>Dataclass <code>{html.escape(doc_name)}</code> · '
            f'<a href="{html.escape(registry_link, quote=True)}">registry entry</a> · '
            f'<a href="{html.escape(self.category_url(category, out), quote=True)}">category</a></p>'
        )

        if cls is not None:
            doc = (cls.__doc__ or "").strip()
            if doc and not doc.startswith(f"{doc_name}("):
                parts.append(f'<div class="class-doc">{self.render_markdown(textwrap.dedent(doc))}</div>')

        groups: dict[str, list[dict]] = {}
        for field in fields:
            groups.setdefault(sections.get(field["name"], "Fields"), []).append(field)

        for section, items in groups.items():
            parts.append(f"<h2>{html.escape(section)}</h2>")
            parts.append(
                '<table class="fields"><thead><tr><th>Field</th><th>Type</th><th>Default</th>'
                "<th>Meaning</th></tr></thead><tbody>"
            )
            for field in items:
                default = self.format_default(field.get("default"))
                options = ""
                if field.get("options"):
                    options = '<div class="options">' + ", ".join(
                        f"<code>{html.escape(str(option))}</code>" for option in field["options"]
                    ) + "</div>"
                parts.append(
                    f'<tr><td id="field-{slugify(field["name"])}"><code>{html.escape(field["name"])}</code></td>'
                    f'<td><code>{html.escape(str(field.get("type") or "").replace("typing.", ""))}</code></td>'
                    f"<td><code>{default}</code></td>"
                    f'<td>{html.escape(field.get("doc") or "")}{options}</td></tr>'
                )
            parts.append("</tbody></table>")

        parts.append(
            f'<h2>In a YAML file</h2><pre><code class="language-yaml">{html.escape(category)}:\n'
            f"  type: {html.escape(variant)}</code></pre>"
        )

        return RenderedPage(
            out, f"Configuration · {category}.{variant}", "".join(parts), [], "Components",
            sidebar=self.category_sidebar(category, out),
            breadcrumbs=self.breadcrumbs(
                out,
                [("Configuration", Path("configuration/index.html")),
                 (category, Path("configuration") / f"{slugify(category)}.html"),
                 (variant, None)],
            ),
            body_class="reference-page",
        )

    @staticmethod
    def format_default(value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, default=str)
        elif isinstance(value, str):
            text = repr(value)
        else:
            text = str(value)
        text = text.replace("typing.", "")
        return html.escape(text if len(text) <= 180 else text[:177] + "...")

    def github_url(self, relative: str, kind: str = "blob") -> Optional[str]:
        base = (self.repo or {}).get("url")
        if not base:
            return None
        branch = (self.repo or {}).get("branch", "main")
        return f"{base.rstrip('/')}/{kind}/{branch}/{relative}"

    def render_file_page(self, relative: str) -> RenderedPage:
        path = REPO_ROOT / relative
        out = self.file_pages[relative]
        text = read_text(path)
        try:
            lexer = get_lexer_for_filename(path.name, text)
        except Exception:
            lexer = TextLexer()
        formatter = HtmlFormatter(linenos="table", lineanchors="L", cssclass="highlight", wrapcode=True)
        code = highlight(text, lexer, formatter)
        code = code.replace('id="L-', 'id="L').replace('name="L-', 'name="L')
        for item in self.file_imports.get(relative, []):
            target = self.resolve_module(item["module"])
            if not target:
                continue
            href = rel_url(out, self.file_pages[target])
            module_html = f'<span class="nn">{html.escape(item["module"])}</span>'
            code = code.replace(
                module_html,
                f'<a href="{html.escape(href, quote=True)}">{module_html}</a>',
            )
        for class_name, (_, bases) in self.class_info.get(relative, {}).items():
            for base in bases:
                location = self.class_location(base, relative)
                if not location:
                    continue
                base_file, base_line, _ = location
                target = self.file_pages.get(base_file)
                if not target:
                    continue
                href = (f"#L{base_line}" if base_file == relative else rel_url(out, target) + f"#L{base_line}")
                pattern = re.compile(
                    r'(<span class="k">class</span><span class="w"> </span><span class="nc">'
                    + re.escape(class_name) + r'</span>.*?<span class="n">'
                    + re.escape(base) + r'</span>)'
                )
                code = pattern.sub(
                    lambda match: match.group(1).replace(
                        f'<span class="n">{html.escape(base)}</span>',
                        f'<a href="{html.escape(href, quote=True)}"><span class="n">{html.escape(base)}</span></a>',
                        1,
                    ),
                    code,
                )
        symbols = self.symbols.get(relative, {})
        index_html = ""
        if symbols:
            rows = []
            for name, line in sorted(symbols.items(), key=lambda item: (item[1], item[0])):
                depth = name.count(".")
                suffix = ""
                if name in self.class_info.get(relative, {}):
                    bases = self.class_info[relative][name][1]
                    links = []
                    for base in bases:
                        location = self.class_location(base, relative)
                        if location:
                            file, base_line, _ = location
                            target = self.file_pages.get(file)
                            if target:
                                href = rel_url(out, target) + f"#L{base_line}"
                                links.append(f'<a href="{html.escape(href, quote=True)}"><code>{html.escape(base)}</code></a>')
                            else:
                                links.append(f"<code>{html.escape(base)}</code>")
                        else:
                            links.append(f"<code>{html.escape(base)}</code>")
                    if links:
                        suffix = ' <span class="bases">(' + ", ".join(links) + ")</span>"
                    chain = self.mro_chain(name, relative)
                    if len(chain) > 1:
                        mro = []
                        for file, chain_name, chain_line in chain:
                            target = self.file_pages.get(file)
                            if target and file != relative:
                                href = rel_url(out, target) + f"#L{chain_line}"
                                mro.append(f'<a href="{html.escape(href, quote=True)}"><code>{html.escape(chain_name)}</code></a>')
                            else:
                                mro.append(f"<code>{html.escape(chain_name)}</code>")
                        suffix += '<div class="mro">MRO: ' + " -> ".join(mro) + "</div>"
                rows.append(
                    f'<li class="depth-{depth}"><a href="#L{line}"><code>{html.escape(name)}</code></a>'
                    f'<span class="line">L{line}</span>{suffix}</li>'
                )
            index_html = f'<details class="symbol-index" open><summary>Classes and functions</summary><ul>{"".join(rows)}</ul></details>'
        imports_html = ""
        imports = self.file_imports.get(relative, [])
        if imports:
            rows = []
            seen = set()
            for item in imports:
                target = self.resolve_module(item["module"])
                if target is None:
                    continue
                key = (item["module"], tuple(item["names"]))
                if key in seen:
                    continue
                seen.add(key)
                href = rel_url(out, self.file_pages[target])
                rows.append(
                    f'<li><a href="{html.escape(href, quote=True)}"><code>{html.escape(item["module"])}</code></a>'
                    f' <span class="muted">{html.escape(", ".join(item["names"]))}</span></li>'
                )
            if rows:
                imports_html = f'<details class="symbol-index"><summary>Internal imports</summary><ul class="imports">{"".join(rows)}</ul></details>'
        parent = Path(relative).parent.as_posix()
        parent = "." if parent == "." else parent
        github = self.github_url(relative)
        github_link = (f' · <a href="{html.escape(github, quote=True)}">view on GitHub</a>' if github else "")
        body = (
            '<a id="top"></a><a class="back-to-top" href="#top">Top</a>'
            f'<div class="file-header"><p><a href="{html.escape(rel_url(out, self.dir_pages[parent]), quote=True)}">'
            f'<code>{html.escape(parent)}/</code></a></p>'
            f"<h1><code>{html.escape(Path(relative).name)}</code></h1>"
            f'<p class="muted">{html.escape(relative)} · {len(text.splitlines())} lines{github_link}</p></div>'
            + index_html
            + imports_html
            + f'<div class="source-view">{code}</div>'
        )
        return RenderedPage(
            out, relative, body, [relative], None,
            sidebar=None,
            breadcrumbs=self.breadcrumbs(out, [("Repository", Path("file/index.html")), (relative, None)]),
            body_class="file-page",
        )

    def render_dir_page(self, relative: str) -> RenderedPage:
        out = self.dir_pages[relative]
        path = REPO_ROOT if relative == "." else REPO_ROOT / relative
        title = "Repository" if relative == "." else relative
        body = f"<h1><code>{html.escape(title)}</code></h1>" + self.dir_listing_html(relative, out)
        return RenderedPage(
            out, title, body, [], None,
            sidebar=None,
            breadcrumbs=self.breadcrumbs(out, [("Files", Path("file/index.html")), (title, None)]),
            body_class="reference-page",
        )

    # ------------------------------------------------------------------ templates

    def top_nav(self, current: Optional[str]) -> str:
        current_out = self._current_out
        entries = [
            ("Documentation", Path("index.html"), "index"),
            ("Registry", Path("registry/index.html"), "registry"),
            ("Configuration", Path("configuration/index.html"), "configuration"),
            ("Commands", self.pages.get("tools/commands").out if self.pages.get("tools/commands") else Path("tools/commands.html"), "tools/commands"),
            ("Repository", Path("file/index.html"), "file"),
        ]
        parts = []
        value = current or ""
        for label, target, marker in entries:
            if marker == "file":
                active = value.startswith("file")
            elif marker == "registry":
                active = value.startswith("registry")
            elif marker == "configuration":
                active = value.startswith("configuration")
            else:
                active = value == marker
            parts.append(
                f'<a class="{"current" if active else ""}" href="{html.escape(rel_url(current_out, target), quote=True)}">{html.escape(label)}</a>'
            )
        github = self.repo.get("url") if self.repo else None
        if github:
            parts.append(f'<a class="external" href="{html.escape(github, quote=True)}">GitHub</a>')
        return "".join(parts)

    def write_page(self, page: RenderedPage) -> None:
        self._current_out = page.out
        header = read_text(TEMPLATES / "header.html")
        footer = read_text(TEMPLATES / "footer.html")
        root = os.path.relpath(SITE, (SITE / page.out).parent).replace(os.sep, "/")
        root = "" if root == "." else root + "/"
        sidebar = page.sidebar if page.sidebar is not None else self.sidebar_html(
            self.key_of_out(page.out)
        )
        shown_title = page.title if page.title == "Matformer" else f"{page.title} · Matformer"
        values = {
            "title": html.escape(shown_title),
            "root": html.escape(root, quote=True),
            "nav": self.top_nav(self.key_of_out(page.out) or page.out.as_posix()),
            "sidebar": sidebar,
            "breadcrumbs": page.breadcrumbs or "",
            "sources": self.source_block(page.sources, page.out),
            "body_class": page.body_class,
            "build_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        head = header
        for key, value in values.items():
            head = head.replace("{{" + key + "}}", value)
        foot = footer
        for key, value in values.items():
            foot = foot.replace("{{" + key + "}}", value)
        target = SITE / page.out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(head + page.body + foot, encoding="utf-8")

    def key_of_out(self, out: Path) -> Optional[str]:
        for key, page in self.pages.items():
            if page.out == out:
                return key
        return None

    # ------------------------------------------------------------------ build

    def build(self, clean: bool = False) -> None:
        if clean and SITE.exists():
            shutil.rmtree(SITE)
        SITE.mkdir(parents=True, exist_ok=True)
        self._current_out = Path("index.html")

        self.load_registry()
        self.load_repo()
        self.load_commands()
        self.load_aliases()
        self.load_nav()
        self.load_sources()
        self.map_variant_docs()
        self.build_inventory()

        for page in self.pages.values():
            self.generated.append(self.render_source_page(page))
        self.generated.append(self.render_registry_index())
        for category in self.registry_categories:
            self.generated.append(self.render_registry_category(category))
            for variant in sorted(self.registry_info[category]["variants"]):
                self.generated.append(self.render_registry_variant(category, variant))
        self.generated.append(self.render_configuration_index())
        for category in self.registry_categories:
            if any(v.get("definition") for v in self.registry_info[category]["variants"].values()):
                self.generated.append(self.render_configuration_category(category))
                for variant, info in sorted(self.registry_info[category]["variants"].items()):
                    if info.get("definition"):
                        self.generated.append(self.render_configuration_variant(category, variant))
        for relative in self.file_pages:
            self.generated.append(self.render_file_page(relative))
        for relative in self.dir_pages:
            self.generated.append(self.render_dir_page(relative))

        seen = set()
        for page in self.generated:
            if page.out in seen:
                self.warnings.append(f"duplicate output page {page.out}")
            seen.add(page.out)
            self.write_page(page)

        self.write_assets()
        self.check_links()
        self.report()

    def write_assets(self) -> None:
        assets = SITE / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TEMPLATES / "style.css", assets / "style.css")
        pygments_css = HtmlFormatter(style="friendly").get_style_defs(".highlight")
        (assets / "pygments.css").write_text(pygments_css, encoding="utf-8")

    def check_links(self) -> None:
        for html_file in SITE.rglob("*.html"):
            text = read_text(html_file)
            for match in re.finditer(r'href="([^"]+)"', text):
                href = html.unescape(match.group(1))
                if href.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path, _, fragment = href.partition("#")
                if not path:
                    continue
                target = (html_file.parent / urllib.parse.unquote(path)).resolve()
                if not target.exists():
                    self.warnings.append(f"{repo_relative(html_file)}: broken href {href}")
                elif fragment and target.suffix == ".html":
                    ids = set(re.findall(r'id="([^"]+)"', read_text(target)))
                    if fragment not in ids:
                        self.warnings.append(f"{repo_relative(html_file)}: missing anchor {href}")

    def report(self) -> None:
        print(f"Documentation built in {repo_relative(SITE)}")
        print(f"  {len(self.pages)} markdown pages")
        print(f"  {len(self.file_pages)} file pages, {len(self.dir_pages)} folder pages")
        print(f"  {len(self.registry_categories)} registry categories")
        if self.warnings:
            print(f"  {len(self.warnings)} warning(s):")
            for warning in self.warnings[:80]:
                print(f"    - {warning}")
            if len(self.warnings) > 80:
                print(f"    ... and {len(self.warnings) - 80} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="remove the previous site first")
    arguments = parser.parse_args()
    Documentation().build(clean=arguments.clean)


if __name__ == "__main__":
    main()
