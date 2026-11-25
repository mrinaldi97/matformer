# matformer/utils/chat_template.py
"""Self-contained chat templating: render and tokenize HuggingFace-style Jinja chat templates
directly, without transformers.

The template source is plain Jinja, exactly the one shipped in a model's tokenizer_config.json or
chat_template.jinja, so any standard template (Qwen, Llama, Mistral, ChatML, Zephyr...) works
unchanged. The only variables injected into a template's world are the ones HuggingFace injects
too: messages, tools, documents, add_generation_prompt and the tokenizer's special tokens (their
own names, e.g. the eos/bos/pad token strings). Everything else a template reads (enable_thinking
and friends) is a toggle surfaced by ChatTemplate.features and passed back through options.

Two entry points, because the two jobs differ:

    render(...)   -> str         the whole conversation as one string. Correct across token
                                 boundaries: what the tokenizer sees is what the model saw during
                                 training. Use it for generation and inference.
    encode(...)   -> list[int]   tokenize each message's content on its own and glue it between
                                 the template's static segments, which are tokenized once and
                                 cached. Fast over big datasets, and it keeps the content's ids
                                 independent of the control tokens around it (Matformer's
                                 pretokenization convention).

Both take the same arguments. tokenize (for encode) is any str -> list[int] callable that adds no
special tokens of its own: the template owns those.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence

import jinja2
import jinja2.ext
import jinja2.meta
from jinja2.sandbox import ImmutableSandboxedEnvironment

# Private-use-area characters: never produced by a real tokenizer, so they are safe sentinels.
_OPEN = chr(0xE000)
_CLOSE = chr(0xE001)
_CONT = chr(0xE002)
_END_CONT = chr(0xE003)
_CONTINUE_TAG = _CONT + "CONTINUE" + _END_CONT

# Variables the renderer always provides: never reported as a template feature.
_PROVIDED = frozenset({
    "messages", "tools", "documents", "add_generation_prompt",
    "bos_token", "eos_token", "unk_token", "pad_token", "padding_token", "mask_token",
    "sep_token", "cls_token", "additional_special_tokens", "raise_exception", "strftime_now",
})
# Message-field names some templates also use as bare "set" targets: Jinja reports them as
# undeclared, but they are not feature toggles, so they are filtered out of the feature list.
_MESSAGE_FIELDS = frozenset({"content", "role", "name", "reasoning_content", "tool_calls",
                             "tool_call_name", "tool_call_arguments", "last_query_index",
                             "multi_step_tool", "loop"})


def _raise_exception(message):
    raise jinja2.TemplateError(message)


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                      sort_keys=sort_keys)


def _strftime_now(fmt):
    return datetime.now().strftime(fmt)


@lru_cache(maxsize=None)
def _compile(source):
    """Compile once per distinct source; the environment mirrors transformers' own sandbox."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = _strftime_now
    return env.from_string(source)


@lru_cache(maxsize=None)
def _feature_names(source):
    """Undeclared variables a template reads beyond the standard ones: its feature toggles."""
    try:
        ast = _compile(source).environment.parse(source)
        names = (jinja2.meta.find_undeclared_variables(ast) - _PROVIDED - _MESSAGE_FIELDS
                 - _assigned_names(source))
        return tuple(sorted(names))
    except Exception:
        return ()


def _assigned_names(source):
    """Names a template binds itself (set / for / macro / import as): Jinja's undeclared-variable
    finder reports them anyway because control-structure scoping hides the binding, so they are
    subtracted from the feature list. Patterns are written without backslashes on purpose."""
    ident = "[A-Za-z_][A-Za-z0-9_]*"
    names = set()
    for pattern in ("set[ ]+(" + ident + ")",
                    "for[ ]+([A-Za-z0-9_, ]+?)[ ]in[ ]",
                    "macro[ ]+(" + ident + ")",
                    "call[ ]+(" + ident + ")",
                    "as[ ]+(" + ident + ")"):
        for found in re.findall(pattern, source):
            for piece in found.split(","):
                piece = piece.strip()
                if piece:
                    names.add(piece)
    return names


class ChatTemplate:
    """One model's chat template, ready to render text or produce token ids.

    Args:
        source:  the Jinja template (the raw chat_template string).
        special: {name: token-string} injected as template globals (bos/eos/pad ...).
    """

    def __init__(self, source, special=None):
        if not source or not source.strip():
            raise ValueError("ChatTemplate needs a non-empty Jinja source.")
        self.source = source
        self.special = dict(special or {})
        self._compiled = _compile(source)
        self._segment_cache = {}

    @property
    def features(self):
        """Template-level toggles (enable_thinking, ...) a caller can set through options."""
        return list(_feature_names(self.source))

    @staticmethod
    def default_option(feature):
        """A sensible default per feature name: thinking models reason unless told not to."""
        return not feature.startswith("no_")

    def __bool__(self):
        return True

    def __repr__(self):
        return "ChatTemplate(features=" + repr(self.features) + ")"

    # --- rendering ---
    def _context(self, messages, tools, documents, add_generation_prompt, options):
        ctx = dict(self.special)
        ctx.setdefault("padding_token", self.special.get("pad_token"))
        for feature in self.features:
            ctx[feature] = self.default_option(feature)
        ctx.update(messages=messages, tools=tools, documents=documents,
                   add_generation_prompt=add_generation_prompt)
        ctx.update(options or {})
        return ctx

    def render(self, messages, tools=None, documents=None, add_generation_prompt=False,
               continue_final_message=False, options=None):
        """The conversation as a single string, the way the model was trained to see it."""
        field = continue_final_message if isinstance(continue_final_message, str) else "content"
        chat = messages
        if continue_final_message and messages:
            chat = [dict(m) for m in messages]
            if chat[-1].get(field) is None:
                raise ValueError("continue_final_message: last message has no " + repr(field)
                                 + " to continue.")
            chat[-1][field] = str(chat[-1][field]) + _CONTINUE_TAG
        rendered = self._compiled.render(**self._context(
            chat, tools, documents, add_generation_prompt and not continue_final_message, options))
        if continue_final_message:
            rendered = _strip_continue_tag(rendered, str(messages[-1][field]))
        return rendered

    def segments(self, messages, tools=None, documents=None, add_generation_prompt=False,
                 options=None):
        """Split the template around each message's content into [('lit', text) | ('content', text)].
        Returns None when the split is unreliable (a sentinel merged or a content vanished), in
        which case the caller falls back to :meth:`encode_whole`."""
        real = [m.get("content") for m in messages]
        marked = []
        sentinels = []
        for index, message in enumerate(messages):
            body = dict(message)
            sentinel = _OPEN + str(index) + _CLOSE if isinstance(body.get("content"), str) else None
            if sentinel is not None:
                body["content"] = sentinel
            elif body.get("content") is None:
                body["content"] = ""
            marked.append(body)
            sentinels.append(sentinel)
        try:
            rendered = self.render(marked, tools=tools, documents=documents,
                                   add_generation_prompt=add_generation_prompt, options=options)
        except Exception:
            return None
        pieces, cursor, index = [], 0, 0
        for sentinel in sentinels:
            if sentinel is None:
                pieces.append(("lit", rendered[cursor:]))
                cursor = len(rendered)
                pieces.append(("content", real[index]))
            else:
                hit = rendered.find(sentinel, cursor)
                if hit < 0:
                    return None
                pieces.append(("lit", rendered[cursor:hit]))
                pieces.append(("content", real[index]))
                cursor = hit + len(sentinel)
            index += 1
        if cursor < len(rendered):
            pieces.append(("lit", rendered[cursor:]))
        return pieces

    def _literal_ids(self, text, tokenize):
        if not text:
            return []
        cached = self._segment_cache.get(text)
        if cached is None:
            cached = list(tokenize(text))
            self._segment_cache[text] = cached
        return cached

    def encode(self, messages, tokenize, tools=None, documents=None, add_generation_prompt=False,
               continue_final_message=False, options=None):
        """Token ids for the conversation. Splits content from control tokens (fast, cacheable);
        falls back to tokenizing the whole rendered string when that path is unavailable."""
        if continue_final_message:
            return self.encode_whole(messages, tokenize, tools=tools, documents=documents,
                                     options=options, continue_final_message=continue_final_message)
        split = self.segments(messages, tools=tools, documents=documents,
                              add_generation_prompt=add_generation_prompt, options=options)
        if split is None:
            return self.encode_whole(messages, tokenize, tools=tools, documents=documents,
                                     add_generation_prompt=add_generation_prompt, options=options)
        ids = []
        for kind, value in split:
            ids.extend(self._literal_ids(value, tokenize) if kind == "lit" else list(tokenize(value)))
        return ids

    def encode_whole(self, messages, tokenize, **render_kwargs):
        """Tokenize the fully rendered string in one go: exact at every boundary."""
        return list(tokenize(self.render(messages, **render_kwargs)))


def _strip_continue_tag(rendered, final_text):
    """Drop the marker and everything the template added after the point of continuation, keeping
    the text the model must continue from (mirrors transformers' trailing-space care)."""
    head = rendered.rfind(_CONTINUE_TAG)
    if head < 0:
        raise ValueError("continue_final_message: the final message did not survive the template.")
    cut = rendered[:head]
    tail = rendered[head + len(_CONTINUE_TAG):]
    if tail.strip() and final_text.rstrip() not in cut:
        return cut[:len(cut.rstrip())]
    return cut
