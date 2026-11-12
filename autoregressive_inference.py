#!/usr/bin/env python3
"""
autoregressive_inference.py
────────────────────────────────────────────────────────────────────────────
Interactive CLI for autoregressive text generation — Matformer framework.

Future extraction points are tagged [MODULE: <name>] in comments:
  console    → matformer/tui/console.py
  gen_config → matformer/config/generation.py
  streaming  → matformer/utils/streaming.py
  metrics    → matformer/utils/metrics.py
  loader     → matformer/utils/loader.py
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import traceback

import torch
import torch.nn.functional as F
from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor
from matformer.matformer_registry import registry
from matformer.matformer_wrapper import load_model
from matformer.matformer_tokenizers import MatformerTokenizer
from matformer.transformer_blocks import Autoregressive_Model
torch.backends.nnpack.enabled = False 
torch.backends.nnpack.set_flags(False)
# ════════════════════════════════════════════════════════════════════════════
# [MODULE: console]  ANSI formatting — zero external dependencies
# ════════════════════════════════════════════════════════════════════════════

def _tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _terminal_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


class S:
    """Namespace of ANSI style constants.  All empty strings on non-TTY."""
    _on = _tty()
    RST  = "\033[0m"  if _on else ""
    BOLD = "\033[1m"  if _on else ""
    DIM  = "\033[2m"  if _on else ""

    RED  = "\033[31m" if _on else ""
    GRN  = "\033[32m" if _on else ""
    YLW  = "\033[33m" if _on else ""
    BLU  = "\033[34m" if _on else ""
    CYN  = "\033[36m" if _on else ""

    BGRN = "\033[92m" if _on else ""  # bright green  — generated text
    BYLW = "\033[93m" if _on else ""  # bright yellow — warnings


def rule(char: str = "─", width: Optional[int] = None) -> str:
    w = width or _terminal_width()
    return f"{S.DIM}{char * w}{S.RST}"


def header(text: str) -> str:
    r = rule()
    return f"{r}\n{S.BOLD}{S.CYN}  {text}{S.RST}\n{r}"


def kv(key: str, value, width: int = 18) -> str:
    return f"  {S.DIM}{key:<{width}}{S.RST}{S.BOLD}{value}{S.RST}"


def ok(msg: str)   -> str: return f"{S.BGRN}✔{S.RST}  {msg}"
def warn(msg: str) -> str: return f"{S.BYLW}⚠{S.RST}  {msg}"
def err(msg: str)  -> str: return f"{S.RED}✖{S.RST}  {msg}"


def stat_bar(stats: Dict[str, str]) -> str:
    sep = f"  {S.DIM}│{S.RST}  "
    parts = [f"{S.DIM}{k}:{S.RST} {S.BOLD}{v}{S.RST}" for k, v in stats.items()]
    return "  " + sep.join(parts)


def prompt_prefix() -> str:
    return f"{S.BOLD}{S.BLU}>>{S.RST} "


# ════════════════════════════════════════════════════════════════════════════
# [MODULE: gen_config]  Generation parameters — live-editable at runtime
# ════════════════════════════════════════════════════════════════════════════

def _bool(s: str) -> bool:
    return s.lower() not in ("0", "false", "no", "off")


# (type_fn, description, hint) — drives both /set parsing and /show display
_FIELD_META: Dict[str, Tuple] = {
    "max_new_tokens":       (int,   "Max new tokens to generate",            "int > 0"),
    "min_new_tokens":       (int,   "Min new tokens before EOS allowed",     "int ≥ 0"),
    "temperature":          (float, "Sampling temperature",                  "float > 0"),
    "top_k":                (int,   "Top-k sampling (0 = disabled)",         "int ≥ 0"),
    "top_p":                (float, "Nucleus sampling threshold",            "0.0 – 1.0"),
    "repetition_penalty":   (float, "Penalty for already-seen tokens",       "float ≥ 1.0"),
    "no_repeat_ngram_size": (int,   "Forbid n-gram repetition (0=disabled)", "int ≥ 0"),
    "do_sample":            (_bool, "Sampling vs. greedy decoding",          "true/false"),
    "stream":               (_bool, "Print tokens as they are generated",    "true/false"),
    "show_perplexity":      (_bool, "Show perplexity after generation",      "true/false"),
    "show_timing":          (_bool, "Show tokens/s after generation",        "true/false"),
}


@dataclass
class GenerationConfig:
    """All user-tunable generation settings.  [MODULE: gen_config]"""
    max_new_tokens:       int   = 200
    min_new_tokens:       int   = 0
    temperature:          float = 0.7
    top_k:                int   = 0
    top_p:                float = 0.9
    repetition_penalty:   float = 1.0
    no_repeat_ngram_size: int   = 0
    do_sample:            bool  = True
    # UI toggles — not forwarded to generate()
    stream:               bool  = True
    show_perplexity:      bool  = True
    show_timing:          bool  = True

    def generate_kwargs(self) -> dict:
        """Return the subset of fields consumed by model.generate()."""
        return dict(
            max_new_tokens       = self.max_new_tokens,
            min_new_tokens       = self.min_new_tokens,
            temperature          = self.temperature,
            top_k                = self.top_k,
            top_p                = self.top_p,
            repetition_penalty   = self.repetition_penalty,
            no_repeat_ngram_size = self.no_repeat_ngram_size,
            do_sample            = self.do_sample,
            output_scores        = True,   # always; perplexity needs it
            return_timing        = True,   # always; cheaply gated on show_timing
        )

    def set_field(self, key: str, raw: str) -> Tuple[str, str]:
        """Parse and apply one setting.  Returns (old_repr, new_repr)."""
        if key not in _FIELD_META:
            raise KeyError(f"Unknown setting '{key}'. Try /show for valid names.")
        type_fn = _FIELD_META[key][0]
        old = getattr(self, key)
        try:
            setattr(self, key, type_fn(raw))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Cannot set '{key}' to {raw!r}: {exc}") from exc
        return str(old), str(getattr(self, key))

    def as_table(self) -> str:
        w_name, w_val = 24, 10
        lines = [
            f"\n  {S.DIM}{'Setting':<{w_name}}{'Value':<{w_val}}Description{S.RST}",
            "  " + "─" * 62,
        ]
        for key, (_, desc, hint) in _FIELD_META.items():
            val = getattr(self, key)
            lines.append(
                f"  {S.CYN}{S.BOLD}{key:<{w_name}}{S.RST}"
                f"{S.YLW}{str(val):<{w_val}}{S.RST}"
                f"{S.DIM}{desc}  [{hint}]{S.RST}"
            )
        return "\n".join(lines) + "\n"


# ════════════════════════════════════════════════════════════════════════════
# [MODULE: streaming]  Incremental token-to-character decoder
# ════════════════════════════════════════════════════════════════════════════

class StreamDecoder:
    """Converts a stream of token IDs into printable character chunks.

    Decodes the full accumulated ID list on each push() and returns only the
    newly decodable characters.  This correctly handles tokenisers where a
    single token does not produce a valid UTF-8 string in isolation
    (SentencePiece ▁ prefix, multi-byte BPE merges, etc.).  O(n) per token
    but completely negligible at CLI throughput.  [MODULE: streaming]
    """

    def __init__(self, tokenizer, prompt_ids: List[int], eos_token_id: Optional[int] = None):
        self._tok      = tokenizer
        self._ids: List[int] = list(prompt_ids)
        self._eos      = eos_token_id
        self._cursor   = len(self._decode())   # chars already printed (prompt)

    def _decode(self) -> str:
        return self._tok.decode(self._ids)

    def push(self, token_id: int) -> str:
        """Append token, return any newly printable characters (empty on EOS)."""
        if token_id == self._eos:
            return ""
        self._ids.append(token_id)
        full      = self._decode()
        new_chars = full[self._cursor:]
        self._cursor = len(full)
        return new_chars


# ════════════════════════════════════════════════════════════════════════════
# [MODULE: metrics]  Post-generation quality metrics
# ════════════════════════════════════════════════════════════════════════════

def compute_perplexity(
    scores: torch.Tensor,         # (1, steps, vocab_size) raw pre-sampling logits
    generated_ids: torch.Tensor,  # (1, prompt_len + steps)
    prompt_len: int,
    eos_token_id: Optional[int] = None,
) -> Optional[float]:
    """Perplexity of the generated tokens under the model's own distribution.

    Computed from the stored raw logits (before temperature / top-p), giving
    the model's intrinsic confidence independently of the sampling parameters.
    Returns None if no real tokens were generated.  [MODULE: metrics]
    """
    gen_tokens = generated_ids[0, prompt_len:]        # (steps,)

    # Truncate at the first EOS (include it, exclude trailing pads)
    if eos_token_id is not None:
        eos_pos = (gen_tokens == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos):
            gen_tokens = gen_tokens[: eos_pos[0] + 1]

    n = len(gen_tokens)
    if n == 0:
        return None

    # Cast to float32: bfloat16 log-softmax loses too much precision
    step_logits = scores[0, :n].float()               # (n, vocab_size)
    log_probs   = F.log_softmax(step_logits, dim=-1)  # (n, vocab_size)
    chosen_lp   = log_probs[torch.arange(n, device=gen_tokens.device), gen_tokens]
    return torch.exp(-chosen_lp.mean()).item()


# ════════════════════════════════════════════════════════════════════════════
# [MODULE: loader]  Model and tokeniser loading helpers
# ════════════════════════════════════════════════════════════════════════════

def load_ar_model(checkpoint: str, tokenizer_arg, device: torch.device, hf: bool = False):
    """Load an autoregressive Matformer checkpoint.  [MODULE: loader]"""
    model_dict = load_model(
        checkpoint,
        map_location=device,
        dtype=torch.bfloat16    
    )
    m=model_dict['model']
    return model_dict['model'], model_dict['configs']


def encode_prompt(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    """Encode a string prompt → 2-D input_ids tensor (1, seq_len).  [MODULE: loader]"""
    ids = tokenizer.encode(text=text, add_bos=True, add_eos=False)["input_ids"][0]
    return torch.tensor([ids], dtype=torch.long, device=device)


def resolve_eos(model) -> Optional[int]:
    """Best-effort EOS token id from whatever the model exposes."""
    tok = getattr(model, "tokenizer", None)
    if tok is not None:
        eid = getattr(tok, "eos_token_id", None)
        if eid is not None:
            return eid
    cfg = getattr(getattr(model, "model", model), "config", None)
    tok_cfg = getattr(cfg, "tokenizer_config", None)
    return getattr(tok_cfg, "eos_token_id", None)


# ════════════════════════════════════════════════════════════════════════════
# Help text (kept as a constant so a future TUI can render it differently)
# ════════════════════════════════════════════════════════════════════════════

_HELP = f"""
{rule()}
  {S.BOLD}Commands{S.RST}   (type on a line by itself)

  {S.YLW}/help{S.RST}                    Show this message
  {S.YLW}/show{S.RST}                    Show all current generation settings
  {S.YLW}/set <param> <value>{S.RST}     Change a setting, e.g.:
                              {S.DIM}/set temperature 0.4
                              /set max_new_tokens 512
                              /set stream false{S.RST}
  {S.YLW}/reset{S.RST}                   Reset all settings to defaults
  {S.YLW}/quit{S.RST}  /  {S.YLW}/exit{S.RST}            Quit

  Everything else is a prompt.  {S.DIM}Ctrl-C interrupts generation.  Ctrl-D quits.{S.RST}
{rule()}"""


# ════════════════════════════════════════════════════════════════════════════
# Interactive session
# ════════════════════════════════════════════════════════════════════════════

class InteractiveSession:
    """Drives the REPL loop.

    Keeping this class separate from the CLI entry-point means a future
    prompt_toolkit or Textual wrapper can import it and call .generate()
    directly without touching argparse or sys.stdout.
    """

    def __init__(self, model, tokenizer, config: GenerationConfig, device: torch.device):
        self.model     = model
        self.tokenizer = tokenizer
        self.config    = config
        self.device    = device
        self._eos      = resolve_eos(model)

    # ── Command handlers ────────────────────────────────────────────────────

    def _cmd_help(self, _):
        print(_HELP)

    def _cmd_show(self, _):
        print(self.config.as_table())

    def _cmd_set(self, args: str):
        parts = args.split(maxsplit=1)
        if len(parts) != 2:
            print(warn("Usage: /set <param> <value>"))
            return
        key, raw = parts
        try:
            old, new = self.config.set_field(key, raw)
            print(ok(
                f"{S.BOLD}{key}{S.RST}:  "
                f"{S.DIM}{old}{S.RST} → {S.YLW}{new}{S.RST}"
            ))
        except (KeyError, ValueError) as exc:
            print(err(str(exc)))

    def _cmd_reset(self, _):
        self.config = GenerationConfig()
        print(ok("All settings reset to defaults."))

    _COMMANDS: Dict[str, Callable] = {
        "help":  _cmd_help,
        "show":  _cmd_show,
        "set":   _cmd_set,
        "reset": _cmd_reset,
    }

    def _dispatch(self, line: str) -> bool:
        """Return True if line was a /command (even an unknown one)."""
        if not line.startswith("/"):
            return False
        tokens = line[1:].split(maxsplit=1)
        verb   = tokens[0].lower()
        rest   = tokens[1] if len(tokens) > 1 else ""
        if verb in ("quit", "exit"):
            print("\nBye.")
            sys.exit(0)
        handler = self._COMMANDS.get(verb)
        if handler is None:
            print(warn(f"Unknown command '/{verb}'.  Try /help."))
        else:
            handler(self, rest)
        return True

    # ── Generation ──────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> None:
        """Encode prompt, run generation, print output and stats."""
        input_ids  = encode_prompt(self.model.tokenizer, prompt, self.device)
        prompt_len = input_ids.shape[1]
        cfg        = self.config

        kwargs = cfg.generate_kwargs()

        # ── Streaming ───────────────────────────────────────────────────────
        if cfg.stream:
            decoder = StreamDecoder(self.model.tokenizer, input_ids[0].tolist(), self._eos)
            print()

            def _stream_cb(next_tokens: torch.LongTensor, _step: int) -> None:
                chars = decoder.push(next_tokens[0].item())
                if chars:
                    print(S.BGRN + chars + S.RST, end="", flush=True)

            kwargs["stream_callback"] = _stream_cb
        else:
            print(f"\n{S.DIM}Generating…{S.RST}")

        # ── Run ─────────────────────────────────────────────────────────────
        # generate() always returns (ids, scores, tok_per_sec) here because
        # we hard-set output_scores=True and return_timing=True above.
        current_ids, all_scores, tok_per_sec = self.model.model.generate(
            input_ids=input_ids, **kwargs
        )

        # ── Non-streaming output ─────────────────────────────────────────────
        if not cfg.stream:
            gen_ids = current_ids[0, prompt_len:].tolist()
            print(f"\n{S.BGRN}{self.tokenizer.decode(gen_ids)}{S.RST}")

        # ── Stats ────────────────────────────────────────────────────────────
        n_new = current_ids.shape[1] - prompt_len
        stats: Dict[str, str] = {"tokens": str(n_new)}

        if cfg.show_perplexity:
            ppl = compute_perplexity(all_scores, current_ids, prompt_len, self._eos)
            if ppl is not None:
                stats["perplexity"] = f"{ppl:.2f}"

        if cfg.show_timing:
            stats["speed"] = f"{tok_per_sec:.1f} tok/s"

        print(f"\n{rule('─', width=44)}")
        print(stat_bar(stats))
        print()

    # ── REPL ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        print(_HELP)
        while True:
            try:
                line = input(prompt_prefix()).strip()
            except EOFError:
                print("\nBye.")
                break
            except KeyboardInterrupt:
                print()
                continue

            if not line:
                continue
            if self._dispatch(line):
                continue

            try:
                self.generate(line)
            except KeyboardInterrupt:
                print(f"\n{S.RST}{S.BYLW}  Generation interrupted.{S.RST}\n")
            except Exception as exc:
                print(err(f"Generation failed: {exc}"))
                traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autoregressive_inference",
        description="Interactive autoregressive generation — Matformer framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      required=True,  help="Path to checkpoint")
    p.add_argument("--hf",         action="store_true", help="Load from HuggingFace Hub")
    p.add_argument("--tokenizer",  default=None,   help="HF tokenizer identifier (optional)")
    p.add_argument("--device",     default=None,   help="Device override (cuda/cpu/mps)")

    g = p.add_argument_group("generation defaults (all changeable live with /set)")
    g.add_argument("--max_new_tokens",       type=int,   default=200)
    g.add_argument("--min_new_tokens",       type=int,   default=0)
    g.add_argument("--temperature",          type=float, default=0.7)
    g.add_argument("--top_k",               type=int,   default=0)
    g.add_argument("--top_p",               type=float, default=0.9)
    g.add_argument("--repetition_penalty",   type=float, default=1.0)
    g.add_argument("--no_repeat_ngram_size", type=int,   default=0)
    g.add_argument("--no_sample",    action="store_true", help="Greedy decoding")
    g.add_argument("--no_stream",    action="store_true", help="Disable token streaming")
    g.add_argument("--no_perplexity",action="store_true", help="Hide perplexity")
    g.add_argument("--no_timing",    action="store_true", help="Hide speed stat")
    return p


def main() -> None:
    args   = _build_parser().parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer_arg = (
        MatformerTokenizer.from_huggingface(args.tokenizer)
        if args.tokenizer else "default"
    )
    print(header("Matformer  ◆  Interactive Generation"))
    print(kv("Model",  args.model))
    print(kv("Device", device))
    print()

    print(f"{S.DIM}Loading model…{S.RST}", end="", flush=True)
    model, cfg = load_ar_model(args.model, tokenizer_arg, device, hf=args.hf)
    print(f"\r{ok('Model loaded.')}          ")
    try:
        print(kv("Config", cfg))
    except:
        pass
    print()

    gen_config = GenerationConfig(
        max_new_tokens       = args.max_new_tokens,
        min_new_tokens       = args.min_new_tokens,
        temperature          = args.temperature,
        top_k                = args.top_k,
        top_p                = args.top_p,
        repetition_penalty   = args.repetition_penalty,
        no_repeat_ngram_size = args.no_repeat_ngram_size,
        do_sample            = not args.no_sample,
        stream               = not args.no_stream,
        show_perplexity      = not args.no_perplexity,
        show_timing          = not args.no_timing,
    )
    model.attach_tokenizer(tokenizer_arg)
    print("RICORDARSI DI SISTEMARE TOKENIZER_ARG")
    InteractiveSession(model, tokenizer_arg, gen_config, device).run()


if __name__ == "__main__":
    main()
