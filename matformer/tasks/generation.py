# matformer/tasks/generation.py
import time
from typing import Optional, ClassVar
import torch
from matformer.tasks.task import MatformerTask, TaskConfig, top_k
from matformer.matformer_registry import registry
from matformer.dataclasses.tensors_dataclasses import PaddedTensor
from dataclasses import dataclass


@registry.register_dataclass("task", "generation", default=True)
@dataclass
class GenerationTaskConfig(TaskConfig):
    max_new_tokens:       int           = 200
    min_new_tokens:       int           = 0
    temperature:          float         = 1.0
    top_k:                int           = 0
    top_p:                float         = 0.9
    repetition_penalty:   float         = 1.0
    no_repeat_ngram_size: int           = 0
    do_sample:            bool          = True
    prompt:               Optional[str] = None
    # Chat template: off by default so plain continuation keeps working unchanged. When on and the
    # model has a template, an interactive request is one turn of a conversation and a batch row is
    # its chat-formatted prompt. The template owns bos/eos, so no boundary token is added twice.
    use_chat_template:    bool          = False
    system_prompt:        Optional[str] = None   # prepended as a system message (interactive + batch)
    add_generation_prompt: bool         = True   # end the prompt with the assistant header
    continue_final_message: bool        = False  # keep the last turn open so the model continues it
    chat_options:         Optional[dict] = None  # template toggles, ex. {"enable_thinking": false}
    stream:               bool          = True   # print tokens as they are produced (interactive)
    keep_context:         bool          = True   # interactive: every request continues the previous ones
    max_context_tokens:   Optional[int] = None   # left-trim of the kept context, default: the model window
    use_cache:            bool          = False  # paged KV cache: needs an attention that reads it
                                                 # (--attention flash_kv / sdpa_kv / flashinfer)
    max_kv_tokens:        Optional[int] = None   # conversation budget the cache is reserved for
    topk_present:         int           = 0      # interactive: keep the top-k candidates of every
                                                 # generated token, so a UI can hover one and see
                                                 # what the model had instead of it. 0 = plain text
                                                 # (fastest), -1 = the whole vocabulary

    _GENERATE_FIELDS: ClassVar[tuple] = (
        "max_new_tokens", "min_new_tokens", "temperature", "top_k",
        "top_p", "repetition_penalty", "no_repeat_ngram_size", "do_sample",
    )

    def generate_kwargs(self) -> dict:
        return {k: getattr(self, k) for k in self._GENERATE_FIELDS}

@registry.register("task", "generation", "default")
class GenerationTask(MatformerTask):
    """Autoregressive generation: one decoded text per item of the batch.

    Streaming is enabled by passing stream_callback to the task (see MatformerTask.__call__):
    it receives (next_tokens, step) after every decoding step, so a UI can print tokens as
    they are produced.

    With config.keep_context the task owns a running context: what was asked and what was
    answered stays in front of the next prompt, and only the newly produced text is returned.
    clear() throws it away. The chat plumbing (system prompt, template) is stored but not
    applied yet: applying it needs the tokenizer's template, which is the next step.
    """

    def begin_task(self):
        super().begin_task()
        self.reset()

    def reset(self):
        self.context = None
        self.kv_state = None
        self.messages = []          # committed chat turns (chat template mode only)
        self._pending_user = None   # the user turn whose reply is being generated
        self.rendered_prompt = None  # the exact templated prompt last sent (shown/hidden by the GUI)

    @property
    def context_tokens(self) -> int:
        return 0 if self.context is None else int(self.context.numel())

    def window(self) -> int:
        return self.config.max_context_tokens \
            or getattr(self.model.config, 'max_position_embeddings', 0) - self.config.max_new_tokens

    def show_context(self) -> str:
        if not self.config.keep_context:
            return "context is off: every request starts fresh ('set keep_context true' to keep it)"
        cached = f", {self.kv_state.lengths[0]} of them cached" if self.kv_state is not None else ''
        return f"{self.context_tokens} tokens in context{cached}"

    def clear(self) -> str:
        self.context, self.status = None, ""
        if self.kv_state is not None:
            self.kv_state.release()
            self.kv_state = None
        return "context cleared" + (" (keep_context is off, so nothing was kept)"
                                    if not self.config.keep_context else "")

    def kv_budget(self) -> int:
        """Tokens of conversation the cache is reserved for: the window the context will ever
        need, so that no turn has to reallocate it behind the running state."""
        window = self.config.max_kv_tokens or self.config.max_context_tokens \
            or getattr(self.model.config, 'max_position_embeddings', 0)
        return int(window) + self.config.max_new_tokens

    def set_system_prompt(self, text) -> str:
        """Set/clear the system message prepended to a conversation (interactive and batch)."""
        self.system_prompt = text or None
        return f"system prompt {'cleared' if not text else 'set'}"

    def set_chat_template(self, text) -> str:
        """off/none clears use of the template; on/default uses the model's own; anything else is an
        inline Jinja source or a .jinja path, installed onto the model as an override."""
        value = (text or '').strip()
        if value.lower() in ('', 'off', 'none', 'false'):
            self.config.use_chat_template = False
            return "chat template off: generation stays plain"
        if value.lower() in ('on', 'default', 'true'):
            self.config.use_chat_template = True
            has = bool(getattr(self.model, 'chat_template', None))
            return f"chat template on (model template: {'available' if has else 'missing'})"
        from pathlib import Path
        source = Path(value).read_text(encoding='utf-8') if Path(value).is_file() else value
        self.model.set_chat_template(source)
        self.config.use_chat_template = True
        return "chat template override applied"

    # --- chat template turns ----------------------------------------------------------------
    @property
    def chat_mode(self) -> bool:
        """True when a conversation is rendered through the model's template this run."""
        return bool(self.config.use_chat_template and getattr(self.model, 'chat_template', None))

    def _system_prompt(self):
        return getattr(self, 'system_prompt', None) or self.config.system_prompt

    def _chat_messages(self, user_text=None):
        """The conversation so far as {role, content} dicts: optional system prompt, committed
        turns, and (when given) the new user turn not yet committed."""
        messages = []
        if self._system_prompt():
            messages.append({"role": "system", "content": self._system_prompt()})
        messages.extend(self.messages)
        if user_text is not None:
            messages.append({"role": "user", "content": str(user_text)})
        return messages

    def build_chat_batch(self, texts):
        """One interactive turn: history + this user text rendered by the template. The text is
        remembered so process_batch can commit the exchange once the reply is generated."""
        self.ensure_began()   # reset() initialises the chat buffers on first use
        template = self.model.chat_template
        messages = self._chat_messages(texts[0])
        options = self.config.chat_options or None
        self.rendered_prompt = template.render(
            messages, add_generation_prompt=self.config.add_generation_prompt,
            continue_final_message=self.config.continue_final_message, options=options)
        ids = template.encode(
            messages, self.model.tokenizer.encode_tokens,
            add_generation_prompt=self.config.add_generation_prompt,
            continue_final_message=self.config.continue_final_message, options=options)
        self._pending_user = str(texts[0])
        pad = getattr(self.model.tokenizer.config, 'pad_token_id', None) or 0
        return PaddedTensor.from_rows([torch.tensor(ids, dtype=torch.long)], pad_value=pad)

    def _with_context(self, batch):
        """Rebuild the batch with the kept context in front of every row."""
        pad = getattr(self.model.tokenizer.config, 'pad_token_id', None) or 0
        rows = [torch.cat([self.context, row]) for row in batch.pad().detach()]
        return PaddedTensor.from_rows(rows, pad_value=pad)

    def _continue(self, batch):
        """The turn to feed when the conversation is cached: only what is new, since the state
        already holds everything before it. A state that no longer matches the context (first
        turn, context trimmed away, cache cleared) starts over with the whole thing."""
        rows = batch.pad().detach()
        if self.kv_state is not None and self.context is not None \
                and self.kv_state.lengths[0] == int(self.context.numel()) \
                and len(rows) == len(self.kv_state.lengths):
            state = self.kv_state
        else:
            if self.kv_state is not None:
                self.kv_state.release()    # that conversation is over: give its pages back
            pad = getattr(self.model.tokenizer.config, 'pad_token_id', None) or 0
            rows = [torch.cat([self.context, row]) if self.context is not None else row
                    for row in rows]
            batch = PaddedTensor.from_rows(rows, pad_value=pad)
            state = self.model.model.paged_cache(self.kv_budget()).new_state(len(rows))
        return batch, state

    def process_batch(self, batch):
        # In chat mode the conversation lives in self.messages, so the token-context accumulator is
        # off: each turn is re-rendered by the template. Plain mode keeps the old behaviour.
        keeping = self.config.keep_context and self.interactive and not self.chat_mode
        caching = keeping and self.config.use_cache
        if caching:
            batch, self.kv_state = self._continue(batch)
        elif keeping and self.context is not None:
            batch = self._with_context(batch)
        prompt_len = int((~batch.pad().padding_mask).sum(-1).max())
        first = []
        callback = self.stream_callback

        def stream(next_tokens, step):
            if not first:
                first.append(time.perf_counter())
            if callback is not None:
                callback(next_tokens, step)

        started = time.perf_counter()
        detailed = self.interactive and self.config.topk_present != 0
        generated = self.model.model.generate(
            input_ids=batch,
            stream_callback=stream,
            output_scores=detailed, return_timing=False, use_cache=self.config.use_cache,
            kv_state=self.kv_state,
            **self.config.generate_kwargs(),
        )
        ids, scores = generated if detailed else (generated, None)
        new_tokens = int(ids.shape[-1] - prompt_len)
        ended = time.perf_counter()
        self.metrics = {'in_tokens': prompt_len, 'new_tokens': new_tokens,
                        'prefill_tokens': prompt_len, 'prefill_seconds': max(first[0] - started, 1e-9)
                        if first else ended - started}
        if new_tokens and first:
            self.metrics.update(decode_tokens=new_tokens, decode_seconds=max(ended - first[0], 1e-9))
        if keeping:
            whole = torch.cat([self.context, ids[0]]) if self.context is not None else ids[0]
            window = self.window()
            self.context = whole[-window:] if window > 0 and whole.numel() > window else whole
            self.status = f"ctx {self.context_tokens} tok"
        rows = ids[:, prompt_len:] if (keeping or self.chat_mode) else ids
        # Never show the stop token itself: trim trailing eos/pad ids from the answer before decoding.
        eos = getattr(self.model.tokenizer.config, 'eos_token_id', None)
        pad = getattr(self.model.tokenizer.config, 'pad_token_id', None)
        stops = {token for token in (eos, pad) if token is not None}

        def _visible(row):
            seq = row.tolist()
            while seq and seq[-1] in stops:
                seq.pop()
            return seq
        texts = [self.model.tokenizer.decode(_visible(row)) for row in rows]
        if self.chat_mode and self.interactive and self._pending_user is not None:
            # Commit the exchange so the next turn sees it. Only the assistant scoring is kept out;
            # the user text and the reply are what a conversation is made of.
            self.messages.append({"role": "user", "content": self._pending_user})
            self.messages.append({"role": "assistant", "content": texts[0] if texts else ""})
            self.status = f"chat: {len(self.messages)} messages"
            self._pending_user = None
        if detailed:
            return batch, [self._candidates(row, scores[i], texts[i])
                           for i, row in enumerate(ids[:, prompt_len:])]
        return batch, texts

    def _candidates(self, row, step_logits, text):
        """The text the model wrote, next to what it could have written: the surface form of every
        generated token and the probability of its top-k. Kept hidden until a UI asks for it."""
        k = top_k(self.config.topk_present, step_logits.shape[-1])
        probs = torch.softmax(step_logits.float(), dim=-1)
        values, indices = probs.topk(k, dim=-1)
        ids = row.tolist()
        pieces, previous = [], ''
        for end in range(1, len(ids) + 1):
            whole = self.model.tokenizer.decode(ids[:end])
            pieces.append(whole[len(previous):])
            previous = whole
        return {'generated': text, 'tokens': self.model.tokenizer.tokens(ids),
                'pieces': pieces, 'topk_ids': indices, 'topk_probs': values}

    def present(self, rows) -> Optional[str]:
        if self.config.presentation == 'full' or not rows:
            return None
        if not isinstance(rows[0], dict):
            return '\n'.join(str(row) for row in rows)
        return rows[0]['generated']
