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
        self.system_prompt = text or None
        return ("system prompt stored, but no chat template is applied yet: generation is "
                "unchanged" if text else "system prompt cleared")

    def set_chat_template(self, text) -> str:
        self.chat_template = (text or '').strip() or None
        return (f"chat template '{self.chat_template}' stored, but applying it to the context "
                "is not implemented yet" if self.chat_template else "chat template cleared")

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
        keeping = self.config.keep_context and self.interactive
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
        rows = ids[:, prompt_len:] if keeping else ids
        texts = [self.model.tokenizer.decode(row.tolist()) for row in rows]
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
        return {'generated': text, 'tokens': self.model.tokenizer.tokens(row.tolist()),
                'topk_ids': indices, 'topk_probs': values}

    def present(self, rows) -> Optional[str]:
        if self.config.presentation == 'full' or not rows:
            return None
        if not isinstance(rows[0], dict):
            return '\n'.join(str(row) for row in rows)
        return rows[0]['generated']
