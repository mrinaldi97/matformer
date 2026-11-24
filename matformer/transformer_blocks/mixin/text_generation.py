import time
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor
def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = float("-inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Top-k and/or nucleus (top-p) filtering on a batch of logit distributions.

    Args:
        logits: (batch_size, vocab_size)
        top_k: keep only the top-k tokens; 0 = disabled.
        top_p: nucleus threshold; 1.0 = disabled.
        min_tokens_to_keep: always keep at least this many candidates per sample.
    Returns:
        logits with disallowed positions set to filter_value (in-place).
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        logits[logits < threshold] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_mask[..., :min_tokens_to_keep] = False
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        logits[sorted_mask.scatter(1, sorted_indices, sorted_mask)] = filter_value

    return logits


def calc_banned_ngram_tokens(
    input_ids: torch.LongTensor,
    num_hypos: int,
    no_repeat_ngram_size: int,
    cur_len: int,
) -> List[List[int]]:
    """Return, for each hypothesis, the tokens banned by the no-repeat-ngram constraint."""
    if cur_len < no_repeat_ngram_size:
        return [[] for _ in range(num_hypos)]

    # Build prefix → [next_token, …] maps for every hypothesis.
    ngram_dicts: List[dict] = [{} for _ in range(num_hypos)]
    for i in range(num_hypos):
        tokens = input_ids[i].tolist()
        d = ngram_dicts[i]
        for ngram in zip(*[tokens[j:] for j in range(no_repeat_ngram_size)]):
            prefix = tuple(ngram[:-1])
            d.setdefault(prefix, []).append(ngram[-1])

    def _banned(i: int) -> List[int]:
        prefix = tuple(
            input_ids[i, cur_len - (no_repeat_ngram_size - 1) : cur_len].tolist()
        )
        return ngram_dicts[i].get(prefix, [])

    return [_banned(i) for i in range(num_hypos)]


class GenerationMixin:

    def _get_logits(self, current_ids) -> torch.Tensor:
        """Run one forward pass and return logits (batch, seq_len, vocab_size)."""
        if isinstance(current_ids, torch.Tensor):
            current_ids = NormalTensor(tensor=current_ids)
        return self(current_ids).tensor

    def paged_cache(self, tokens, dtype=None):
        """The KV cache for this model, sized once for `tokens` of conversation."""
        return self.cache.kv_cache(tokens=tokens, dtype=dtype or next(self.parameters()).dtype,
                                   device=self.device)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        *,                                          
        max_new_tokens: int = 200,
        min_new_tokens: int = 0,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        output_scores: bool = False,
        stopping_criteria: Optional[Callable[[torch.LongTensor, Optional[list]], bool]] = None,
        stream_callback: Optional[Callable[[torch.LongTensor, int], None]] = None,
        return_timing: bool = False,
        use_cache: bool = False,
        kv_state=None,
    ) -> Union[torch.LongTensor, Tuple]:
        """Generate token sequences autoregressively.

        Args:
            input_ids: (batch, seq_len) prompt tensor, or None to start from BOS.
                TensorDC inputs are normalised automatically via .pad().tensor.
            max_new_tokens: Maximum new tokens to generate (prompt not counted).
            min_new_tokens: EOS is suppressed for this many steps.
            do_sample: True → multinomial sampling; False → greedy argmax.
            temperature: Scales logits before sampling (do_sample=True only).
            top_k: Restrict sampling to the top-k tokens (0 = disabled).
            top_p: Nucleus probability threshold (1.0 = disabled).
            repetition_penalty: > 1.0 penalises tokens already in the context.
            no_repeat_ngram_size: Forbid repeating any n-gram of this size.
            num_return_sequences: Sequences generated per input item.
            pad_token_id / eos_token_id / bos_token_id: Override config values.
            output_scores: Include raw pre-sampling logits in the return value.
            stopping_criteria: fn(current_ids, scores_so_far) → bool.
                Return True to stop generation early.
            stream_callback: fn(next_tokens: LongTensor[batch], step: int).
                Called immediately after each decoding step; use for streaming UIs.
            return_timing: Include tokens/second throughput in the return value.
            use_cache: Keep the keys and values of what was already computed in a paged KV cache
                (matformer.utils.kv_cache) and feed the model one token per step instead of the
                whole sequence. Needs an attention implementation that reads the cache, and
                prompts of the same length across the batch.
            kv_state: A state from a previous call, already holding the prefix: the conversation
                continues from it instead of being recomputed, and the state is left intact (and
                grown) for the next caller. Implies use_cache.

        Returns:
            current_ids  [always]          (batch * num_return_sequences, full_len)
            all_scores   [if output_scores] (batch, steps, vocab_size)
            tok_per_sec  [if return_timing] float
        """
        self.eval()

        tok_cfg = getattr(self.config, "tokenizer_config", None)

        def _resolve(name, override):
            if override is not None:
                return override
            return getattr(tok_cfg, name, None) if tok_cfg else None

        pad_token_id = _resolve("pad_token_id", pad_token_id) or 0
        eos_token_id = _resolve("eos_token_id", eos_token_id)
        bos_token_id = _resolve("bos_token_id", bos_token_id)

        if input_ids is None:
            if bos_token_id is None:
                raise ValueError("Provide either input_ids or bos_token_id.")
            input_ids = torch.full(
                (1, 1), bos_token_id, dtype=torch.long, device=self.device
            )
        else:
            if hasattr(input_ids, "pad"):           # TensorDC → PaddedTensor
                input_ids = input_ids.pad().pad_right().tensor
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(self.device)

        # repeat_interleave keeps batch items together: [a,a,b,b] not [a,b,a,b]
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)

        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]

        # Convert relative bounds → absolute sequence lengths for helpers
        max_length = prompt_len + max_new_tokens
        min_length = prompt_len + min_new_tokens

        current_ids = input_ids
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        all_scores: Optional[List[torch.Tensor]] = [] if output_scores else None
        t0 = time.perf_counter() if return_timing else None

        state, owned = kv_state, False
        if state is None and use_cache and max_new_tokens > 0:
            if len(set(input_ids.ne(pad_token_id).sum(-1).tolist() + [prompt_len])) > 1:
                raise ValueError("Paged generation cannot start from a padded batch: every sequence "
                                 "needs the same number of real tokens.")
            state = self.paged_cache(batch_size * (prompt_len + max_new_tokens)).new_state(batch_size)
            owned = True

        def forward(ids, q_lens, phase):
            if state is None:
                return self._get_logits(ids)
            state.prepare(q_lens, phase=phase)
            step = NormalTensor(tensor=ids, attributes={
                'kv_state': state, 'seqlen_offsets': state.lengths[0] - q_lens[0]})
            return self._get_logits(step)

        for step in range(max_new_tokens):
            cur_len = current_ids.shape[1]

            logits = forward(current_ids if state is None or step == 0 else current_ids[:, -1:],
                             [prompt_len] * batch_size if step == 0 else [1] * batch_size,
                             'prefill' if step == 0 else 'decode')
            next_token_logits = logits[:, -1, :]        # (B, V) — view into buffer

            # Store raw logits before any processing
            if output_scores:
                all_scores.append(next_token_logits.clone())

            # Repetition penalty, min-length EOS suppression, no-repeat-ngram
            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=current_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            if do_sample:
                if temperature != 1.0:
                    scores = scores / temperature
                scores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                probs = F.softmax(scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
            else:
                next_tokens = torch.argmax(scores, dim=-1)                        # (B,)

            # Pad already-finished sequences instead of extending them
            next_tokens = next_tokens.masked_fill(finished, pad_token_id)
            current_ids = torch.cat([current_ids, next_tokens.unsqueeze(1)], dim=1)

            # Stream: fire callback before updating finished so the caller
            # receives the actual token chosen, not the pad substitution.
            if stream_callback is not None:
                stream_callback(next_tokens, step)

            if eos_token_id is not None:
                finished = finished | (next_tokens == eos_token_id)

            if stopping_criteria is not None and stopping_criteria(current_ids, all_scores):
                break

            if finished.all():
                break

        extras = []
        if owned:
            state.release()
        if output_scores:
            extras.append(torch.stack(all_scores, dim=1))  # (B, steps, V)
        if return_timing:
            steps_done = current_ids.shape[1] - prompt_len
            tok_per_sec = (steps_done * batch_size) / (time.perf_counter() - t0)
            extras.append(tok_per_sec)

        return (current_ids, *extras) if extras else current_ids


    def enforce_repetition_penalty_(
        self,
        lprobs: torch.Tensor,
        batch_size: int,
        num_beams: int,
        prev_output_tokens: torch.LongTensor,
        repetition_penalty: float,
    ) -> None:
        """In-place repetition penalty (CTRL, Keskar et al. 2019).

        Tokens that appeared before are downscaled (positive score) or
        upscaled in magnitude (negative score, i.e. log-prob).
        """
        for i in range(batch_size * num_beams):
            for token in set(prev_output_tokens[i].tolist()):
                if lprobs[i, token] < 0:
                    lprobs[i, token] *= repetition_penalty
                else:
                    lprobs[i, token] /= repetition_penalty

    def postprocess_next_token_scores(
        self,
        scores: torch.Tensor,
        input_ids: torch.LongTensor,
        no_repeat_ngram_size: int,
        cur_len: int,
        min_length: int,
        max_length: int,
        eos_token_id: Optional[int],
        repetition_penalty: float,
        batch_size: int,
        num_beams: int,
    ) -> torch.Tensor:
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores, batch_size, num_beams, input_ids, repetition_penalty
            )

        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = float("-inf")

        if no_repeat_ngram_size > 0:
            banned = calc_banned_ngram_tokens(
                input_ids, batch_size * num_beams, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned):
                if banned_tokens:
                    scores[i, banned_tokens] = float("-inf")

        return scores
