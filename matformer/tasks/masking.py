# matformer/tasks/masking.py
from dataclasses import dataclass
from typing import Optional

import torch

from matformer.dataclasses.tensors_dataclasses import TensorDC
from matformer.matformer_registry import registry
from matformer.tasks.evaluation import EvaluationTask, EvaluationTaskConfig
from matformer.tasks.task import per_position, surface
from matformer.utils.masked_models import Maskerator


@registry.register_dataclass("task", "masking")
@dataclass
class MaskingTaskConfig(EvaluationTaskConfig):
    learning_signal: Optional[str] = None
    masking_ratio: float = 0.15
    random_seed: Optional[int] = None
    bidirectional: bool = True                 # attend in both directions while restoring
    use_tokenizer: bool = True                 # predictions are vocabulary entries, not classes
    topk_present: int = 4                      # candidates per hole: 0 = none, -1 = whole vocabulary


@registry.register("task", "masking", "default")
class MaskingTask(EvaluationTask):
    """Cloze test: the training Maskerator hides a fraction of the tokens, the model restores
    them, and the score is the accuracy on the hidden positions only. Driven over a dataset it
    accumulates exactly the same metrics as the evaluation task; driven interactively it also
    shows what the model put in every hole, with the probability it gave it."""

    def begin_task(self):
        tokenizer = self.model.tokenizer.config
        if tokenizer.mask_token_id is None:
            raise ValueError("The masking task needs a tokenizer with a mask token.")
        self.maskerator = Maskerator(mask_token=tokenizer.mask_token_id,
                                     substitution_rate=self.config.masking_ratio,
                                     pad_token_id=tokenizer.pad_token_id or 0,
                                     vocab_size=self.model.config.vocab_size,
                                     random_seed=self.config.random_seed)
        self.model_kwargs = {'is_causal': False} if self.config.bidirectional else {}
        super().begin_task()

    @torch.no_grad()
    def process_batch(self, batch):
        masked, ratio = self.maskerator(batch)
        cloze = masked.following_attributes['cloze_mask'].reshape(-1).bool()
        if not bool(cloze.any()):
            return batch, None
        predictions = self.model(masked, **self.model_kwargs)
        preds = predictions.tensor if isinstance(predictions, TensorDC) else predictions
        preds = preds.reshape(-1, preds.shape[-1])
        ids = (batch.tensor if isinstance(batch, TensorDC) else batch).reshape(-1)
        # Gold is a token id: comparable only when the head predicts over the vocabulary. A class
        # head (POS, NER) predicts labels, not tokens, so the original token is what was hidden,
        # not what the model can answer: scored as an open-ended restoration, never as a target.
        vocab_head = preds.shape[-1] >= self.model.config.vocab_size
        data = self._process_single(preds[cloze], ids[cloze] if vocab_head else None,
                                    None, self.interactive)
        if data is not None and self.interactive:
            tokens = self.model.tokenizer
            data.update(sentence=tokens.decode(ids.tolist()), holes=tokens.tokens(ids[cloze].tolist()),
                        length=int(ids.numel()), masking_ratio=ratio, vocab_head=vocab_head)
            if vocab_head:
                filled = ids.clone()
                filled[cloze] = data['argmax']
                data['restored'] = tokens.decode(filled.tolist())
        return batch, data

    def present(self, rows) -> Optional[str]:
        show = self.config.presentation
        if show == 'full' or not rows or not isinstance(rows[0], dict) or 'sentence' not in rows[0]:
            return None
        row = rows[0]
        got = row['argmax'].reshape(-1).tolist()
        head = f"  {row['sentence']}\n"
        if row.get('restored'):
            head += f"  {row['restored']}\n"
        head += (f"  {len(got)} of {row['length']} tokens masked ({row['masking_ratio']:.0%})")
        if row.get('targets') is not None:
            head += f", accuracy {row['accuracy'].mean():.1%}"
        return head if show == 'argmax' else head + '\n' + self._table(row, got)

    def _table(self, row, got):
        ids, probs = per_position(row['topk_ids']), per_position(row['topk_probs'])
        gold = row['targets'].reshape(-1).tolist() if row.get('targets') is not None else None
        names = [[self.class_name(c) for c in one] for one in ids.tolist()]
        holes = [surface(hole) for hole in row['holes']]
        width = max([max(len(n) for n in one) for one in names] + [len(str(h)) for h in holes] + [4])
        lines = [f"  {'was':<{width}}  {'restored':<{width}}  {'p':>7}  alternatives"]
        for i, hole in enumerate(holes):
            if i >= ids.shape[0]:
                break
            alternatives = '  '.join(f"{n} {p:.0%}" for n, p in zip(names[i][1:], probs[i][1:].tolist()))
            mark = '=' if gold and gold[i] == got[i] else (' ' if gold is None else '≠')
            lines.append(f"  {str(hole):<{width}}  {mark} {names[i][0]:<{width}}"
                         f"  {probs[i][0]:6.1%}  {alternatives}")
        return '\n'.join(lines)
