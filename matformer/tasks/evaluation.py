# matformer/tasks/evaluation.py
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Union
import torch
import torch.nn.functional as F
from matformer.matformer_registry import registry
from matformer.dataclasses.tensors_dataclasses import TensorDC
from matformer.tasks.task import TaskConfig, MatformerTask, per_position, surface, top_k
import math
@registry.register_dataclass("task", "evaluation")
@dataclass
class EvaluationTaskConfig(TaskConfig):
    learning_signal: Optional[str] = "autoregressive"
    # Summary-related configs
    num_classes: int = None                     # required if compute_confusion_matrix or compute_perplexity_by_token
    compute_confusion_matrix: bool = False      # O(num_classes^2) memory
    compute_perplexity_by_token: bool = False   # O(num_classes) memory, grouped by target id
    topk: Union[bool,tuple] = (5,)   
    # Single batch related configs 
    save_logits: bool = False 
    save_logprobs: bool = False
    save_argmax: bool = True
    limit_to_classes: Optional[list] = None     # Usage [27, 42, 9] to restrict saved outputs
    logit_diff_classes: Optional[list] = None   # Usage [idx1, idx2] to track specific logit margins
    use_tokenizer: bool = False
    topk_present: int = 2                       # candidates kept per prediction (interactive):
                                                # 0 = none, -1 = the whole class set
    save_probs: bool = False                    # keep the full distribution: task.probs answers
                                                # any position, not only the kept candidates

@registry.register("task", "evaluation", "default")
class EvaluationTask(MatformerTask):
    def begin_task(self):
        super().begin_task()
        if self.config.learning_signal:
            self.learning_signal = registry.create("learning_signal", self.config.learning_signal, wrapper=self.model)
        self.reset()

    def reset(self):
        self._state = {}


    def _state_for(self, name):
        if name not in self._state:
            s = {
                "nll_sum": 0.0,             
                "token_count": 0.0,         
                "correct_count": 0.0,      
                "confidence_sum": 0.0,      
                "entropy_sum": 0.0,         
                "pred_count": 0.0,          
                "brier_sum": 0.0,           
                "token_nll_sum": None,      
                "token_count_by_id": None,  
                "confusion": None,          
                "topk_correct": {},         # filled per top-k actually computed (see process_batch)
            }
            self._state[name] = s
        return self._state[name]

    @property
    def id2label(self):
        """Class id -> class name, from the model config or from the dataset it was trained on.
        None when the head has no named classes (a language model, usually)."""
        l2id = getattr(self.model.config, "label2id", None) or getattr(self.config.data, "label2id", None)
        return {int(v): k for k, v in l2id.items()} if l2id else None

    def input_tokens(self, batch):
        """What to show on the left of a per-token prediction, and which row of the predictions
        each shown item corresponds to. A token-classification head labels the first subword of
        every word and nothing of the rest, so when the input was split into words the words
        themselves are what deserves a row."""
        tokenizer = getattr(self.model, 'tokenizer', None)
        if tokenizer is None:
            return None, None
        words, word_ids = batch.attributes.get('words'), batch.attributes.get('word_ids')
        if words and word_ids and len(words) == len(word_ids):
            rows = [[(word, start) for word, start in zip(vocabulary, self._word_starts(ids))]
                    for vocabulary, ids in zip(words, word_ids)]
            return [[word for word, _ in row] for row in rows], [[start for _, start in row] for row in rows]
        pieces = [tokenizer.tokens(row.flatten().tolist()) for row in batch.pad().detach()]
        return pieces, [[range(len(pieces[row]))] and list(range(len(pieces[row]))) for row in range(len(pieces))]

    @staticmethod
    def _word_starts(word_ids):
        """Position of the first subword of every word: where the label of that word lives."""
        previous, starts = None, []
        for position, word in enumerate(word_ids):
            if word is not None and word != previous:
                starts.append(position)
            previous = word
        return starts

    def class_name(self, index):
        labels = self.id2label
        if labels:
            return labels.get(index, str(index))
        if self.config.use_tokenizer:
            return self.model.tokenizer.decode([index])
        return str(index)

    @torch.no_grad()
    def process_batch(self, batch, interactive=None) -> None:
        # interactive defaults to how the task is being driven: called one batch at a time, the
        # caller wants the per-item outputs, even when saving them was not configured.
        if interactive is None:
            interactive = self.interactive
        if self.config.learning_signal:
            input_sequence, metadata = self.learning_signal.prepare_inputs(batch)
            predictions = self.model(input_sequence, **self.learning_signal.model_kwargs(metadata))
            preds, targets = self.learning_signal.select_targets(predictions, batch, metadata)
        else:
            preds = self.model(batch)
            targets = None


        if isinstance(preds, dict):
            # multiprobe: {address: logits}. targets from multiprobe_ls is a
            # tensor (batch.attributes['labels']), but a dict is handled too.
            tokens, positions = self.input_tokens(batch) if interactive else (None, None)
            data_out = {}
            for name, preds_i in preds.items():
                targets_i = targets[name] if isinstance(targets, dict) else targets
                data_out[name] = self._process_single(preds_i, targets_i, name, interactive, tokens, positions)
            if any(v is not None for v in data_out.values()):
                return batch, data_out
            return batch, None

        # single-head path 
        tokens, positions = self.input_tokens(batch) if interactive else (None, None)
        data = self._process_single(preds, targets, None, interactive, tokens, positions)
        if data is not None:
            return batch, data
        return batch, None


    def _process_single(self, preds, targets, name, interactive, tokens=None, positions=None):
        s = self._state_for(name)


        if targets is not None:
            targets = targets.to(preds.device)

        log_probs = F.log_softmax(preds.float(), dim=-1)   # fp32: bf16 log_softmax saturates at exactly
        probs = log_probs.exp()                            # log p = 0, hiding resolution from nll/codelength
        predicted_ids = preds.argmax(-1)

        s["confidence_sum"] += probs.max(-1).values.sum().item()          
        s["entropy_sum"] += (-(probs * log_probs).sum(-1)).sum().item()  
        s["pred_count"] += preds.shape[:-1].numel()                       

        if targets is not None:
            nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) 
            s["nll_sum"] += nll.sum().item()                               
            s["token_count"] += targets.numel()                            
            s["correct_count"] += (predicted_ids == targets).sum().item()  
            s["brier_sum"] += (1 - 2 * probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) + (probs ** 2).sum(-1)).sum().item()  

            if self.config.topk:
                for k in self.config.topk:
                    if k > preds.shape[-1]:      # top-k wider than the number of classes is meaningless
                        continue
                    s["topk_correct"].setdefault(k, 0.0)
                    s["topk_correct"][k] += (preds.topk(k, dim=-1).indices == targets.unsqueeze(-1)).any(-1).sum().item()  

        if self.config.compute_perplexity_by_token:
            n = self.config.num_classes
            if s["token_nll_sum"] is None:                                 
                s["token_nll_sum"] = torch.zeros(n, device=targets.device, dtype=torch.float32)
                s["token_count_by_id"] = torch.zeros(n, device=targets.device, dtype=torch.float32)

            s["token_nll_sum"].scatter_add_(0, targets, nll.to(torch.float32))                            
            s["token_count_by_id"].scatter_add_(0, targets, torch.ones_like(nll, dtype=torch.float32))     
        if self.config.compute_confusion_matrix:
            n = self.config.num_classes
            if s["confusion"] is None:                                
                s["confusion"] = torch.zeros(n, n, device=targets.device)
            s["confusion"] += torch.bincount(targets * n + predicted_ids, minlength=n * n).view(n, n).float()  
        if getattr(self.config, "save_batch", False) or interactive:
            data = {"targets": targets, "accuracy": (predicted_ids == targets).float()} if targets is not None else {}
            cls_idx = torch.tensor(self.config.limit_to_classes, device=preds.device) if self.config.limit_to_classes else None

            if self.config.save_argmax:
                data["argmax"] = predicted_ids
                if cls_idx is not None:
                    data["selection_argmax"] = cls_idx[preds.index_select(-1, cls_idx).argmax(-1)]

            if self.config.save_logits:
                data["logits"] = preds.index_select(-1, cls_idx) if cls_idx is not None else preds

            if self.config.save_logprobs:
                data["logprobs"] = log_probs.index_select(-1, cls_idx) if cls_idx is not None else log_probs

            if self.config.save_probs:
                data["probs"] = probs.index_select(-1, cls_idx) if cls_idx is not None else probs

            if interactive and self.config.topk_present:
                k = top_k(self.config.topk_present, probs.shape[-1])
                values, indices = probs.topk(k, dim=-1)
                data["topk_ids"], data["topk_probs"] = indices, values
                if tokens is not None:
                    data["tokens"], data["positions"] = tokens, positions

            if getattr(self.config, "logit_diff_classes", None) and len(self.config.logit_diff_classes) == 2:
                data["logit_diff"] = preds[..., self.config.logit_diff_classes[0]] - preds[..., self.config.logit_diff_classes[1]]

            i2l = self.id2label
            if i2l or getattr(self.config, "use_tokenizer", False):
                fmt = (lambda x: i2l.get(x, str(x))) if i2l else (lambda x: self.model.tokenizer.decode([x]))
                for k in [x for x in ["targets", "argmax", "selection_argmax"] if x in data]:
                    lst = data[k].tolist()
                    data[f"{k}_labels"] = [[fmt(i) for i in row] for row in lst] if data[k].ndim > 1 else [fmt(i) for i in lst]

            return data
        return None

    @property
    def token_level(self) -> bool:
        """True when the head predicts one class per token rather than one per sequence."""
        return getattr(self.model.config, 'scope', None) == 'token' \
            or 'token' in str(self.config.learning_signal)

    @property
    def word_level(self) -> bool:
        """True when the input should be split into words before tokenizing, the way the
        dataset it was trained on was: labelling one token at a time only makes sense if the
        word boundaries the model saw at training time are reproduced."""
        return self.token_level

    def present(self, rows) -> Optional[str]:
        """Read the predictions: a token/class table for sequence labelling, the ranked classes
        for a whole-sequence decision. 'full' leaves the raw result to the console."""
        show = self.config.presentation
        if show == 'full' or not rows or not isinstance(rows[0], dict):
            return None
        if 'topk_ids' not in rows[0] and 'argmax' not in rows[0]:
            return None
        blocks = [self._present_row(row, show) for row in rows]
        return '\n'.join(block for block in blocks if block)

    def _present_row(self, row, show):
        if 'topk_ids' not in row:               # detail knob off: the decision, without the odds
            argmax = per_position(row['argmax'])
            labels = [self.class_name(int(one[0])) for one in argmax.tolist()] if argmax.dim() > 1 \
                else [self.class_name(int(argmax))]
            tokens = row.get('tokens') or [str(i + 1) for i in range(len(labels))]
            return self._present_table([surface(t) for t in tokens], [[name] for name in labels],
                                       [[1.0] for _ in labels], show)
        ids, probs = per_position(row['topk_ids']), per_position(row['topk_probs'])
        if self.token_level and row.get('tokens'):
            keeps = row['positions']
            return self._present_table(row['tokens'], ids[keeps].tolist() if keeps else ids.tolist(),
                                       probs[keeps].tolist() if keeps else probs.tolist(), show)
        if self.token_level:
            return self._present_table([str(i) for i in range(ids.shape[0])], ids.tolist(),
                                       probs.tolist(), show)
        return '  '.join(f"{self.class_name(cls)} {p:.1%}" if show == 'probs' else self.class_name(cls)
                         for cls, p in zip(ids[0].tolist(), probs[0].tolist()))

    def _present_table(self, tokens, ids, probs, show):
        names = [[self.class_name(cls) for cls in row] for row in ids]
        tokens = [surface(token) for token in tokens]
        width = max([len(token) for token in tokens] + [5])
        if show == 'argmax':
            return '\n'.join(f"  {token:<{width}}  {name[0]}" for token, name in zip(tokens, names))
        label_width = max(len(name[0]) for name in names)
        lines = [f"  {'token':<{width}}  {'label':<{label_width}}  {'p':>7}"]
        for token, candidates, ps in zip(tokens, names, probs):
            runner_up = f"   ({candidates[1]} {ps[1]:.0%})" if len(candidates) > 1 and ps[1] >= 0.05 else ''
            lines.append(f"  {token:<{width}}  {candidates[0]:<{label_width}}  {ps[0]:6.1%}{runner_up}")
        return '\n'.join(lines)

    def compute(self) -> dict:
        names = list(self._state.keys())
        multi = any(n is not None for n in names)
        result = {}
        for name in names:
            metrics = self._compute_metrics(self._state[name])
            if multi:
                prefix = f"{name}/" if name is not None else ""
                result.update({f"{prefix}{k}": v for k, v in metrics.items()})
            else:
                result.update(metrics)
        return result


    def _compute_metrics(self, s) -> dict:
        denom_preds = max(s["pred_count"], 1.0)                            

        metrics = {
            "mean_confidence": s["confidence_sum"] / denom_preds,          
            "mean_entropy": s["entropy_sum"] / denom_preds,                
        }

        if self.config.learning_signal and s["token_count"] > 0:           
            mean_nll = s["nll_sum"] / s["token_count"]                     
            metrics.update({
                "nll": mean_nll,
                "perplexity": torch.tensor(mean_nll).exp().item(),
                "accuracy": s["correct_count"] / s["token_count"],         
                "brier_score" : s["brier_sum"] / s["token_count"],
                "codelength_bits": s["nll_sum"] / math.log(2),   
                "num_tokens": s["token_count"]            
            })
            if self.config.num_classes:
                uniform_bits = s["token_count"] * math.log2(self.config.num_classes)
                metrics["uniform_codelength_bits"] = uniform_bits
                # A fully correct, fully confident block costs exactly 0 bits (bf16 log_softmax
                # saturates at log p = 0): compression is unbounded, and inf is not JSON-safe.
                metrics["compression"] = uniform_bits / metrics["codelength_bits"] if metrics["codelength_bits"] > 0 else None
            if s.get("topk_correct"):
                for k, v in s["topk_correct"].items():                     
                    metrics[f"acc@{k}"] = v / s["token_count"]             

            if s["token_nll_sum"] is not None:                             
                has_data = s["token_count_by_id"] > 0                     
                metrics["perplexity_by_token"] = (s["token_nll_sum"][has_data] / s["token_count_by_id"][has_data]).exp()  
                metrics["token_ids_for_perplexity_by_token"] = has_data.nonzero(as_tuple=True)[0]

            if s["confusion"] is not None:                                
                true_pos = s["confusion"].diag()                           
                precision = true_pos / s["confusion"].sum(0).clamp_min(1e-9)   
                recall = true_pos / s["confusion"].sum(1).clamp_min(1e-9)     
                f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
                metrics["macro_f1"] = f1.nanmean().item()
                metrics["micro_f1"] = metrics["accuracy"]
                metrics["macro_precision"] = precision.nanmean().item()
                metrics["macro_recall"] = recall.nanmean().item()
                metrics["precision_per_class"] = precision.nan_to_num().tolist()
                metrics["recall_per_class"] = recall.nan_to_num().tolist()
                metrics["f1_per_class"] = f1.nan_to_num().tolist()

        return metrics