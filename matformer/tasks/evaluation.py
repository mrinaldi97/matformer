# matformer/tasks/evaluation.py
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Union
import torch
import torch.nn.functional as F
from matformer.matformer_registry import registry
from matformer.tasks.task import TaskConfig, MatformerTask
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
            }
            if self.config.topk:
                s["topk_correct"] = {k: 0.0 for k in self.config.topk}  
            self._state[name] = s
        return self._state[name]

    @torch.no_grad()
    def process_batch(self, batch, interactive=False) -> None:
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
            data_out = {}
            for name, preds_i in preds.items():
                targets_i = targets[name] if isinstance(targets, dict) else targets
                data_out[name] = self._process_single(preds_i, targets_i, name, interactive)
            if any(v is not None for v in data_out.values()):
                return batch, data_out
            return batch, None

        # single-head path 
        data = self._process_single(preds, targets, None, interactive)
        if data is not None:
            return batch, data
        return batch, None


    def _process_single(self, preds, targets, name, interactive):
        s = self._state_for(name)


        if targets is not None:
            targets = targets.to(preds.device)

        log_probs = F.log_softmax(preds.float(), dim=-1)   # fp32: bf16 log_softmax saturates at exactly
        probs = log_probs.exp()                            # log p = 0, hiding resolution from nll/codelength
        predicted_ids = preds.argmax(-1)

        s["confidence_sum"] += probs.max(-1).values.sum().item()          
        s["entropy_sum"] += (-(probs * log_probs).sum(-1)).sum().item()  
        s["pred_count"] += preds.shape[:-1].numel()                       

        if self.config.learning_signal:
            nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) 
            s["nll_sum"] += nll.sum().item()                               
            s["token_count"] += targets.numel()                            
            s["correct_count"] += (predicted_ids == targets).sum().item()  
            s["brier_sum"] += (1 - 2 * probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) + (probs ** 2).sum(-1)).sum().item()  

            if self.config.topk:
                for k in self.config.topk:
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
            data = {"targets": targets, "accuracy": (predicted_ids == targets).float()} if self.config.learning_signal else {}
            cls_idx = torch.tensor(self.config.limit_to_classes, device=preds.device) if self.config.limit_to_classes else None

            if self.config.save_argmax:
                data["argmax"] = predicted_ids
                if cls_idx is not None:
                    data["selection_argmax"] = cls_idx[preds.index_select(-1, cls_idx).argmax(-1)]

            if self.config.save_logits:
                data["logits"] = preds.index_select(-1, cls_idx) if cls_idx is not None else preds

            if self.config.save_logprobs:
                data["logprobs"] = log_probs.index_select(-1, cls_idx) if cls_idx is not None else log_probs

            if getattr(self.config, "logit_diff_classes", None) and len(self.config.logit_diff_classes) == 2:
                data["logit_diff"] = preds[..., self.config.logit_diff_classes[0]] - preds[..., self.config.logit_diff_classes[1]]

            l2id = getattr(self.model.config, "label2id", None)
            i2l = {v: k for k, v in l2id.items()} if l2id else None

            if i2l or getattr(self.config, "use_tokenizer", False):
                fmt = (lambda x: i2l.get(x, str(x))) if i2l else (lambda x: self.model.tokenizer.decode([x]))
                for k in [x for x in ["targets", "argmax", "selection_argmax"] if x in data]:
                    lst = data[k].tolist()
                    data[f"{k}_labels"] = [[fmt(i) for i in row] for row in lst] if data[k].ndim > 1 else [fmt(i) for i in lst]

            return data
        return None

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
            if self.config.topk:
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