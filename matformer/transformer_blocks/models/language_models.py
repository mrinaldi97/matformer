"""
File: matformer/transformer_blocks/models/language_models.py
"""
#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#Matformer
from matformer.matformer_module import MatformerModule
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor, ModuleWrapper
from matformer.model_config import ModelConfig  
from matformer.utils.matformer_cache import ensure_cache_and_registry, CachedStuff
from matformer.matformer_registry import registry
from matformer.utils.masked_models import Maskerator
from matformer.transformer_blocks import TransformerWithLMHead

#Other
from dataclasses import replace
from copy import deepcopy
from warnings import warn
from typing import Optional, List, Literal, Any



class BERTModel(TransformerWithLMHead):
    
    def init_training(self,config: ModelConfig,tokenizer=None,device=None,cache=None):
        # Training a BERT-Like model
        self.config=config
        self.cache = ensure_cache_and_registry(cache) 
        super().__init__()
        
        # 1. Init the maskerator
        self.init_maskerator()
        # 2. Define the loss function
        if self.has_fused_loss:
            pass
        else:
            pass
   
    def init_maskerator(self, masking_ratio=None, random_seed=None):
        from matformer.utils.masked_models import Maskerator
        # Maskerator setup
        if masking_ratio is not None: # The bert model was not istantiated with the config, masking ratio parameter has to be passed as argument (case for inference)
             assert masking_ratio is not None
             self.masking_ratio=masking_ratio
             self.maskerator=Maskerator(mask_token=self.config.mask_token_id,
                    substitution_rate=masking_ratio,
                    random_seed=random_seed,
                    pad_token_id=self.config.pad_token_id,
                    vocab_size=self.config.vocab_size)
             print(f"Masking ratio: {self.masking_ratio}")
        else: # We get the maskerator settings from the config
            if masking_ratio is not None:
                self.masking_ratio=masking_ratio
            else:
                try:
                    self.masking_ratio=self.config.masked_substitution_rate
                except:
                    self.masking_ratio=0.15
            try:
                try:
                    cloze_prob = self.config.get("cloze_prob", 1.0)
                    random_prob = self.config.get("random_prob", None)
                    same_prob = self.config.get("same_prob", None)
                    vocab_size = self.config.get("vocab_size", None)
                except:
                    cloze_prob=1.0
                    random_prob=0.0
                    same_prob=0.0
                    try:
                        vocab_size=self.config.vocab_size
                    except:
                        vocab_size=None
                print(f"Setting up maskerator with {self.masking_ratio}% substitution rate.")
                self.maskerator=Maskerator(mask_token=self.config.mask_token_id,
                                           substitution_rate=self.masking_ratio,
                                           pad_token_id=self.config.pad_token_id,
                                           cloze_prob=cloze_prob,
                                           random_prob=random_prob,
                                           same_prob=same_prob,
                                           vocab_size=vocab_size)
            except:
                print("Maskerator not set up. Fine for Autoregressive model")     


    def init_classification_head(self, num_labels=2, pooling_type='cls',cache=None):
        """Initialize classification head. Compatible with HuggingFace Trainer."""
        self.cache = ensure_cache_and_registry(cache)                               
        self.classification_head = ModuleWrapper(self.cache.registry.create(
            "linear", "linear", in_features=config.hidden_size, out_features=num_labels
        ))
        self.pooling_type = pooling_type
        self.num_labels = num_labels
        
    def forward_classification(self, x, attention_mask=None):
        """Forward pass for sequence classification."""
        hidden_states = self.encoder(x)
        
        if self.pooling_type == 'cls':
            pooled = hidden_states.tensor[:, 0]
        elif self.pooling_type == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.tensor.size()).float()
                pooled = torch.sum(hidden_states.tensor * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            else:
                pooled = hidden_states.tensor.mean(dim=1)
        else:
            raise ValueError(f"pooling_type must be 'cls' or 'mean', got {self.pooling_type}")
            
        return self.classification_head(pooled).tensor        
    def inference_testing(self, input_text=None, masking_ratio=0.25,datatype=torch.bfloat16, tokens=None, recurrence_mask=None,random_seed=None):
        #assert (is input_text or is_tokens)
        if not hasattr(self,'maskerator') or masking_ratio!=self.masking_ratio:
            self.init_maskerator(masking_ratio, random_seed=random_seed)
        if not tokens:
            sequence = self.tokenizer.encode(input_text)
        else:
            sequence=tokens
        sequence = torch.tensor(sequence).to(self.device)
        masked_list, cloze_list, _ = self.maskerator(sequence)
        masked_list.to(self.device)
        if recurrence_mask is None:
            masked_sequence = NormalTensor(tensor=masked_list.unsqueeze(0))
        else:
            masked_sequence = NormalTensor(tensor=masked_list.unsqueeze(0),recurrence_mask=torch.tensor([recurrence_mask]).to(self.device))
        model_input=deepcopy(masked_sequence)
        with torch.no_grad():
            logits = self(model_input)        
        predictions = torch.argmax(logits.tensor, dim=-1)
        targets = sequence.squeeze()
        mask = cloze_list.squeeze().bool()
        correct = (predictions.squeeze()[mask] == targets[mask]).sum().item()
        total = mask.sum().item()
        if total==0:
            print("WARNING: No tokens were masked")
            return None,None,None
        accuracy = correct / total
        out_tokens=list()
        squeezed_tensor = masked_sequence.tensor.squeeze()
        if squeezed_tensor.dim() == 0:
            print("WARNING: sequence had length 1!")
            token_list = [squeezed_tensor.item()]
        else:
            token_list = squeezed_tensor.tolist()
        for i,token in enumerate(token_list):        
            if token != self.config.mask_token_id:
                out_tokens.append(self.tokenizer.decode(token))
            else:
                out_tokens.append(f"[ {self.tokenizer.decode(predictions.squeeze()[i])} ]")
        #ppl
        try:
            log_probs = F.log_softmax(logits.tensor, dim=-1)
            nll_loss = F.nll_loss(
                log_probs.squeeze(0)[mask],
                targets[mask],
                reduction='mean'
            )
            pseudo_perplexity = torch.exp(nll_loss).item()
        except:
            print("WARNING: It was impossible to compute pseudo perplexity. Very short sequence?")
            pseudo_perplexity=float("inf")
        return accuracy, out_tokens, pseudo_perplexity

class Autoregressive_Model(TransformerWithLMHead):
    def generate(
        self, 
        prompt=None, 
        input_ids=None, 
        max_length=100, 
        min_length=0,
        temperature=1.0, 
        top_k=0, 
        top_p=0.9,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        num_return_sequences=1,
        pad_token_id=None,
        eos_token_id=None,
        bos_token_id=None,
        stopping_criteria=None,
        output_scores=False,
        return_type='text'
    ):
        self.eval()
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
            if pad_token_id is None: #dirty
                pad_token_id=0
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if bos_token_id is None:
            bos_token_id = self.config.bos_token_id
        
        if input_ids is None:
            if prompt is None:
                current_ids = torch.tensor([[bos_token_id]], device=self.device).repeat(num_return_sequences, 1)
            else:
                assert isinstance(prompt, str), "Prompt expected as string"
                tokenizer = self.tokenizer
                prompt_ids = tokenizer.encode(text=prompt, add_bos=True, add_eos=False, add_special_tokens=False)
                current_ids = torch.tensor([prompt_ids], device=self.device).repeat(num_return_sequences, 1)
        else:
            if isinstance(input_ids, NormalTensor):
                current_ids = input_ids.tensor
            elif isinstance(input_ids, torch.Tensor):
                current_ids = input_ids
                if current_ids.dim() == 1:
                    current_ids = current_ids.unsqueeze(0)
            elif isinstance(input_ids, list):
                current_ids = torch.tensor([input_ids], device=self.device)
            else:
                raise TypeError(f"input_ids must be NormalTensor, torch.Tensor, or list, got {type(input_ids)}")
            
            if num_return_sequences > 1 and current_ids.shape[0] == 1:
                current_ids = current_ids.repeat(num_return_sequences, 1)
        
        current_ids = current_ids.to(self.device)
        batch_size = current_ids.shape[0]
        
        all_scores = [] if output_scores else None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for step in range(max_length):
            with torch.no_grad():
                outputs = self(NormalTensor(tensor=current_ids)).tensor
            
            next_token_logits = outputs[:, -1, :]
            
            if output_scores:
                all_scores.append(next_token_logits.clone())
            
            if repetition_penalty != 1.0:
                for batch_idx in range(batch_size):
                    for token_id in set(current_ids[batch_idx].tolist()):
                        if next_token_logits[batch_idx, token_id] < 0:
                            next_token_logits[batch_idx, token_id] *= repetition_penalty
                        else:
                            next_token_logits[batch_idx, token_id] /= repetition_penalty
            
            if no_repeat_ngram_size > 0 and current_ids.shape[1] >= no_repeat_ngram_size:
                for batch_idx in range(batch_size):
                    ngrams = {}
                    seq = current_ids[batch_idx].tolist()
                    for i in range(len(seq) - no_repeat_ngram_size + 1):
                        ngram = tuple(seq[i:i + no_repeat_ngram_size - 1])
                        next_token = seq[i + no_repeat_ngram_size - 1]
                        if ngram not in ngrams:
                            ngrams[ngram] = []
                        ngrams[ngram].append(next_token)
                    
                    if len(seq) >= no_repeat_ngram_size - 1:
                        current_ngram = tuple(seq[-(no_repeat_ngram_size - 1):])
                        if current_ngram in ngrams:
                            for banned_token in ngrams[current_ngram]:
                                next_token_logits[batch_idx, banned_token] = float('-inf')
            
            if current_ids.shape[1] < min_length:
                next_token_logits[:, eos_token_id] = float('-inf')
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            if top_k > 0:
                for batch_idx in range(batch_size):
                    top_k_values, top_k_indices = torch.topk(next_token_logits[batch_idx], min(top_k, next_token_logits.shape[-1]), dim=-1)
                    next_token_logits[batch_idx] = torch.full_like(next_token_logits[batch_idx], float('-inf'))
                    next_token_logits[batch_idx].scatter_(0, top_k_indices, top_k_values)
            
            if top_p < 1.0:
                for batch_idx in range(batch_size):
                    sorted_logits, sorted_indices = torch.sort(next_token_logits[batch_idx], descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[batch_idx, indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            next_tokens = next_tokens.masked_fill(finished.unsqueeze(1), pad_token_id)
            current_ids = torch.cat([current_ids, next_tokens], dim=1)
            
            finished = finished | (next_tokens.squeeze(1) == eos_token_id)
            
            if stopping_criteria is not None:
                if stopping_criteria(current_ids, None):
                    break
            
            if finished.all():
                break
        
        if return_type == 'text':
            if batch_size == 1:
                return self.tokenizer.decode(current_ids[0].tolist())
            else:
                return [self.tokenizer.decode(seq.tolist()) for seq in current_ids]
        elif return_type == 'pt':
            if output_scores:
                return current_ids, torch.stack(all_scores, dim=1)
            return current_ids
        else:
            if output_scores:
                return [seq.tolist() for seq in current_ids], all_scores
            return [seq.tolist() for seq in current_ids]


class TextDiffusionModel:
    """
    The definition of a model similar to LLada
    """
    def diffusion_generate(
        self,
        prompt=None,
        input_ids=None,
        num_steps=50,
        max_length=100,
        min_length=0,
        output_scores=False,
        return_type='text'):
        """
        1. Mask the input prompt at 100%
        2. 
        """
        raise NotImplementedError
        
                    
