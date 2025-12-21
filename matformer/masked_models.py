"""
matformer/masked_models.py
"""
import random
from typing import Optional, Union, List, Tuple, Literal
import torch
from matformer.tensors_dataclasses import UnpaddedTensor,PaddedTensor

class Maskerator:
    def __init__(self, mask_token:int, substitution_rate: Union[float, Tuple[float, float], List[float], None], 
                 pad_token_id:int, variable_masking_rate:Literal['per_batch','per_document']='per_document', 
                 cloze_prob:Optional[float]=1.0, random_prob:Optional[float]=None, 
                 same_prob:Optional[float]=None, vocab_size:Optional[int]=None):
        """
        The maskerator has three modalities:
        - cloze_mask : replace with <mask>
        - random_substitution : replace with random token id [Not used by Neobert or Modernbert]
        - same_token_ratio : keep same token but mark in cloze mask [Not used by Neobert or Modernbert]

        Args:
            input_ids: [B,L] [L], torch tensor
            mask_token: int
            substitution_rate: probability of tokens to be masked
            pad_token_id: The token ID for [PAD]. This needs to avoid masking padding
            cloze_prob, random_prob, same_prob: proportions of different strategies, sum must be 1
            vocab_size: Required if random_prob > 0.

        Returns:
            output_ids: masked tensor
            cloze_mask: bool mask (True where prediction required)
        """
        self.mask_token=mask_token
        self.variable_masking_rate=variable_masking_rate
        
        if isinstance(substitution_rate,float):
            #Fixed substitution rate
            self.substitution_rate=substitution_rate # overall probability of masking
            self.substitution_rate_lower=None
            self.substitution_rate_upper=None
            self.external_schedule=False
        elif isinstance(substitution_rate,tuple) or isinstance(substitution_rate,list):
            assert len(substitution_rate)==2
            self.substitution_rate=None
            self.substitution_rate_lower=substitution_rate[0]
            self.substitution_rate_upper=substitution_rate[1]
            self.external_schedule=False
        elif substitution_rate==None:
            #The substitution rate is not set, the call function expects the substitution rate to be passed as argument
            self.substitution_rate=None
            self.substitution_rate_lower=None
            self.substitution_rate_upper=None
            self.external_schedule=True
        
        self.pad_token_id=pad_token_id
        self.vocab_size=vocab_size
        
        self.use_only_cloze_mask = (random_prob is None) and (same_prob is None)
        
        if not self.use_only_cloze_mask :
            assert vocab_size is not None, "vocab_size is required for random substitution"
            random_p = random_prob or 0.0
            same_p = same_prob or 0.0
            total = cloze_prob + random_p + same_p
            assert abs(total - 1.0) < 1e-6, "The proportions (cloze, random, same) must sum to 1"
            # Pre-calculate cumulative probabilities
            self.cloze_cutoff = cloze_prob
            self.random_cutoff = cloze_prob + random_p

    def __call__(self, input_ids, substitution_rate=None):
        """
        * If it receives a Nested Tensor, it will be unpacked and repacked
        * If it receives a Batch, the masking function will be applied to each batch's element
        * If it receives a list/tensor of elements, the masking function will be directly applied
        * If it receives an UnpaddedTensor, it will mask directly without padding
        """
        # Handle UnpaddedTensor - mask directly without padding/unpadding
        if isinstance(input_ids,UnpaddedTensor):
            return self._mask_unpadded_tensor(input_ids, substitution_rate)
        if isinstance(input_ids,PaddedTensor):
            tensor, cloze_mask, rate = self(input_ids.tensor, substitution_rate)
            return replace(input_ids, tensor=tensor, cloze_mask=cloze_mask), rate
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, but got {type(input_ids)}")
        
        if self.external_schedule:
            #The substitution rate is decided by the caller
            if substitution_rate is None:
                raise ValueError
            else:
                assert isinstance(substitution_rate,float)
                self.substitution_rate=substitution_rate
        
        if self.substitution_rate_upper is not None:
            #The substitution rate is uniformly sampled between lower and upper bound
            self.substitution_rate=random.uniform(self.substitution_rate_lower,self.substitution_rate_upper)
        
        if input_ids.is_nested:
            # NestedTensors. At the moment old method.
            return (*self._iterative_call(input_ids), self.substitution_rate)
        
        if input_ids.dim() == 1:
            # Input is a single tensor
            output_ids, cloze_mask = self._masking_function(input_ids.unsqueeze(0))
            return output_ids.squeeze(0), cloze_mask.squeeze(0), self.substitution_rate
        else:
            # Input is a batch of tokens
            return (*self._masking_function(input_ids), self.substitution_rate)

    def _mask_unpadded_tensor(self, unpadded_tensor, substitution_rate=None):
        """
        Mask an UnpaddedTensor directly without padding/unpadding.
        Maintains exact same behavior as _masking_function but operates on flattened tensors.
        """
        if self.external_schedule:
            #The substitution rate is decided by the caller
            if substitution_rate is None:
                raise ValueError
            else:
                assert isinstance(substitution_rate, float)
                self.substitution_rate = substitution_rate
        
        if self.substitution_rate_upper is not None:
            #The substitution rate is uniformly sampled between lower and upper bound
            self.substitution_rate = random.uniform(self.substitution_rate_lower, self.substitution_rate_upper)
        
        input_ids = unpadded_tensor.tensor
        device = input_ids.device
        
        # Compute per-token substitution rates (flattened version of per_document logic)
        if (self.variable_masking_rate == "per_document" and self.substitution_rate_upper is not None):
            # One rate per document
            B = unpadded_tensor.batch_size
            rates = torch.empty(B, device=device).uniform_(
                self.substitution_rate_lower, self.substitution_rate_upper
            )
            # Broadcast to each token in its document using cu_seqlens
            substitution_rate_vector = torch.zeros(input_ids.shape[0], device=device)
            for i in range(B):
                start = unpadded_tensor.cu_seqlens[i]
                end = unpadded_tensor.cu_seqlens[i + 1]
                substitution_rate_vector[start:end] = rates[i]
        else:
            # single rate over the batch
            substitution_rate_vector = self.substitution_rate
        
        # Apply core masking logic (same as _masking_function but on 1D tensors)
        output_ids, cloze_mask = self._apply_masking_logic(
            input_ids, substitution_rate_vector, device
        )
        
        # Return new UnpaddedTensor with masked data and cloze_mask
        from dataclasses import replace
        return replace(unpadded_tensor, tensor=output_ids, cloze_mask=cloze_mask), self.substitution_rate

    def _apply_masking_logic(self, input_ids, substitution_rate, device):
        """
        Core masking logic extracted for reuse between padded and unpadded paths.
        Works on any shaped tensor (1D, 2D, etc.) as long as substitution_rate broadcasts correctly.
        """
        prob_matrix = torch.rand(input_ids.shape, device=device)
        substitution_mask = (prob_matrix < substitution_rate)
        
        # Dont't mask [PAD] tokens.
        non_pad_mask = (input_ids != self.pad_token_id)
        cloze_mask = substitution_mask & non_pad_mask
        
        output_ids = input_ids.clone()
        
        if self.use_only_cloze_mask :
            output_ids.masked_fill_(cloze_mask, self.mask_token)
        else:
            # Multiple strategies
            mask_decision_matrix = torch.rand(input_ids.shape, device=device)
            
            # [MASK]
            mask_token_mask = (mask_decision_matrix < self.cloze_cutoff) & cloze_mask
            output_ids.masked_fill_(mask_token_mask, self.mask_token)
            
            # Random token
            random_token_mask = (mask_decision_matrix >= self.cloze_cutoff) & (mask_decision_matrix < self.random_cutoff) & cloze_mask
            if random_token_mask.any():
                random_tokens = torch.randint(0, self.vocab_size, input_ids.shape, device=device, dtype=input_ids.dtype)
                output_ids[random_token_mask] = random_tokens[random_token_mask]
        
        return output_ids, cloze_mask

    def _masking_function(self, input_ids):
        device = input_ids.device
        
        if (self.variable_masking_rate == "per_document" and self.substitution_rate_upper is not None):
            # One rate per document
            B = input_ids.size(0)
            rates = torch.empty(B, device=device).uniform_(
                self.substitution_rate_lower, self.substitution_rate_upper
            )
            # Broadcast to [B, L]
            substitution_rate_matrix = rates[:, None]
        else:
            # single rate over the batch
            substitution_rate_matrix = self.substitution_rate
        
        # Delegate to shared masking logic
        return self._apply_masking_logic(input_ids, substitution_rate_matrix, device)

    def _iterative_call(self, input_ids):
        if isinstance(input_ids, torch.Tensor) and input_ids.is_nested:
            # Input is a nested tensor
            masked_sequences, cloze_masks = zip(*[self._iterative_masking_function(seq) for seq in input_ids.unbind()])
            sequence = torch.nested.nested_tensor(torch.stack(masked_sequences), layout=torch.jagged)
            cloze_mask = torch.nested.nested_tensor(torch.stack(cloze_masks), layout=torch.jagged)
            return sequence,cloze_mask

        if isinstance(input_ids, torch.Tensor) and input_ids.dim() > 1:
            # Input is a batch of tokens
            outs, masks = [], []
            device = input_ids.device
            for row in input_ids:
                o, m = self._iterative_masking_function(row.tolist())
                outs.append(torch.tensor(o, device=device))
                masks.append(torch.tensor(m, device=device, dtype=torch.bool))
            return torch.stack(outs), torch.stack(masks)

        if isinstance(input_ids, torch.Tensor):
            # Input is a single tensor
            tokens = input_ids.tolist()
            device = input_ids.device
            output, mask = self._iterative_masking_function(tokens)
            return (torch.tensor(output, device=device),
                    torch.tensor(mask, dtype=torch.bool, device=device))
        else:
            tokens = list(input_ids)  # Input is a list of token
            return self._iterative_masking_function(tokens)

    def _iterative_masking_function(self,tokens):
        output, cloze_mask_out = [], []
        for tok in tokens:
            if random.random() < self.substitution_rate:
                r = random.random()
                if (self.random_prob is None) and (self.same_prob is None):
                    # only mask mode
                    output.append(self.mask_token)
                    cloze_mask_out.append(True)
                else:
                    if r < self.cloze_cutoff:
                        output.append(self.mask_token)
                        cloze_mask_out.append(True)
                    elif r < self.random_cutoff:
                        output.append(random.randint(0, self.vocab_size-1))
                        cloze_mask_out.append(True)
                    else:
                        # same_token
                        output.append(tok)
                        cloze_mask_out.append(True)
            else:
                output.append(tok)
                cloze_mask_out.append(False)
        return output, cloze_mask_out
