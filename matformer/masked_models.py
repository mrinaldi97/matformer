"""
matformer/masked_models.py
"""   
import random
from typing import Optional, Union, List, Tuple
import torch
class Maskerator:
    def __init__(self,
                 mask_token:int,
                 substitution_rate: Union[float, Tuple[float, float], List[float], None],
                 pad_token_id:int,
                 cloze_prob:Optional[float]=1.0,
                 random_prob:Optional[float]=None,
                 same_prob:Optional[float]=None,
                 vocab_size:Optional[int]=None):
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
        elif substitution_rate==None: #The substitution rate is not set, the call function expects the substitution rate to be passed as argument
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
        
    def __call__(self,input_ids,substitution_rate=None):
        """
            * If it receives a Nested Tensor, it will be unpacked and repacked
            * If it receives a Batch, the masking function will be applied to each batch's element
            * If it receives a list/tensor of elements, the masking function will be directly applied
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, but got {type(input_ids)}")
            
        if self.external_schedule: #The substitution rate is decided by the caller
            if substitution_rate is None:
                raise ValueError
            else:
                assert isintance(substitution_rate,float)
                self.substitution_rate=substitution_rate
        if self.substitution_rate_upper is not None: #The substitution rate is uniformly sampled between lower and upper bound
            self.substitution_rate=random.uniform(self.substitution_rate_lower,self.substitution_rate_upper)
            
        if input_ids.is_nested:
            # NestedTensors. At the moment old method.
            return self._iterative_call(input_ids)
        
        if input_ids.dim() == 1:
            # Input is a single tensor
            output_ids, cloze_mask = self._masking_function(input_ids.unsqueeze(0))
            return output_ids.squeeze(0), cloze_mask.squeeze(0), self.substitution_rate
        
        else:
            # Input is a batch of tokens
            return self._masking_function(input_ids)
                
    def _masking_function(self,input_ids):
        device = input_ids.device
        
        prob_matrix = torch.rand(input_ids.shape, device=device)
        substitution_mask = (prob_matrix < self.substitution_rate)
        
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
        
        
        # Avoid all masked or none mask to avoid problems such as NaN gradients
        """
        # Count non-pad tokens per sequence
        tokens_per_seq = (input_ids != self.pad_token_id).sum(dim=-1)
        # Count masks per sequence
        masked_per_seq = cloze_mask.sum(dim=-1)
        rows_needing_mask = (tokens_per_seq > 1) & (masked_per_seq == 0)
        if rows_needing_mask.any():
            # Get the row that we have to mask
            row_indices = rows_needing_mask.nonzero().squeeze(-1)
            # Create a probability distribution (1.0 for non-pad, 0.0 for pad)
            sample_probs = (input_ids[rows_needing_mask] != self.pad_token_id).float()
            # Sample one index (column) for each of these rows
            col_indices = torch.multinomial(sample_probs, num_samples=1).squeeze(-1)
            # Use the row and col indices to force a mask
            output_ids[row_indices, col_indices] = self.mask_token
            cloze_mask[row_indices, col_indices] = True
            
        # Ensure that no sequence is fully masked
        rows_fully_masked = (tokens_per_seq > 1) & (masked_per_seq == tokens_per_seq)
        if rows_fully_masked.any():
            # For these rows, pick one token to unmask (not [PAD] token)
            row_indices = rows_fully_masked.nonzero().squeeze(-1)
            sample_probs = (input_ids[rows_fully_masked] != self.pad_token_id).float()
            col_indices = torch.multinomial(sample_probs, num_samples=1).squeeze(-1)
            # Revert this token to its original value
            output_ids[row_indices, col_indices] = input_ids[row_indices, col_indices]
            cloze_mask[row_indices, col_indices] = False
        """               
        return output_ids, cloze_mask, self.substitution_rate

    def _iterative_call(self, input_ids):
        
        if isinstance(input_ids, torch.Tensor) and input_ids.is_nested:
            # Input is a nested tensor
            masked_sequences, cloze_masks = zip(*[self._iterative_masking_function(seq) for seq in input_ids.unbind()])
            sequence = torch.nested.nested_tensor(torch.stack(masked_sequences), layout=torch.jagged)
            cloze_mask = torch.nested.nested_tensor(torch.stack(cloze_masks), layout=torch.jagged)
            return sequence,cloze_mask,self.substitution_rate
            
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() > 1:
            # Input is a batch of tokens
            outs, masks = [], []
            device = input_ids.device
            for row in input_ids:
                o, m = self._iterative_masking_function(row.tolist())
                outs.append(torch.tensor(o, device=device))
                masks.append(torch.tensor(m, device=device, dtype=torch.bool))
            return torch.stack(outs), torch.stack(masks),self.substitution_rate

        if isinstance(input_ids, torch.Tensor):
            # Input is a single tensor
            tokens = input_ids.tolist()
            device = input_ids.device
            output, mask = self._iterative_masking_function(tokens)
            return (torch.tensor(output, device=device),
                    torch.tensor(mask, dtype=torch.bool, device=device), self.substitution_rate)            
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
                    else:  # same_token
                        output.append(tok)
                        cloze_mask_out.append(True)
            else:
                output.append(tok)
                cloze_mask_out.append(False)
        """
        # Avoid all masked or none mask to avoid problems such as NaN gradients
        if len(tokens) > 1:
            if all(not x for x in cloze_mask_out):
                i = random.randrange(len(tokens))
                output[i] = self.mask_token
                cloze_mask_out[i] = True
            if all(cloze_mask_out):
                i = random.randrange(len(tokens))
                output[i] = tokens[i]
                cloze_mask_out[i] = False
        """
        return output, cloze_mask_out, self.substitution_rate
