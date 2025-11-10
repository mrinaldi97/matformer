"""
matformer/masked_models.py
"""   
import random
from typing import Optional
import torch
class Maskerator:
    def __init__(self,mask_token:int,substitution_rate:float,cloze_prob:Optional[float]=1.0,random_prob:Optional[float]=None,same_prob:Optional[float]=None,max_position_embeddings:Optional[int]=None):
        """
        The maskerator has three modalities: 
          - cloze_mask : replace with <mask>
          - random_substitution : replace with random token id [Not used by Neobert or Modernbert]
          - same_token_ratio : keep same token but mark in cloze mask [Not used by Neobert or Modernbert]

        Args:
          input_ids: [B,L] [L], torch tensor
          mask_token: int
          substitution_rate: probability of tokens to be masked
          cloze_mask, random_substitution, same_token_ratio: proportions of different strategies, sum must be 1
          max_positional_embeddings: required if random or same != None

        Returns:
          output_ids: masked tensor
          cloze_mask: bool mask (True where prediction required)
        """
        self.mask_token=mask_token
        self.substitution_rate=substitution_rate
        self.cloze_prob=cloze_prob
        self.random_prob=random_prob
        self.same_prob=same_prob
        self.max_position_embeddings=max_position_embeddings
        if (random_prob is not None or same_prob is not None):
            assert max_positional_embeddings is not None, "Need max_positional_embeddings"
            total = cloze_mask + (random_prob or 0.0) + (same_prob or 0.0)
            assert abs(total - 1.0) < 1e-6, "The proportions must sum to 1"
        
    def __call__(self,input_ids):
        """
            * If it receives a Nested Tensor, it will be unpacked and repacked
            * If it receives a Batch, the masking function will be applied to each batch's element
            * If it receives a list/tensor of elements, the masking function will be directly applied
        """
        if isinstance(input_ids,torch.Tensor) and input_ids.is_nested:
            # Input is a nested tensor
            masked_sequences, cloze_masks = zip(*[self._masking_function(seq) for seq in sequence.unbind()])
            sequence = torch.nested.nested_tensor(torch.stack(masked_sequences), layout=torch.jagged)
            cloze_mask = torch.nested.nested_tensor(torch.stack(cloze_masks), layout=torch.jagged)
            return sequence,cloze_mask
            
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() > 1:
            # Input is a batch of tokens
            outs, masks = [], []
            device=input_ids.device
            for row in input_ids:
                o, m = self._masking_function(row)
                outs.append(torch.tensor(o,device=device))
                masks.append(torch.tensor(m,device=device,dtype=torch.bool))
            return torch.stack(outs), torch.stack(masks)

        if isinstance(input_ids, torch.Tensor):
            # Input is a single tensor
            tokens = input_ids.tolist()
            device = input_ids.device
            output,mask=self._masking_function(tokens)
            return (torch.tensor(output, device=device),
                    torch.tensor(mask, dtype=torch.bool, device=device))            
        else:
            tokens = list(input_ids) # Input is a list of token
            return self._masking_function(tokens)
        
                
    def _masking_function(self,tokens):
        output, cloze_mask_out = [], []
        for tok in tokens:
            if random.random() < self.substitution_rate:
                r = random.random()
                if (self.random_prob is None) and (self.same_prob is None):
                    # only mask mode
                    output.append(self.mask_token)
                    cloze_mask_out.append(True)
                else:
                    if r < cloze_mask:
                        output.append(self.mask_token)
                        cloze_mask_out.append(True)
                    elif r < cloze_mask + (self.random_prob or 0.0):
                        output.append(random.randint(0, self.max_positional_embeddings-1))
                        cloze_mask_out.append(True)
                    else:  # same_token
                        output.append(tok)
                        cloze_mask_out.append(True)
            else:
                output.append(tok)
                cloze_mask_out.append(False)

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
        return output, cloze_mask_out


