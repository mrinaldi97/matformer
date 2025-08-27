"""
matformer/masked_models.py
"""   
import random
import torch

def maskerator(input_ids,mask_token,substitution_rate=0.3,cloze_mask=1.0,random_substitution=None,same_token_ratio=None,max_positional_embeddings=None):
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
    if (random_substitution is not None or same_token_ratio is not None):
        assert max_positional_embeddings is not None, "Need max_positional_embeddings"
        total = cloze_mask + (random_substitution or 0.0) + (same_token_ratio or 0.0)
        assert abs(total - 1.0) < 1e-6, "The proportions must sum to 1"

    if isinstance(input_ids, torch.Tensor) and input_ids.dim() > 1:
        outs, masks = [], []
        for row in input_ids:
            o, m = maskerator(row, mask_token, substitution_rate,
                              cloze_mask, random_substitution,
                              same_token_ratio, max_positional_embeddings)
            outs.append(o)
            masks.append(m)
        return torch.stack(outs), torch.stack(masks)

    if isinstance(input_ids, torch.Tensor):
        tokens = input_ids.tolist()
        device = input_ids.device
    else:
        tokens = list(input_ids)
        device = None

    output, cloze_mask_out = [], []
    for tok in tokens:
        if random.random() < substitution_rate:
            r = random.random()
            if (random_substitution is None) and (same_token_ratio is None):
                # only mask mode
                output.append(mask_token)
                cloze_mask_out.append(True)
            else:
                if r < cloze_mask:
                    output.append(mask_token)
                    cloze_mask_out.append(True)
                elif r < cloze_mask + (random_substitution or 0.0):
                    output.append(random.randint(0, max_positional_embeddings-1))
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
            output[i] = mask_token
            cloze_mask_out[i] = True
        if all(cloze_mask_out):
            i = random.randrange(len(tokens))
            output[i] = tokens[i]
            cloze_mask_out[i] = False

    if device is not None:
        return (torch.tensor(output, device=device),
                torch.tensor(cloze_mask_out, dtype=torch.bool, device=device))
    return output, cloze_mask_out


