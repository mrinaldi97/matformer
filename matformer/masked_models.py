"""
matformer/masked_models.py
"""   
import random
import torch

def maskerator(input_ids, MASK_TOKEN=-100, substitution_rate=0.3):
    if isinstance(input_ids, torch.Tensor) and input_ids.dim() > 1:
        # Input is batched
        batch_outputs = []
        batch_cloze_masks = []
        for batch_item in input_ids:
            output, cloze_mask = maskerator(batch_item, MASK_TOKEN, substitution_rate)
            batch_outputs.append(output)
            batch_cloze_masks.append(cloze_mask)
        return torch.stack(batch_outputs), torch.stack(batch_cloze_masks)
    
    if isinstance(input_ids, torch.Tensor):
        input_list = input_ids.tolist()
        device = input_ids.device
    else:
        input_list = input_ids
        device = None
    
    output = []
    bool_cloze_mask = [] 
    
    for token in input_list:
        destino = 1 - random.random()
        if destino <= substitution_rate:
            if destino <= 1:
                output.append(MASK_TOKEN)
                bool_cloze_mask.append(True) 
            else:
                if destino <= 0.5:
                    # Sostituisco con Random (non implementato)
                    output.append(random.randint(0, 259))  # 259 is max token ID
                    bool_cloze_mask.append(True)
                else:
                    # Non sostituisco (non implementato)
                    output.append(token)
                    bool_cloze_mask.append(True)                                
        else:
            # Il token non viene sostituito
            output.append(token)
            bool_cloze_mask.append(False) 
    
    assert len(output) == len(bool_cloze_mask)
    
    if device is not None:
        return (
            torch.tensor(output, device=device), 
            torch.tensor(bool_cloze_mask, dtype=torch.bool, device=device) 
        )
    return output, bool_cloze_mask
