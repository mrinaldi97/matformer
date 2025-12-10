import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import replace
from matformer.tensors_dataclasses import TensorDC,UnpaddedTensor,PaddedTensor
from matformer.transformer_blocks import MultiHeadAttention
from matformer.matformer_registry import registry

@registry.register(
    "hooks",
    "previous_state_saver",
    "default",
    requires=["torch"],
    priority=0,
)
class RecurrenceSaver(nn.Module):
    def __init__(self, config, cache, layer_idx, detach_depth=1):
        super().__init__()
        self.cache = cache.storage
        self.layer_idx = layer_idx
        self.detach_depth = detach_depth
        if 'for_recurrence' not in self.cache:
            self.cache['for_recurrence'] = {}
        if 'recurrence_steps' not in self.cache:
            self.cache['recurrence_steps'] = {}
        self.cache['recurrence_steps'][layer_idx] = 0
    def forward(self, x, *args, **kwargs):
        step = self.cache['recurrence_steps'][self.layer_idx]
        saved = replace(x, tensor=x.tensor.detach()) if step >= self.detach_depth else x
        self.cache['for_recurrence'][self.layer_idx] = saved
        self.cache['recurrence_steps'][self.layer_idx] += 1
        return x
        
@registry.register(
    "hooks",
    "previous_state_injection",
    "default",
    requires=["torch"],
    priority=0,
)
class RecurrenceInjector(nn.Module):
    def __init__(self, config, cache, layer_idx, receive_from):
        super().__init__()
        self.cache = cache.storage
        self.xattn = MultiHeadAttention(
            q_dim=config.hidden_size, k_dim=config.hidden_size,
            v_dim=config.hidden_size, is_cross_attention=True,
            nheads=config.num_attention_heads,
            positional_encoding='alibi', is_causal=False, cache=cache
        )
        self.layer_idx = layer_idx
        self.receive_from = receive_from
        assert receive_from >= self.layer_idx

    def forward(self, x, *args, **kwargs):
        try:
            previous_state = self.cache['for_recurrence'][self.receive_from]
            if previous_state is None:
                      print("No recurrency (None), skipping")
                      return x
            recurrence_mask = previous_state.recurrence_mask
            if recurrence_mask is None or not recurrence_mask.any():
                             print("No recurrency. Skipping")
                             return x               
            if recurrence_mask.device != x.tensor.device:
                recurrence_mask = recurrence_mask.to(x.tensor.device)
          
            wasUnpadded = False
            if isinstance(x, UnpaddedTensor):
                wasUnpadded = True
                x = x.pad()
                previous_state = previous_state.pad()
            
            selected_x = replace(x, tensor=x.tensor[recurrence_mask])
            selected_y = replace(previous_state, tensor=previous_state.tensor[recurrence_mask])
            
            processed_x = self.xattn(query_input=selected_x, key_input=selected_y, value_input=selected_y)
            
            out_tensor = x.tensor.clone()
            
            update_val = processed_x.tensor
            if update_val.dtype != out_tensor.dtype:
                update_val = update_val.to(out_tensor.dtype)
                
            out_tensor[recurrence_mask] += update_val

            result = replace(x, tensor=out_tensor)          
            return result.unpad() if wasUnpadded else result
        except Exception as e:
            print(f"Caught exception: {e} in hook at layer {self.layer_idx}")
            return x
