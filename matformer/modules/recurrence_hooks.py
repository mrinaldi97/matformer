import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import replace
from matformer.tensors_dataclasses import TensorDC,UnpaddedTensor,PaddedTensor, ModuleWrapper
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


import torch
import torch.nn as nn

@registry.register(
    "hooks",
    "previous_state_injection_multilayer",
    "default",
    requires=["torch"],
    priority=0,
)
class MultiLayerRecurrenceInjector(nn.Module):
    def __init__(self, config, cache, layer_idx, receive_from, injection_type='full_transformer', transformer_layers=2):
        super().__init__()
        self.cache = cache.storage #Todo change name
        self.layer_idx = layer_idx
        self.receive_from = receive_from
        self.injection_type = injection_type
        assert receive_from >= self.layer_idx
        self.layers = nn.ModuleList()
        norm_kwargs = {
            "normalized_shape": config.hidden_size,
            "eps": config.rms_norm_eps,
            "elementwise_affine": True
        }
        for i in range(transformer_layers):
            layer_components = nn.ModuleDict()
            layer_components['xattn'] = MultiHeadAttention(
                q_dim=config.hidden_size, k_dim=config.hidden_size,
                v_dim=config.hidden_size, is_cross_attention=True,
                nheads=config.num_attention_heads,
                positional_encoding='alibi', is_causal=False, cache=cache
            )

            if injection_type != 'attention_only':
                layer_components['attn_norm'] = ModuleWrapper(
                    cache.registry.create("norm", "layernorm", **norm_kwargs)
                )     
                layer_components['mlp_norm'] = ModuleWrapper(
                    cache.registry.create("norm", "layernorm", **norm_kwargs)
                )
                layer_components['mlp'] = ModuleWrapper(
                    cache.registry.create(
                        "mlp", 
                        'swiglu',
                        hidden_size=config.hidden_size,
                        ffn_factor=2
                    )
                )    
            self.layers.append(layer_components)

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

            out_tensor = x.tensor.clone()
            
            selected_x_tensor = x.tensor[recurrence_mask]
            selected_y_tensor = previous_state.tensor[recurrence_mask]
            current_working_state = replace(x, tensor=selected_x_tensor)
            memory_state = replace(previous_state, tensor=selected_y_tensor)

            for i, layer in enumerate(self.layers):
                
                attn_out = layer['xattn'](
                    query_input=current_working_state, 
                    key_input=memory_state, 
                    value_input=memory_state
                )
                
                current_working_state = current_working_state + attn_out

                if self.injection_type != 'attention_only':
                    normed_x = layer['attn_norm'](current_working_state)
                    mlp_out = layer['mlp'](normed_x)
                    current_working_state = current_working_state + mlp_out
                    current_working_state = layer['mlp_norm'](current_working_state)


            update_val = current_working_state.tensor
            out_tensor[recurrence_mask] += update_val

            result = replace(x, tensor=out_tensor)          
            return result.unpad() if wasUnpadded else result

        except Exception as e:
            print(f"Caught exception: {e} in hook at layer {self.layer_idx}")
            import traceback
            traceback.print_exc()
            return x        
            
@registry.register(
    "hooks",
    "previous_state_injection",
    "default",
    requires=["torch"],
    priority=0,
)
class RecurrenceInjector(nn.Module):
    def __init__(self, config, cache, layer_idx, receive_from, injection_type='attention_only', transformer_layers=3):
        """
        Injection type:
            * Attention only => A cross attention layer
            * Transformer => a transformer-like model with attention, residual and mlp
        Transformer_layers:
            * If "transformer" is selected, stack attn + residual + mlp layers
        """
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
