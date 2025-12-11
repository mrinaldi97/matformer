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
from matformer.matformer_registry import registry
from dataclasses import replace

@registry.register("hooks", "gated_bridge_injector", "default", requires=["torch"], priority=0)
class GatedBridgeInjector(nn.Module):
    def __init__(self, config, cache, layer_idx, receive_from, bridge_type='conv', additional_loss=True):
        super().__init__()
        self.full_cache=cache
        self.cache = cache.storage
        self.layer_idx = layer_idx
        self.receive_from = receive_from
        self.hidden_size = config.hidden_size
        self.additional_loss=additional_loss
        self.bridge_type=bridge_type
        if self.bridge_type=='mlp':
            self.bridge_mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
        else:
            self.bridge_conv = nn.Sequential(
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, groups=self.hidden_size),
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=1),
                nn.GELU(),
                
            )  
            self.conv_norm=nn.LayerNorm(self.hidden_size) # requires [B, L, D] format      
        self.xattn = MultiHeadAttention(
            q_dim=config.hidden_size, k_dim=config.hidden_size, v_dim=config.hidden_size,
            is_cross_attention=True, nheads=config.num_attention_heads,
            positional_encoding='nope',
            is_causal=False, cache=cache
        )
        self.linear_post_attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.post_attn_norm = nn.LayerNorm(self.hidden_size)
        #self.gate = nn.Parameter(torch.tensor(0.8)) #Initialized at 0.8
        self.gating_layer = nn.Linear(2 * self.hidden_size, 1)


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
            
            current_x = x.tensor[recurrence_mask] 
            memory_raw = previous_state.tensor[recurrence_mask]
            if self.bridge_type=='mlp':
                memory_bridged = self.bridge_mlp(memory_raw)
            else:
                mem_transposed = memory_raw.transpose(1, 2) 
                bridged_transposed = self.bridge_conv(mem_transposed)
                memory_bridged = bridged_transposed.transpose(1, 2)
                memory_bridged=self.conv_norm(memory_bridged)
            curr_x_obj = replace(x, tensor=current_x)
            if self.additional_loss:
                aux_loss = F.mse_loss(memory_bridged, current_x)
                if 'aux_losses' not in self.cache:
                    self.cache['aux_losses'] = []
                self.cache['aux_losses'].append(aux_loss)
                
            mem_obj = replace(previous_state, tensor=memory_bridged)
            attn_out = self.xattn(query_input=curr_x_obj, key_input=mem_obj, value_input=mem_obj)
            injection_signal = attn_out.tensor
            injection_signal = self.post_attn_norm(injection_signal)
            #for layer in self.processing_layers:
            #    injection_signal = injection_signal + layer(injection_signal)
            injection_signal = self.linear_post_attn(injection_signal)
            #gate_hyperparam = torch.sigmoid(self.gate)
            gate_per_token = torch.sigmoid(self.gating_layer(
                torch.cat([current_x, memory_bridged], dim=-1)
            ))    
            self.full_cache.additional_logs[f"gate/{self.layer_idx}/update_norm"]=((g * torch.tanh(injection_signal)).norm(dim=-1)).mean().item()
            self.full_cache.additional_logs[f"gate/{self.layer_idx}/active_fraction"]=(gate_per_token > 0.5).float().mean().item()
            self.full_cache.additional_logs[f"gate/{self.layer_idx}/mean"]=gate_per_token.mean().item()
            self.full_cache.additional_logs[f"gate/{self.layer_idx}/std"]=gate_per_token.std().item()              
            gated_signal = gate_per_token * torch.tanh(injection_signal)
            #gated_signal = self.gate * torch.tanh(injection_signal)   
            out_tensor = x.tensor.clone()
            out_tensor[recurrence_mask] += gated_signal
            result = replace(x, tensor=out_tensor)
            return result.unpad() if wasUnpadded else result
        except Exception as e:
            print(f"Caught exception: {e} in hook at layer {self.layer_idx}")
            import traceback
            traceback.print_exc()
            return x 


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
