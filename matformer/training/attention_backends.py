"""Validate attention backend selection and restore backend-specific buffers."""
import torch
from torch import nn

from matformer.transformer_blocks.attention import MultiHeadAttention
from matformer.utils.matformer_cache import CachedStuff


def portable_state(module, state):
    """Validate/reconstruct fixed ALiBi buffers when switching attention backends."""
    expected = nn.Module.state_dict(module)
    state = dict(state)
    cache = CachedStuff()
    for key in list(state):
        if key.endswith('.alibi_slopes'):
            value = state[key]
            canonical = cache.get_alibi_slopes(value.numel(), torch.device('cpu'), torch.float32)
            torch.testing.assert_close(value.detach().float().cpu(), canonical, atol=0, rtol=0)
            if key not in expected:
                del state[key]
    for key, value in expected.items():
        if key.endswith('.alibi_slopes') and key not in state:
            state[key] = value
    return state


def validate_flash_attention(model, *, unpadding):
    for name, module in model.named_modules():
        if not isinstance(module, MultiHeadAttention) or module.requested_attn_impl != 'flash':
            continue
        kernels = (module.attn_kernel_causal, module.attn_kernel_noncausal) if module.is_hybrid else (
            getattr(module, 'attn_kernel', None),)
        for kernel in kernels:
            actual = getattr(kernel, '_matformer_implementation', None)
            if actual != 'flash':
                raise RuntimeError(
                    f'{name}: FlashAttention was requested but selected {actual!r}. '
                    'Install a matching flash-attn build and run on a supported CUDA GPU; '
                    'use attn_impl="sdpa" explicitly for a portable fallback.')
            if unpadding and not kernel._matformer_metadata.get('supports_unpadding', False):
                raise RuntimeError(f'{name}: selected attention cannot process unpadded sequences')
        if unpadding and 'rope' in module.positional_encoding:
            if not module.rotary_embedding_meta.get('supports_unpadding', False):
                raise RuntimeError(f'{name}: unpadded FlashAttention requires unpadded rotary embeddings')
