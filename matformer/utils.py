import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import reduce
import numpy as np

class MaskBuilder:
    def __init__(self, config):
        self.config = config

    def build_mask_tensor(self, attention_types, query, kv=None, batch_size=None, num_heads=None,
                          is_sliding=False, document_mask=None, nested=False, **kwargs):
        if kv is None:
            kv = query.tensor
        else:
            kv = kv.tensor
        query = query.tensor

        B, L, S = query.shape[0], query.shape[-2], kv.shape[-2]
        device = query.device

        if self.config.attn_impl == 'sdpa':

            q_idx = torch.arange(L, device=device).view(L, 1)
            k_idx = torch.arange(S, device=device).view(1, S)
            mask = torch.ones(B, L, S, dtype=torch.bool, device=device)

            if 'causal' in attention_types:
                causal_mask = q_idx >= k_idx
                mask &= causal_mask

            if is_sliding:
                sliding_mask = torch.abs(q_idx - k_idx) <= self.config.sliding_window_size
                mask &= sliding_mask

            if 'document' in attention_types and document_mask is not None:
                doc_mask = document_mask[:, :, None] == document_mask[:, None, :]
                mask |= doc_mask

            return mask

        elif self.config.attn_impl == 'flex':
            def mask_fn(b, h, q, k):
                m = torch.ones_like(q, dtype=torch.bool)

                if 'causal' in attention_types:
                    m &= q >= k
                if is_sliding:
                    m &= (torch.abs(q - k) <= self.config.sliding_window_size)
                if 'document' in attention_types and document_mask is not None:
                    m |= (document_mask[b, q] == document_mask[b, k])

                return m

            if nested:
                return create_nested_block_mask(mask_mod=mask_fn, q_nt=query, kv_nt=kv,
                                                B=batch_size, H=num_heads, device=device)
            else:
                return create_block_mask(mask_mod=mask_fn, Q_LEN=L, KV_LEN=S,
                                         B=batch_size, H=num_heads, device=device)

        else:
            print("Unsupported attention implementation!")
            return None

class Patcher(nn.Module):
    """
        This module receives a tensor as input and call a patching function.
        According to the result of the patching function, it outputs the patched 
        version of the tensor.
    """
    

    
class LpPooling(torch.nn.Module):
    def __init__(self, dim=0, p=2, keepdim=False):
        super().__init__()
        self.dim = dim
        self.p = torch.nn.Parameter(torch.tensor(float(p)))
        self.keepdim = keepdim

    def forward(self, x, dim):
        p_clamped = torch.clamp(self.p, min=1e-6)
        return torch.pow(torch.mean(torch.pow(torch.abs(x), p_clamped), dim=dim, keepdim=self.keepdim), 1 / p_clamped)



def printmem(text):
    # Una funzioncina rozza per controllare l'uso della memoria, da togliere una volta che il codice è completo
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    print(f"Memory at {text}: {mem_used_MB}")


from dataclasses import is_dataclass, fields
from typing import TypeVar

T = TypeVar("T")


def dataclass_from_dict(d: dict, ty: type[T]) -> T:
#https://gist.github.com/paxbun/3eb7d762a013f915fce69d4daecfab37    
    assert is_dataclass(ty)
    for field in fields(ty):
        if field.name in d:
            if is_dataclass(field.type):
                assert isinstance(d[field.name], dict)
                d[field.name] = fromdict(d[field.name], field.type)
    return ty(**d)


#https://github.com/tatp22/multidim-positional-encoding/blob/efeb8d9d70e8184da50eae9fddd1bbda10896529/positional_encodings/torch_encodings.py
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.channels = channels
        self.dtype_override = dtype_override

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros(
            (x, self.channels),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels, dtype_override)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.channels = channels

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels, dtype_override)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.channels = channels

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels, dtype_override)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert (
            tensor.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size(), penc.size()
        )
        return tensor + penc.to(tensor.device)
