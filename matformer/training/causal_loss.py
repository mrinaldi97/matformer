"""Next-token targets respecting padding, sequence boundaries and token budgets."""
import torch
from matformer.tensors_dataclasses import UnpaddedTensor


def target_mask(sequence, pad_token_id=None):
    tokens = sequence.tensor
    valid = torch.ones_like(tokens, dtype=torch.bool)
    padding = getattr(sequence, 'padding_mask', None)
    if padding is not None:
        valid &= ~padding
    elif not isinstance(sequence, UnpaddedTensor) and pad_token_id is not None:
        valid &= tokens != pad_token_id
    if isinstance(sequence, UnpaddedTensor):
        starts = sequence.cu_seqlens[:-1].long()
        valid[starts[starts < valid.numel()]] = False
    else:
        valid[:, 1:] &= valid[:, :-1].clone()
        valid[:, 0] = False
    supplied = sequence.extra_attributes.get('loss_mask')
    if supplied is not None:
        valid &= supplied.bool()
    return valid


def causal_pairs(output, sequence, pad_token_id):
    if isinstance(output, UnpaddedTensor):
        sequence = sequence.unpad()
        valid = target_mask(sequence, pad_token_id)[1:]
        return output.tensor[:-1][valid], sequence.tensor[1:][valid]
    sequence = sequence.pad()
    valid = target_mask(sequence, pad_token_id)[:, 1:]
    return output.tensor[:, :-1, :][valid], sequence.tensor[:, 1:][valid]
