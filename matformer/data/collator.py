"""Collate independent token sequences without treating EOS as padding."""
from dataclasses import dataclass
import torch
from matformer.tensors_dataclasses import PaddedTensor


@dataclass
class TokenCollator:
    pad_token_id: int
    max_seq_len: int
    varlen_strategy: str = 'padding'
    with_meta: bool = False

    def __call__(self, batch):
        if self.varlen_strategy not in ('padding', 'unpadding'):
            raise ValueError('varlen_strategy must be padding or unpadding')
        if not batch or self.max_seq_len is None or self.max_seq_len < 2:
            raise ValueError('A non-empty batch and max_seq_len >= 2 are required')
        rows, finished, recurrence, metadata = [], [], [], []
        for item in batch:
            is_record = isinstance(item, dict)
            done = bool(item.get('worker_has_finished', False)) if is_record else False
            row = item.get('object') if is_record else item
            if isinstance(row, dict):
                row = row.get('tokens')
            if done:
                row = [self.pad_token_id]
            if row is None or len(row) == 0 or len(row) > self.max_seq_len:
                raise ValueError('Token sequence is empty or exceeds max_seq_len; fix preprocessing')
            rows.append(torch.as_tensor(row, dtype=torch.long))
            finished.append(done)
            recurrence.append(item.get('is_same_document') if is_record else None)
            metadata.append(item.get('metadata') if is_record else None)
        tokens = torch.full((len(rows), self.max_seq_len), self.pad_token_id, dtype=torch.long)
        padding = torch.ones_like(tokens, dtype=torch.bool)
        loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
        for i, row in enumerate(rows):
            tokens[i, :len(row)] = row
            padding[i, :len(row)] = False
            loss_mask[i, 1:len(row)] = not finished[i]
        attributes = {'loss_mask': loss_mask}
        if self.with_meta:
            attributes['metadata'] = metadata
        sequence = PaddedTensor(
            tensor=tokens, padding_mask=padding, extra_attributes=attributes,
            extra_follow_keys=['loss_mask'],
            recurrence_mask=(torch.tensor([bool(x) for x in recurrence])
                             if any(x is not None for x in recurrence) else None),
        )
        if self.varlen_strategy == 'unpadding':
            sequence = sequence.unpad()
        return {'sequence': sequence, 'worker_has_finished': all(finished)}
