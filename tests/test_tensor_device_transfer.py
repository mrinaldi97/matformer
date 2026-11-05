"""Device-transfer regressions using PyTorch's storage-free meta device."""
import unittest
import torch
from matformer.tensors_dataclasses import PaddedTensor


class TensorDeviceTransferTests(unittest.TestCase):
    def make_sequence(self):
        ids = torch.tensor([[5, 7, 6, 0], [5, 8, 9, 6]])
        return PaddedTensor(
            tensor=ids, padding_mask=ids == 0,
            cloze_mask=ids == 7, document_mask=torch.ones_like(ids),
            recurrence_mask=torch.tensor([True, False]),
        )

    def test_padded_masks_follow_device(self):
        sequence = self.make_sequence()
        result = sequence.to('meta')
        self.assertIs(result, sequence)
        for name in ('tensor', 'padding_mask', 'cloze_mask', 'document_mask', 'recurrence_mask'):
            self.assertEqual(getattr(sequence, name).device.type, 'meta', name)
        self.assertEqual(sequence.padding_mask.dtype, torch.bool)

    def test_unpadded_indices_and_lengths_follow_device(self):
        sequence = self.make_sequence().unpad().to('meta')
        for name in ('tensor', 'indices', 'cu_seqlens', 'cloze_mask', 'document_mask', 'recurrence_mask'):
            self.assertEqual(getattr(sequence, name).device.type, 'meta', name)
        self.assertEqual(sequence.indices.dtype, torch.int64)
        self.assertEqual(sequence.cu_seqlens.dtype, torch.int32)

    def test_dtype_conversion_preserves_mask_and_index_types(self):
        sequence = self.make_sequence().to(dtype=torch.float16)
        self.assertEqual(sequence.tensor.dtype, torch.float16)
        self.assertEqual(sequence.padding_mask.dtype, torch.bool)
        self.assertEqual(sequence.cloze_mask.dtype, torch.bool)
        self.assertEqual(sequence.recurrence_mask.dtype, torch.bool)
        self.assertEqual(sequence.document_mask.dtype, torch.int64)
        unpadded = self.make_sequence().unpad().to(dtype=torch.float16)
        self.assertEqual(unpadded.indices.dtype, torch.int64)
        self.assertEqual(unpadded.cu_seqlens.dtype, torch.int32)

if __name__ == '__main__':
    unittest.main()
