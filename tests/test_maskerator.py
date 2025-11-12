import torch
import unittest
from matformer.masked_models import Maskerator 

class TestMaskerator(unittest.TestCase):

    def setUp(self):
        """Set up common variables for all tests."""
        self.vocab_size = 100
        self.mask_token = 99
        self.pad_token = 0
        self.batch_size = 40
        self.seq_len = 1024
        
        # Create a large tensor of random, valid token IDs (1 to 98)
        # This ensures that no input token is accidentally the same as the mask_token (99)
        self.input_ids = torch.randint(
            low=1, 
            high=self.mask_token, # Generate tokens from 1 up to (but not including) 99
            size=(self.batch_size, self.seq_len), 
            dtype=torch.long
        )
        
        # Create variable-length padding
        # Generate random sequence lengths for each row
        min_len = 10 # Ensure no row is completely empty
        seq_lengths = torch.randint(
            min_len, 
            self.seq_len + 1, 
            size=(self.batch_size,)
        )
        
        # Create a mask to apply padding
        # (arange(1024) >= seq_lengths_per_row_expanded_to_1024)
        padding_mask = torch.arange(self.seq_len) >= seq_lengths.unsqueeze(1)
        
        # Apply padding
        self.input_ids[padding_mask] = self.pad_token
        
        # Calculate the non-padding mask and the total count of real tokens
        self.non_pad_mask = (self.input_ids != self.pad_token)
        self.non_pad_count = self.non_pad_mask.sum().item()
        
    def test_bert_style_masking(self):
        """Tests the simple masking mode (only [MASK])."""
        print("Running test_bert_style_masking...")
        torch.manual_seed(42) # For reproducible results
        
        rate = 0.15
        maskerator = Maskerator(
            mask_token=self.mask_token,
            substitution_rate=rate,
            pad_token_id=self.pad_token,
            vocab_size=self.vocab_size
        )
        
        output_ids, cloze_mask = maskerator(self.input_ids)

        # --- Assert: Shape and Types ---
        self.assertEqual(output_ids.shape, self.input_ids.shape)
        self.assertEqual(cloze_mask.shape, self.input_ids.shape)
        self.assertEqual(cloze_mask.dtype, torch.bool)

        # --- Assert: Padding Integrity ---
        # 1. No [PAD] token should be masked
        pad_mask = (self.input_ids == self.pad_token)
        self.assertFalse(cloze_mask[pad_mask].any(), "Error: A [PAD] token was masked!")
        
        # 2. [PAD] tokens in the output must remain [PAD]
        self.assertTrue((output_ids[pad_mask] == self.pad_token).all(), "Error: A [PAD] token was modified!")

        # --- Assert: Masking Logic ---
        # 1. Check that *all* masked tokens became [MASK]
        self.assertTrue((output_ids[cloze_mask] == self.mask_token).all())
        
        # 2. Check the rate (approximately)
        num_masked = cloze_mask.sum().item()
        masked_ratio = num_masked / self.non_pad_count
        
        print(f"BERT Style: Target mask rate: {rate:.2%}")
        print(f"Masked {num_masked} of {self.non_pad_count} non-pad tokens. Actual rate: {masked_ratio:.2%}")
        # Assert that the actual mask rate is close to the target rate
        # delta=0.05 means it must be within +/- 5%
        self.assertAlmostEqual(masked_ratio, rate, delta=0.05)

    def test_80_10_10_masking(self):
        """Tests the complex mode (80% MASK, 10% RAND, 10% SAME)."""
        print("Running test_80_10_10_masking...")
        torch.manual_seed(42) # For reproducible results
        
        rate = 0.5 # High rate to get a good sample
        cloze_p, random_p, same_p = 0.8, 0.1, 0.1
        
        maskerator = Maskerator(
            mask_token=self.mask_token,
            substitution_rate=rate,
            pad_token_id=self.pad_token,
            vocab_size=self.vocab_size,
            cloze_prob=cloze_p,
            random_prob=random_p,
            same_prob=same_p
        )
        
        output_ids, cloze_mask = maskerator(self.input_ids)
        
        # --- Assert: Shape and Padding (as before) ---
        self.assertEqual(output_ids.shape, self.input_ids.shape)
        pad_mask = (self.input_ids == self.pad_token)
        self.assertFalse(cloze_mask[pad_mask].any())

        # --- Assert: Masking Logic ---
        num_masked = cloze_mask.sum().item()
        
        # Debug print
        print(f"80/10/10 Style: Target mask rate: {rate:.2%}")
        print(f"Masked {num_masked} of {self.non_pad_count} non-pad tokens. Actual rate: {num_masked/self.non_pad_count:.2%}")

        # Separate the 3 cases
        mask_token_mask = (output_ids == self.mask_token) & cloze_mask
        same_token_mask = (output_ids == self.input_ids) & cloze_mask
        
        # Random tokens are those masked but are NOT [MASK] and NOT the same
        random_token_mask = cloze_mask & ~mask_token_mask & ~same_token_mask
        
        num_mask = mask_token_mask.sum().item()
        num_random = random_token_mask.sum().item()
        num_same = same_token_mask.sum().item()

        # 1. The total must match
        self.assertEqual(num_mask + num_random + num_same, num_masked)

        # 2. Check the proportions (approximately)
        if num_masked > 0:
            mask_ratio = num_mask / num_masked
            random_ratio = num_random / num_masked
            same_ratio = num_same / num_masked
            
            print(f"Total masked: {num_masked}")
            print(f"[MASK] ratio: {mask_ratio:.2%} (Target: {cloze_p:.2%})")
            print(f"Random ratio: {random_ratio:.2%} (Target: {random_p:.2%})")
            print(f"Same ratio: {same_ratio:.2%} (Target: {same_p:.2%})")

            # Use a larger delta (e.g., 10%) because these are proportions *of* a
            # random sample, so they will have higher variance.
            self.assertAlmostEqual(mask_ratio, cloze_p, delta=0.1)
            self.assertAlmostEqual(random_ratio, random_p, delta=0.1)
            self.assertAlmostEqual(same_ratio, same_p, delta=0.1)


    def test_1d_tensor_input(self):
        """Tests that the maskerator works with a 1D input."""
        print("Running test_1d_tensor_input...")
        torch.manual_seed(42)
        input_1d = self.input_ids[0] # Take the first row
        non_pad_count_1d = (input_1d != self.pad_token).sum().item()
        
        rate = 0.15
        
        maskerator = Maskerator(
            mask_token=self.mask_token,
            substitution_rate=rate,
            pad_token_id=self.pad_token,
            vocab_size=self.vocab_size
        )
        
        output_ids, cloze_mask = maskerator(input_1d)
        
        # Assert: Shape and Types
        self.assertEqual(output_ids.shape, input_1d.shape)
        self.assertEqual(cloze_mask.shape, input_1d.shape)
        self.assertEqual(output_ids.dim(), 1)
        
        # Assert: Logic
        num_masked = cloze_mask.sum().item()
        
        print(f"1D Test: Target mask rate: {rate:.2%}")
        print(f"Masked {num_masked} of {non_pad_count_1d} non-pad tokens. Actual rate: {num_masked/non_pad_count_1d:.2%}")
        
        self.assertTrue((output_ids[cloze_mask] == self.mask_token).all())
        # Check rate approximation
        self.assertAlmostEqual(num_masked / non_pad_count_1d, rate, delta=0.05)
        # Check padding
        pad_mask = (input_1d == self.pad_token)
        self.assertFalse(cloze_mask[pad_mask].any())

if __name__ == '__main__':
    unittest.main()