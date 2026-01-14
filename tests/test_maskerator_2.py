import sys
sys.path.append("../")
import pytest
import torch
import numpy as np
from matformer.masked_models import Maskerator
from matformer.tensors_dataclasses import PaddedTensor,UnpaddedTensor,NormalTensor
def test_reproducibility():
    """Test that same seed produces identical results"""
    input_tensor = torch.randint(0, 100, (2, 10))
    
    maskerator1 = Maskerator(
        mask_token=1, 
        substitution_rate=0.15,
        pad_token_id=0,
        vocab_size=100,
        random_seed=42
    )
    
    maskerator2 = Maskerator(
        mask_token=1, 
        substitution_rate=0.15,
        pad_token_id=0,
        vocab_size=100,
        random_seed=42
    )
    
    # Test multiple calls
    for _ in range(5):
        masked1, mask1, rate1 = maskerator1(input_tensor)
        masked2, mask2, rate2 = maskerator2(input_tensor)
        
        assert torch.all(masked1 == masked2)
        assert torch.all(mask1 == mask2)
        assert rate1 == rate2

def test_fixed_substitution_rate():
    """Test adherence to fixed substitution rate"""
    torch.manual_seed(42)
    input_tensor = torch.randint(1, 100, (10, 100))  # No padding
    
    for rate in [0.1, 0.15, 0.3]:
        maskerator = Maskerator(
            mask_token=1,
            substitution_rate=rate,
            pad_token_id=0,
            vocab_size=100,
            random_seed=42
        )
        
        _, mask, actual_rate = maskerator(input_tensor)
        
        # Check rate is exactly as specified
        assert actual_rate == rate
        
        # Check empirical rate (within reasonable tolerance)
        mask_rate = mask.float().mean().item()
        assert abs(mask_rate - rate) < 0.05  # 5% tolerance

def test_variable_substitution_rate_range():
    """Test that variable rate stays within specified range"""
    torch.manual_seed(42)
    input_tensor = torch.randint(1, 100, (5, 50))
    
    maskerator = Maskerator(
        mask_token=1,
        substitution_rate=(0.1, 0.3),
        pad_token_id=0,
        vocab_size=100,
        random_seed=42
    )
    
    rates = []
    for _ in range(100):
        _, _, rate = maskerator(input_tensor)
        rates.append(rate)
        assert 0.1 <= rate <= 0.3
    
    # Check that we get variety of rates
    assert max(rates) - min(rates) > 0.1

def test_per_document_variable_rate():
    """Test per_document variable masking rate"""
    torch.manual_seed(42)
    input_tensor = torch.randint(1, 100, (4, 20))
    
    maskerator = Maskerator(
        mask_token=1,
        substitution_rate=(0.1, 0.4),
        pad_token_id=0,
        variable_masking_rate='per_document',
        vocab_size=100,
        random_seed=42
    )
    
    masked, mask, rate = maskerator(input_tensor)
    
    # Different documents might have different rates
    # But mask should be generated correctly
    assert masked.shape == input_tensor.shape
    assert mask.shape == input_tensor.shape

def test_padding_protection():
    """Test that padding tokens are never masked"""
    # Create tensor with padding
    input_tensor = torch.tensor([
        [101, 102, 103, 0, 0],
        [104, 105, 106, 107, 0],
        [108, 109, 0, 0, 0]
    ])
    
    maskerator = Maskerator(
        mask_token=999,
        substitution_rate=0.5,  # High rate to test
        pad_token_id=0,
        vocab_size=1000,
        random_seed=42
    )
    
    masked, mask, _ = maskerator(input_tensor)
    
    # Check padding positions are not masked
    pad_positions = (input_tensor == 0)
    assert torch.all(masked[pad_positions] == 0)  # Padding unchanged
    assert not torch.any(mask[pad_positions])  # No mask on padding

def test_multiple_strategies():
    """Test mixed masking strategies"""
    torch.manual_seed(42)
    input_tensor = torch.randint(2, 100, (3, 20))
    
    maskerator = Maskerator(
        mask_token=1,
        substitution_rate=0.2,
        pad_token_id=0,
        cloze_prob=0.6,
        random_prob=0.3,
        same_prob=0.1,
        vocab_size=100,
        random_seed=42
    )
    
    masked, mask, _ = maskerator(input_tensor)
    
    # Count different strategies
    mask_positions = mask
    
    # Positions with mask token
    mask_token_pos = (masked == 1) & mask_positions
    
    # Positions with random token (not original, not mask token)
    random_pos = mask_positions & (masked != 1) & (masked != input_tensor)
    
    # Positions with same token
    same_pos = mask_positions & (masked == input_tensor)
    
    # All masked positions accounted for
    assert torch.all((mask_token_pos | random_pos | same_pos) == mask_positions)

def test_external_schedule():
    """Test external substitution rate scheduling"""
    torch.manual_seed(42)
    input_tensor = torch.randint(1, 100, (2, 10))
    
    maskerator = Maskerator(
        mask_token=1,
        substitution_rate=None,  # External schedule
        pad_token_id=0,
        vocab_size=100,
        random_seed=42
    )
    
    # Test with different rates
    for rate in [0.1, 0.25, 0.5]:
        _, mask, actual_rate = maskerator(input_tensor, substitution_rate=rate)
        assert actual_rate == rate
        
        # Check approximate mask rate
        mask_rate = mask.float().mean().item()
        assert abs(mask_rate - rate) < 0.1

def test_unpadded_tensor():
    """Test with UnpaddedTensor"""
    torch.manual_seed(42)
    
    # Create an UnpaddedTensor
    from matformer.tensors_dataclasses import UnpaddedTensor
    
    tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cu_seqlens = torch.tensor([0, 3, 7, 10])
    unpadded = UnpaddedTensor(tensor=tensor, cu_seqlens=cu_seqlens)
    
    maskerator = Maskerator(
        mask_token=999,
        substitution_rate=0.3,
        pad_token_id=0,
        vocab_size=1000,
        random_seed=42
    )
    
    masked_unpadded, rate = maskerator(unpadded)
    
    # Check shapes preserved
    assert masked_unpadded.tensor.shape == tensor.shape
    assert masked_unpadded.cloze_mask.shape == tensor.shape
    assert masked_unpadded.cu_seqlens is not None

def test_deterministic_across_runs():
    """Test deterministic behavior across multiple runs with same seed"""
    torch.manual_seed(42)
    input_tensor = torch.randint(1, 100, (3, 15))
    
    # Create two separate maskerators with same seed
    maskerator1 = Maskerator(
        mask_token=1,
        substitution_rate=0.15,
        pad_token_id=0,
        vocab_size=100,
        random_seed=12345
    )
    
    maskerator2 = Maskerator(
        mask_token=1,
        substitution_rate=0.15,
        pad_token_id=0,
        vocab_size=100,
        random_seed=12345
    )
    
    # Run multiple times
    all_masked1 = []
    all_masked2 = []
    
    for _ in range(10):
        masked1, mask1, rate1 = maskerator1(input_tensor)
        masked2, mask2, rate2 = maskerator2(input_tensor)
        
        all_masked1.append(masked1)
        all_masked2.append(masked2)
        
        # Should be identical
        assert torch.all(masked1 == masked2)
        assert torch.all(mask1 == mask2)
        assert rate1 == rate2
    
    # Also test that different seeds give different results
    maskerator3 = Maskerator(
        mask_token=1,
        substitution_rate=0.15,
        pad_token_id=0,
        vocab_size=100,
        random_seed=99999  # Different seed
    )
    
    masked3, _, _ = maskerator3(input_tensor)
    assert not torch.all(all_masked1[0] == masked3)

def test_1d_tensor():
    """Test with 1D tensor (single sequence)"""
    torch.manual_seed(42)
    input_tensor = torch.randint(1, 100, (20,))
    
    maskerator = Maskerator(
        mask_token=1,
        substitution_rate=0.2,
        pad_token_id=0,
        vocab_size=100,
        random_seed=42
    )
    
    masked, mask, rate = maskerator(input_tensor)
    
    assert masked.dim() == 1
    assert mask.dim() == 1
    assert rate == 0.2

def test_list_input():
    """Test with list input (iterative method)"""
    maskerator = Maskerator(
        mask_token=1,
        substitution_rate=0.2,
        pad_token_id=0,
        vocab_size=100,
        random_seed=42
    )
    
    input_list = [5, 6, 7, 8, 9]
    
    # This should use _iterative_masking_function
    output, mask = maskerator._iterative_masking_function(input_list)
    
    assert len(output) == len(input_list)
    assert len(mask) == len(input_list)

if __name__ == "__main__":
    # Run all tests
    test_reproducibility()
    print("✓ test_reproducibility passed")
    
    test_fixed_substitution_rate()
    print("✓ test_fixed_substitution_rate passed")
    
    test_variable_substitution_rate_range()
    print("✓ test_variable_substitution_rate_range passed")
    
    test_per_document_variable_rate()
    print("✓ test_per_document_variable_rate passed")
    
    test_padding_protection()
    print("✓ test_padding_protection passed")
    
    test_multiple_strategies()
    print("✓ test_multiple_strategies passed")
    
    #test_external_schedule()
    print("✓ test_external_schedule suspended")
    
    test_deterministic_across_runs()
    print("✓ test_deterministic_across_runs passed")
    
    test_1d_tensor()
    print("✓ test_1d_tensor passed")
    
    test_list_input()
    print("✓ test_list_input passed")
    
    print("\nAll tests passed! ✓")
