import torch
import numpy as np
from matformer.transformer_blocks import MaskBuilder

# Mock config class
class Config:
    def __init__(self):
        self.sliding_window_size = 3

def print_mask(mask, description):
    print(f"\n{description}:")
    print("   ", " ".join([f"{i:2d}" for i in range(len(mask[0]))]))
    for i, row in enumerate(mask):
        print(f"{i:2d}:", " ".join([" T" if x else " F" for x in row]))

def test_masks():
    config = Config()
    builder = MaskBuilder(config)
    
    # Test parameters
    q_len, kv_len = 12, 12
    batch_size, num_heads = 1, 1
    
    # Create fake document mask: [0,0,1,1,2,2] (3 documents)
    document_mask = [[0, 0, 1, 1, 2, 2,3,3,3,3,3,4]]
    
    print("Testing MaskBuilder with q_len=6, kv_len=6, sliding_window=3")
    print("Document mask:", document_mask[0])
    
    # Test 1: Causal only
    mask1 = builder.build_mask_tensor(['causal'], q_len, kv_len, batch_size, num_heads, implementation='sdpa')
    print_mask(mask1, "1) Causal mask")
    
    # Test 2: Causal + Sliding
    mask2 = builder.build_mask_tensor(['causal'], q_len, kv_len, batch_size, num_heads, 
                                     is_sliding=True, implementation='sdpa')
    print_mask(mask2, "2) Causal + Sliding window (size=3)")
    
    # Test 3: Document only
    mask3 = builder.build_mask_tensor(['document'], q_len, kv_len, batch_size, num_heads,
                                     document_mask=document_mask, implementation='sdpa')
    print_mask(mask3, "3) Document mask only")
    
    # Test 4: Document + Causal + Sliding
    mask4 = builder.build_mask_tensor(['document', 'causal'], q_len, kv_len, batch_size, num_heads,
                                     is_sliding=True, document_mask=document_mask, implementation='sdpa')
    print_mask(mask4, "4) Document + Causal + Sliding")

if __name__ == "__main__":
    test_masks()
