import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

class WERSAAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, decomp_levels: int = 2, 
                 random_features: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.decomp_levels = decomp_levels
        self.random_features = random_features
        
        # WERSA-specific components
        self.local_pooler = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=33,
            padding="same",
            groups=hidden_size
        )
        
        self.filter_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, decomp_levels + 1),
            nn.Sigmoid()
        )
        
        # Learnable parameters
        self.log_bandwidth = nn.Parameter(torch.tensor(0.0))
        self.scale_weights = nn.Parameter(torch.ones(decomp_levels + 1) * 0.5)
        
        # Random projection matrices
        random_q_matrix = torch.randn(self.head_dim, random_features) / math.sqrt(self.head_dim)
        random_k_matrix = torch.randn(self.head_dim, random_features) / math.sqrt(self.head_dim)
        self.register_buffer("random_q_matrix", random_q_matrix)
        self.register_buffer("random_k_matrix", random_k_matrix)
        
        # Post-attention norm
        self.attention_layer_norm = nn.LayerNorm(self.head_dim)

    def _vectorized_haar_dwt(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        seq_len = x.shape[2]
        orig_len = seq_len
        
        next_pow2 = 2 ** math.ceil(math.log2(seq_len)) if seq_len > 0 else 1
        padded = F.pad(x, (0, 0, 0, next_pow2 - seq_len)) if next_pow2 != seq_len else x
        
        details = []
        current = padded
        
        for _ in range(self.decomp_levels):
            if current.shape[2] < 2:
                break
            
            even = current[:, :, 0::2, :]
            odd = current[:, :, 1::2, :]
            
            approx = (even + odd) / math.sqrt(2.0)
            detail = (even - odd) / math.sqrt(2.0)
            
            details.append(detail)
            current = approx
            
        return details + [current], orig_len

    def _vectorized_haar_idwt(self, coeffs: List[torch.Tensor], orig_len: int) -> torch.Tensor:
        approx = coeffs[-1]
        details = coeffs[:-1]
        
        current_approx = approx
        for detail in reversed(details):
            if current_approx.shape[2] != detail.shape[2]:
                if current_approx.shape[2] < detail.shape[2]:
                    current_approx = F.pad(current_approx, (0, 0, 0, detail.shape[2] - current_approx.shape[2]))
                else:
                    detail = F.pad(detail, (0, 0, 0, current_approx.shape[2] - detail.shape[2]))
            
            even = (current_approx + detail) / math.sqrt(2.0)
            odd = (current_approx - detail) / math.sqrt(2.0)
            
            batch, heads, half_seq, depth = even.shape
            combined = torch.zeros(batch, heads, half_seq * 2, depth, device=even.device, dtype=even.dtype)
            combined[:, :, 0::2, :] = even
            combined[:, :, 1::2, :] = odd
            current_approx = combined
            
        return current_approx[:, :, :orig_len, :]

    def _filtered_idwt(self, coeffs: List[torch.Tensor], filters: torch.Tensor, orig_len: int) -> torch.Tensor:
        filtered_coeffs = []
        is_per_token = filters.dim() == 3
        
        for i, c in enumerate(coeffs):
            if is_per_token:
                level_filter = filters[:, :, i] * self.scale_weights[i]
                upsampled_filter = F.interpolate(
                    level_filter.unsqueeze(1),
                    size=c.shape[2],
                    mode='nearest'
                ).squeeze(1)
                broadcast_filter = upsampled_filter.unsqueeze(1).unsqueeze(3)
                filtered_coeffs.append(c * broadcast_filter)
            else:
                level_filter = filters[:, i] * self.scale_weights[i]
                level_filter = level_filter.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                filtered_coeffs.append(c * level_filter)
                
        return self._vectorized_haar_idwt(filtered_coeffs, orig_len)

    def _optimized_random_projection(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normalizer = math.sqrt(self.random_features)
        bandwidth = F.softplus(self.log_bandwidth) + 0.1
        
        q_scaled = q / bandwidth
        k_scaled = k / bandwidth
        
        q_proj = F.relu(torch.einsum("bhsd,dr->bhsr", q_scaled, self.random_q_matrix)) / normalizer
        k_proj = F.relu(torch.einsum("bhsd,dr->bhsr", k_scaled, self.random_k_matrix)) / normalizer
        
        return q_proj, k_proj

    def _linear_attention(self, q_proj: torch.Tensor, k_proj: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        if is_causal:
            kv = torch.einsum("bhsr,bhsd->bhrd", k_proj, v)
            kv_cumsum = torch.cumsum(kv, dim=2)
            k_cumsum = torch.cumsum(k_proj, dim=2)
            denominator = torch.einsum("bhsr,bhsr->bhs", q_proj, k_cumsum) + 1e-6
            attention = torch.einsum("bhsr,bhrd->bhsd", q_proj, kv_cumsum) / denominator.unsqueeze(-1)
        else:
            kv = torch.einsum("bhsr,bhsd->bhrd", k_proj, v)
            k_sum = torch.sum(k_proj, dim=2)
            denominator = torch.einsum("bhsr,bhr->bhs", q_proj, k_sum) + 1e-6
            attention = torch.einsum("bhsr,bhrd->bhsd", q_proj, kv) / denominator.unsqueeze(-1)
            
        return attention

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                filter_source: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        """
        q, k, v: (B, H, S, Hd) - already projected and split into heads
        filter_source: (B, S, D) - source for generating filters
        """
        
        # Generate per-token filters using local pooling
        local_features = filter_source.permute(0, 2, 1)  # (B, D, S)
        local_features = self.local_pooler(local_features)
        local_features = local_features.permute(0, 2, 1)  # (B, S, D)
        filters = self.filter_mlp(local_features)  # (B, S, levels+1)
        
        # Apply GELU activation to q and k
        q = F.gelu(q)
        k = F.gelu(k)
        
        # Wavelet decomposition
        q_coeffs, orig_len = self._vectorized_haar_dwt(q)
        k_coeffs, _ = self._vectorized_haar_dwt(k)
        
        # Filtered reconstruction
        q_filtered = self._filtered_idwt(q_coeffs, filters, orig_len)
        k_filtered = self._filtered_idwt(k_coeffs, filters, orig_len)
        
        # Random projection for linear attention
        q_proj, k_proj = self._optimized_random_projection(q_filtered, k_filtered)
        
        # Linear attention
        attention = self._linear_attention(q_proj, k_proj, v, is_causal)
        
        # Post-attention normalization
        attention = self.attention_layer_norm(attention)
        
        return attention
