import sys
sys.path.append('../') # Da sostituire con pyproject.toml eccetera ok per ora
import torch
import torch.nn as nn
from dataclasses import replace
from matformer.transformer_blocks import NakedTransformer, PackedSwiGLUFFN, RMSNorm, MultiHeadAttention
from matformer.tensors_dataclasses import ModuleWrapper
from matformer.utils import PositionalEncodingPermute1D
from matformer.model_config import ModelConfig

class TransCharAutoencoder_Encoder(nn.Module):
    def __init__(self, config: ModelConfig, device='cuda'):
        super().__init__()
        self.pos_enc = ModuleWrapper(PositionalEncodingPermute1D(config.max_position_embeddings))
        self.embed_chars = ModuleWrapper(nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id))
        self.embed_seq_lens = ModuleWrapper(nn.Embedding(config.max_position_embeddings, config.hidden_size))
        self.encoder_transformer = NakedTransformer(config)
        self.normalization = ModuleWrapper(RMSNorm(config.hidden_size))
        self.mlp = PackedSwiGLUFFN(hidden_size=config.hidden_size, ffn_factor=1.0)
        self.attn = MultiHeadAttention(
            q_dim=config.hidden_size, k_dim=config.hidden_size, v_dim=config.hidden_size,
            hidden_size=config.hidden_size, nheads=config.num_attention_heads,
            bias=False, block_mask=None, attn_impl=config.attn_impl,
            alibi=False, is_causal=False, device=device
        )
        self.device = device
        self.to(device)

    def forward(self, input_ids, lengths):
        chars_emb = self.embed_chars(input_ids)              # (B,S,D)
        len_emb = self.embed_seq_lens(lengths)              # (B,1,D)
        p_enc = self.pos_enc(chars_emb)
        chars_emb = chars_emb + p_enc
        chars_emb = self.encoder_transformer(chars_emb) + chars_emb
        pooled = chars_emb.tensor.mean(-2)
        pooled = replace(chars_emb, tensor=pooled.unsqueeze(1))
        pooled = pooled + len_emb
        pooled = self.normalization(pooled)
        bottleneck = self.attn(query_input=pooled, key_input=chars_emb, value_input=chars_emb) + pooled
        bottleneck = self.mlp(bottleneck) + bottleneck
        return bottleneck

class TransCharAutoencoder_Decoder(nn.Module):
    def __init__(self, config: ModelConfig, device='cuda'):
        super().__init__()
        self.decoder_transformer = NakedTransformer(config)
        self.pos_enc = ModuleWrapper(PositionalEncodingPermute1D(config.max_position_embeddings))
        self.cross_attn = MultiHeadAttention(
            q_dim=config.hidden_size, k_dim=config.hidden_size, v_dim=config.hidden_size,
            hidden_size=config.hidden_size, nheads=config.num_attention_heads,
            bias=False, block_mask=None, attn_impl=config.attn_impl,
            alibi=False, is_causal=False, device=device
        )
        self.char_head = ModuleWrapper(nn.Linear(config.hidden_size, config.vocab_size))
        self.seqlen_guesser = ModuleWrapper(nn.Linear(config.hidden_size, config.max_position_embeddings))
        self.config = config
        self.device = device
        self.to(device)

    def forward(self, z, lengths, real_mode=False):
        B, _, D = z.tensor.size()
        guessed_seqlen_logits = self.seqlen_guesser(z)
        guessed_seqlen_logits = replace(guessed_seqlen_logits, tensor=guessed_seqlen_logits.tensor.squeeze(1))
        guessed_seqlen_value = guessed_seqlen_logits.tensor.argmax(-1).clamp(min=1)
        
        target_lengths = guessed_seqlen_value if real_mode else lengths.tensor.squeeze(-1)
        max_len = target_lengths.max().item()
        expanded_z = z.tensor.expand(-1, max_len, -1)
        expanded_z = replace(z, tensor=expanded_z)
        p_enc = self.pos_enc(expanded_z)
        x = expanded_z + p_enc
        x = self.cross_attn(query_input=x, key_input=z, value_input=z) + x
        out = self.decoder_transformer(x)
        char_logits = self.char_head(out)
        return char_logits, guessed_seqlen_logits

class TransCharAutoencoder(nn.Module):
    """Combined encoder-decoder model."""
    def __init__(self, encoder_config: ModelConfig, decoder_config: ModelConfig, device='cuda'):
        super().__init__()
        self.encoder = TransCharAutoencoder_Encoder(encoder_config, device=device)
        self.decoder = TransCharAutoencoder_Decoder(decoder_config, device=device)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
    
    def forward(self, input_ids, lengths, real_mode=False):
        z = self.encoder(input_ids, lengths)
        return self.decoder(z, lengths, real_mode=real_mode)
