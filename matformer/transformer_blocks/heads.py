"""
File: matformer/transformer_blocks/heads.py
"""
import sys
sys.path.append('../')
#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#Matformer
from matformer.matformer_module import MatformerModule
from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor
from matformer.dataclasses.model_dataclasses import ModelConfig  
from matformer.utils.matformer_cache import ensure_cache, CachedStuff
from matformer.matformer_registry import registry
from matformer.transformer_blocks import NakedTransformer
#Other
from dataclasses import replace
from warnings import warn
from typing import Optional, List, Literal, Any

class AddPositions:
    def __init__(self, embed_positions, offset=0): self.embed, self.offset = embed_positions, offset
    def __call__(self, embeddings, **kw):
        if isinstance(embeddings, UnpaddedTensor):
            # packed layout (total_tokens, D): exact per-token positions from `indices`
            # (the original flat (b*s) position of each packed token) — modulo the row
            # length this is the token's original column, for ANY padding pattern.
            # NB: `indices` may sit on CPU even when the data is on GPU.
            ids = (embeddings.indices % embeddings.original_seq_len + self.offset).long().to(embeddings.tensor.device)
        else:
            batch, seq = embeddings.tensor.shape[:2]
            ids = torch.arange(seq, dtype=torch.long, device=embeddings.tensor.device)
            ids = (ids + self.offset).unsqueeze(0).expand(batch, -1)   # RoBERTa-style offset: positions start at padding_idx+1
        pos = self.embed(replace(embeddings, tensor=ids))
        return replace(embeddings, tensor=embeddings.tensor + pos.tensor)


class AddTokenType:
    """Adds segment ("token type") embeddings — BERT-style sentence A/B distinction.
    token_type_ids defaults to all zeros (single segment), like HF's default."""
    def __init__(self, embed_token_type): self.embed = embed_token_type
    def __call__(self, embeddings, token_type_ids=None, **kw):
        if token_type_ids is None:
            # may ride along as a following_attribute (already gathered by unpad())
            token_type_ids = embeddings.following_attributes.get("token_type_ids")
        if token_type_ids is None:
            shape = embeddings.tensor.shape[:1] if isinstance(embeddings, UnpaddedTensor) \
                else embeddings.tensor.shape[:2]
            token_type_ids = torch.zeros(shape, dtype=torch.long, device=embeddings.tensor.device)
        elif isinstance(embeddings, UnpaddedTensor) and token_type_ids.dim() > 1:
            # padded (b, s) ids on the packed stream: gather the packed tokens' original columns
            token_type_ids = token_type_ids.reshape(-1)[embeddings.indices].to(embeddings.tensor.device)
        types = self.embed(replace(embeddings, tensor=token_type_ids))
        return replace(embeddings, tensor=embeddings.tensor + types.tensor)


class ScaleEmbeddings:
    """Scales the token embeddings by a constant — Gemma-style normalizer
    (config.embedding_scale, typically derived as hidden_size ** 0.5)."""
    def __init__(self, embed_tokens, scale): self.embed, self.scale = embed_tokens, scale
    def __call__(self, ids, **kw):
        out = self.embed(ids, **kw)
        return replace(out, tensor=out.tensor * self.scale)


class TransformerWithEmbeddingHead(MatformerModule):
    # Stable parameter names
    embed_tokens: "param_name:embed_tokens"
    embed_positions: "param_name:embed_positions"
    embed_token_type: "param_name:embed_token_type"
    embeddings_norm: "param_name:embeddings_norm"
    blocks: "param_name:blocks"

    def __init__(self, config: ModelConfig, cache=None):
        super().__init__()
        self.config = config
        self.cache = ensure_cache(cache)
        self.init_sockets(stream_type=config.stream_type)
        # Token embeddings (optionally scaled by config.embedding_scale, Gemma-style)
        self.embed_tokens = registry.create(
            "embedding", "embedding", num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size, padding_idx=config.tokenizer.pad_token_id)
        scale = getattr(config, "embedding_scale", None)
        if scale:
            self.register_socket("embed_tokens", ScaleEmbeddings(self.embed_tokens, float(scale)))
        else:
            self.register_socket("embed_tokens", self.embed_tokens, attr="embed_tokens")

        # Learnable positional embeddings (if configured)
        learnable = "learnable" in config.default_layer.sequence_mixer_config.positional_encoding
        self.embed_positions = registry.create(
            "embedding", "embedding", num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size) if learnable else None
        if learnable:
            self.register_socket("embed_positions", AddPositions(
                self.embed_positions, offset=getattr(config, "position_embedding_offset", 0) or 0))
        else:
            self.add_socket("embed_positions")            # empty = identity

        # Segment ("token type") embeddings — BERT-style (if configured)
        if config.type_vocab_size:
            self.embed_token_type = registry.create(
                "embedding", "embedding", num_embeddings=config.type_vocab_size,
                embedding_dim=config.hidden_size)
            self.register_socket("embed_token_type", AddTokenType(self.embed_token_type))
        else:
            self.add_socket("embed_token_type")           # empty = identity

        # Norm applied to the embedded stream (word + position + type), BERT-style
        if getattr(config, "embedding_norm", False):
            # bias only when explicitly disabled: RMSNorm kernels take no bias argument
            norm_kwargs = {"normalized_shape": config.hidden_size,
                           "eps": config.rms_norm_eps, "elementwise_affine": True}
            if not getattr(config, "norm_bias", True):
                norm_kwargs["bias"] = False
            self.embeddings_norm = registry.create(
                "norm", config.default_layer["normalization"], **norm_kwargs)
            self.register_socket("embeddings_norm", self.embeddings_norm)
        else:
            self.add_socket("embeddings_norm")            # empty = identity

        self.add_socket("model_input")
        # Core transformer
        self.register_socket("blocks", NakedTransformer(config, cache=self.cache), attr="blocks")
    def forward(self, x, **kwargs):
        """Embed tokens (+ positions + token types) then process through transformer."""
        self.stream.init(x)
        self.stream.apply("model_input")
        self.stream.apply("embed_tokens")
        self.stream.apply("embed_positions")
        self.stream.apply("embed_token_type", token_type_ids=kwargs.get("token_type_ids"))
        self.stream.apply("embeddings_norm")
        self.stream.apply("blocks", **kwargs)
        return self.stream.exit()

    def add_positional_embeddings(self, x, embeddings):
        batch_size, seq_len = x.tensor.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.tensor.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_ids_wrapped = replace(x, tensor=position_ids)
        position_embeds = self.embed_positions(position_ids_wrapped)
        embeddings = replace(embeddings, tensor=embeddings.tensor + position_embeds.tensor)


class LMHeadTransform(nn.Module):
    """HF BERT-style MLM head transform: dense -> gelu -> LayerNorm, applied to the
    final hidden states before the decoder (cls.predictions.transform on HF side)."""
    # Stable parameter names
    dense: "param_name:dense"
    norm: "param_name:norm"

    def __init__(self, hidden_size, eps, bias=True, norm_bias=True):
        super().__init__()
        self.dense = registry.create("linear", "linear", in_features=hidden_size,
                                     out_features=hidden_size, bias=bias)
        self.norm = registry.create("norm", "layernorm", normalized_shape=hidden_size,
                                    eps=eps, elementwise_affine=True, bias=norm_bias)

    def forward(self, x):
        x = self.dense(x)
        x = x.apply(F.gelu)
        return self.norm(x)


class TransformerWithLMHead(MatformerModule):
    """
    Complete language model with embeddings and language modeling head.
    Suitable for autoregressive (GPT-like) or masked (BERT-like) language models.
    Supports weight tying between embeddings and output projection.
    """
    
    # Stable parameter names
    lm_head: "param_name:lm_head"
    lm_head_transform: "param_name:lm_head_transform"
    encoder: "param_name:encoder"
    
    def __init__(self, config: ModelConfig, device=None, cache=None, **kwargs):
        super().__init__()
        self.config = config
        self.cache = ensure_cache(cache)
        self.init_sockets(stream_type=config.stream_type)
        # Optional transform before the output projection (BERT-style MLM heads)
        if getattr(config, "lm_head_transform", False):
            self.register_socket("lm_head_transform", LMHeadTransform(
                config.hidden_size, config.rms_norm_eps,
                bias=getattr(config, "lm_head_transform_bias", True),
                norm_bias=getattr(config, "norm_bias", True)),
                attr="lm_head_transform")
        else:
            self.add_socket("lm_head_transform")     # empty = identity
        # Language modeling head (output projection)
        self.register_socket("lm_head", registry.create(
            "linear", "linear", in_features=config.hidden_size, out_features=config.vocab_size,
            bias=getattr(config, "lm_head_bias", True)),
            attr="lm_head")
        # Transformer with embeddings
        self.register_socket("encoder", TransformerWithEmbeddingHead(config, cache=self.cache),
                             attr="encoder")
        
        ################## WARNING !!! ##################
        # Due to a bug discovered too late, currently weight tying is allowed only
        # with the environment variable ALLOW_WEIGHT_TYING set to TRUE
        # This is extremely temporary: we have several checkpoints trained with
        # weight tying set up in the config but that actually didn't worked
        # until we fix the configurations file present in those checkpoints
        # for easiness the weight tying is disabled
        # Be aware of this if you are going to train a model with Matformer!
        ################## WARNING !!! ##################
        # Weight tying: share embeddings with output projection
        import os 
        allow_tying = os.environ.get("ALLOW_WEIGHT_TYING", "true").lower() == "true" # necessary for a legacy bug

        if config.tie_word_embeddings and allow_tying:
            self.lm_head.inner.weight = self.encoder.embed_tokens.inner.weight
            
    def forward(self, x, return_type='logits', **kwargs):
        """Forward pass with optional return type.
        
        Args:
            x: Input token IDs
            return_type: 'logits' for language modeling head output,
                        'hidden' for transformer hidden states
        """
        self.stream.init(x)
        self.stream.apply("encoder", **kwargs)
        self.stream.apply("lm_head_transform")
        if return_type == "logits":
            self.stream.apply("lm_head")
            cap = getattr(self.config, "final_logit_softcap", None)
            if cap:
                self.stream.x = cap * torch.tanh(self.stream.x / cap)
        return self.stream.exit()
        

class MultiModalModel(MatformerModule):
    def __init__(self, config: ModelConfig, device=None, cache=None):
        self.router=None
        pass
    def forward(self, x, modality):
        # 1. "Preludio"
        x=self.preludi[modality](x)
        # 2. "Atto"
        x=self.atto(x)
        # 3. Router
        router_output=router(x)
        chosen_modality=torch.argmax(router_output)

        # 4. "Coda"
        x=self.code[modality](x)

        return x, chosen_modality
        pass
