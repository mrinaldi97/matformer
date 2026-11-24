import json
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin, GenerateDecoderOnlyOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

MATFORMER_CONFIG_FILENAME = "matformer.yaml"
WEIGHTS_FILENAME = "model.safetensors"
MODELING_FILENAME = "modeling_matformer.py"
DEFAULT_CHECKPOINT_NAMES = (WEIGHTS_FILENAME, "checkpoint.ckpt", "checkpoint.pt")
PUBLISH_EXCLUDED_SECTIONS = ("data", "training", "validation", "task", "weights")

_warned = set()


def _warn_once(message):
    if message not in _warned:
        _warned.add(message)
        warnings.warn(message, stacklevel=2)


def _is_classifier_config(model_config):
    """True if a Matformer model config is already a classification config."""
    return model_config is not None and hasattr(model_config, "scope")


def _load_yaml_or_json(path):
    text = Path(path).read_text()
    return json.loads(text) if Path(path).suffix == ".json" else yaml.safe_load(text)


def save_matformer_yaml(configs, path, exclude=PUBLISH_EXCLUDED_SECTIONS):
    """Write configs to a YAML file"""
    data = configs.save_config(fmt="dict", exclude=exclude)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def _auto_map(tasks):
    mapping = {
        "AutoConfig": f"{MODELING_FILENAME[:-3]}.MatformerConfig",
        "AutoModel": f"{MODELING_FILENAME[:-3]}.MatformerModel",
    }
    task_to_auto = {
        "causal": "AutoModelForCausalLM",
        "masked": "AutoModelForMaskedLM",
        "sequence": "AutoModelForSequenceClassification",
        "token": "AutoModelForTokenClassification",
    }
    for task in tasks:
        if task in task_to_auto:
            mapping[task_to_auto[task]] = f"{MODELING_FILENAME[:-3]}.MatformerFor{ {
                'causal': 'CausalLM', 'masked': 'MaskedLM',
                'sequence': 'SequenceClassification', 'token': 'TokenClassification'}[task]}"
    return mapping


class MatformerConfig(PretrainedConfig):
    """
    """

    model_type = "matformer"

    def __init__(self, matformer_config_file=None, checkpoint_file=None, **kwargs):
        self.matformer_config_file = matformer_config_file or MATFORMER_CONFIG_FILENAME
        self.checkpoint_file = checkpoint_file or WEIGHTS_FILENAME
        super().__init__(**kwargs)

    @classmethod
    def from_matformer_configs(cls, configs, **overrides):
        """Build the light HF config from a MatformerConfigs tree."""
        config = cls(**overrides)
        return config.sync_from_matformer(configs)

    def sync_from_matformer(self, configs):
        """Copy standard fields from a MatformerConfigs tree into this config."""
        model = getattr(configs, "model", None)
        if model is not None:
            for key in ("hidden_size", "vocab_size", "num_hidden_layers",
                        "tie_word_embeddings", "max_position_embeddings", "is_causal"):
                value = getattr(model, key, None)
                if value is not None:
                    setattr(self, key, value)
            mixer_config = getattr(getattr(model, "default_layer", None), "sequence_mixer_config", None)
            if mixer_config is not None:
                heads = mixer_config.get("num_attention_heads") if hasattr(mixer_config, "get") \
                    else getattr(mixer_config, "num_attention_heads", None)
                if heads is not None:
                    self.num_attention_heads = heads
            num_classes = getattr(model, "num_classes", None)
            if num_classes is not None:
                self.num_labels = num_classes
            self.matformer_model_name = getattr(model, "name", None)
        tokenizer = getattr(configs, "tokenizer", None)
        if tokenizer is not None:
            for key in ("pad_token_id", "bos_token_id", "eos_token_id", "mask_token_id"):
                value = getattr(tokenizer, key, None)
                if value is not None:
                    setattr(self, key, value)
            if getattr(tokenizer, "pretrained_name", None):
                self.tokenizer_name_or_path = tokenizer.pretrained_name
        return self


def _to_tensor_dc(input_ids, attention_mask):
    """Normalize HF-style (input_ids, attention_mask) into a Matformer TensorDC.

    A real attention_mask becomes a PaddedTensor (padding_mask semantics belong to
    Matformer), so pooling and losses inside the framework see it natively.
    """
    from matformer.dataclasses.tensors_dataclasses import NormalTensor, PaddedTensor

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        attention_mask = attention_mask.to(input_ids.device)
        if attention_mask.shape == input_ids.shape and (attention_mask == 0).any():
            return PaddedTensor(tensor=input_ids, padding_mask=(attention_mask == 0))
    return NormalTensor(tensor=input_ids)


class MatformerPreTrainedModel(PreTrainedModel):
    config_class = MatformerConfig
    base_model_prefix = "matformer_model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _no_split_modules = ["TransformerBlock"]
    # Which HF task this class exposes: None, "causal", "masked", "sequence", "token"
    matformer_task = None

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.matformer_model = None
        self.matformer_configs = None

    def _init_weights(self, module):
        # Matformer owns initialization (registry kernels + init_transformer_weights_).
        pass

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """HF-compatible entry point that delegates loading to Matformer.

        Extra kwargs beyond the HF ones:
            checkpoint: path/name of the weights file (overrides config/check auto-detect)
            matformer_config: path to a YAML/JSON file, or a dict of Matformer configs
            num_labels / pooling_type: used when adding a fresh classification head
        """
        from matformer.loading.utils import make_holder
        from matformer.matformer_configs import MatformerConfigs
        from matformer.matformer_wrapper import load_model

        config = kwargs.pop("config", None)
        matformer_config = kwargs.pop("matformer_config", None)
        checkpoint = kwargs.pop("checkpoint", None)
        num_labels = kwargs.pop("num_labels", None)
        pooling_type = kwargs.pop("pooling_type", None)
        device = kwargs.pop("device", None)
        device_map = kwargs.pop("device_map", None)
        dtype = kwargs.pop("dtype", None)
        if dtype is None:
            dtype = kwargs.pop("torch_dtype", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        cache_dir = kwargs.pop("cache_dir", None)
        # Purely HF-side kwargs that mean nothing to Matformer: accept and ignore.
        for ignored in ("trust_remote_code", "attn_implementation", "low_cpu_mem_usage",
                        "use_safetensors", "offload_folder", "offload_state_dict",
                        "max_shard_size", "_fast_init"):
            kwargs.pop(ignored, None)

        map_location = cls._resolve_device(device, device_map)
        dtype = cls._resolve_dtype(dtype)

        repo_dir, ckpt_path = cls._resolve_paths(
            pretrained_model_name_or_path, checkpoint,
            token=token, revision=revision, cache_dir=cache_dir)
        if config is None:
            if repo_dir is not None and (repo_dir / "config.json").exists():
                config = MatformerConfig.from_pretrained(repo_dir)
            else:
                config = MatformerConfig()
        if ckpt_path is None:
            ckpt_path = cls._find_checkpoint(repo_dir, config.checkpoint_file)

        base_configs = MatformerConfigs()
        if isinstance(matformer_config, dict):
            base_configs.import_config(matformer_config)
        else:
            yaml_path = matformer_config or (repo_dir / config.matformer_config_file
                                             if repo_dir else None)
            if yaml_path is None and repo_dir is not None:
                yamls = sorted(repo_dir.glob("*.yaml"))
                if len(yamls) == 1:
                    yaml_path = yamls[0]
            if yaml_path is not None:
                if not Path(yaml_path).exists():
                    raise FileNotFoundError(f"Matformer config not found: {yaml_path}")
                raw = _load_yaml_or_json(yaml_path)
                if isinstance(raw.get("model"), dict) and raw["model"].get("type") == "language_model":
                    raw["model"]["type"] = "default"  # see save_matformer_yaml()
                base_configs.import_config(raw)

        holder = make_holder(str(ckpt_path), map_location=map_location) if ckpt_path else None
        if holder is not None and base_configs.list_configs():
            holder.configs = base_configs
        ckpt_configs = holder.get_configs() if holder else base_configs

        overrides = MatformerConfigs()
        if repo_dir is not None and any(
                (repo_dir / f).exists() for f in ("tokenizer_config.json", "tokenizer.json")):
            overrides.import_config({"tokenizer": {"pretrained_name": str(repo_dir)}})
        overrides = cls._apply_task(overrides, ckpt_configs, config,
                                    num_labels=num_labels, pooling_type=pooling_type)
        if holder is None:
            ckpt_configs.import_config(overrides.asdict(include_defaults=False))
            overrides = MatformerConfigs()

        model_dict = load_model(checkpoint_holder=holder, configs=overrides,
                                map_location=map_location, dtype=dtype, inference=True,
                                inherit_sections=("model", "tokenizer", "masking"))

        instance = cls(config)
        instance._install(model_dict)
        return instance

    def _install(self, model_dict):
        """Attach a freshly loaded MatformerModelWrapper to this HF shell."""
        self.matformer_model = model_dict["model"]
        self.matformer_configs = model_dict["configs"]
        self.config.sync_from_matformer(model_dict["configs"])
        self.post_init()
        return self

    @classmethod
    def from_matformer(cls, wrapper, configs=None, config=None):
        """Wrap an already-loaded MatformerModelWrapper (e.g. from load_model)."""
        config = config or MatformerConfig()
        instance = cls(config)
        instance._install({"model": wrapper, "configs": configs or wrapper.configs})
        return instance

    # ------------------------------------------------------------- path helpers
    @staticmethod
    def _resolve_device(device, device_map):
        if device is not None:
            return str(device)
        if device_map is not None:
            if isinstance(device_map, str) and device_map not in ("auto", "balanced", "sequential"):
                return device_map
            _warn_once(
                "Matformer's from_pretrained does not shard models: 'device_map' is "
                "ignored, use 'device' (or .to()) instead.")
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_dtype(dtype):
        if dtype is None or dtype == "auto":
            return torch.bfloat16
        if isinstance(dtype, str):
            return getattr(torch, dtype)
        return dtype

    @staticmethod
    def _resolve_paths(name, checkpoint=None, token=None, revision=None, cache_dir=None):
        """Return (repo_dir, checkpoint_path) for dirs, bare weight files and hub repo ids."""
        path = Path(checkpoint) if checkpoint else None
        source = Path(name)
        if source.is_dir():
            return source, (path if path and path.exists() else None)
        if source.is_file():
            # A bare checkpoint file: its folder is NOT treated as a repo, so stray
            # YAML/tokenizer files next to it can't hijack the load.
            return None, source
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(str(name), token=token, revision=revision,
                                          cache_dir=cache_dir))
        return snapshot, (path if path and path.exists() else None)

    @staticmethod
    def _find_checkpoint(repo_dir, explicit=None):
        if repo_dir is None:
            return None
        candidates = [explicit] if explicit else []
        candidates += list(DEFAULT_CHECKPOINT_NAMES)
        for candidate in candidates:
            if candidate and (repo_dir / candidate).is_file():
                return repo_dir / candidate
        for pattern in ("*.ckpt", "*.safetensors", "*.pt"):
            found = sorted(repo_dir.glob(pattern))
            if found:
                return found[0]
        return None

    # ------------------------------------------------------------- task glue
    @classmethod
    def _apply_task(cls, configs, ckpt_configs, config, num_labels=None, pooling_type=None):
        """Adapt the Matformer configs to the HF task being requested.

        Classification over a non-classification checkpoint: switch the model
        section to Matformer's own 'classification' variant (the framework's
        ClassificationPlugin provides the head, nothing is reimplemented here).
        If the checkpoint is already a classifier, its configuration wins unless
        the user explicitly passed num_labels.
        """
        if cls.matformer_task not in ("sequence", "token"):
            return configs
        base_model = getattr(ckpt_configs or configs, "model", None)
        already_classifier = _is_classifier_config(base_model)
        if already_classifier and num_labels is None and pooling_type is None:
            return configs
        override = {"type": "classification", "scope": cls.matformer_task}
        labels = num_labels if num_labels is not None else getattr(config, "num_labels", None)
        if labels is not None:
            override["num_classes"] = labels
        if pooling_type is not None:
            override["pooling_type"] = pooling_type
        configs.import_config({"model": override})
        return configs

    # ------------------------------------------------------------------ access
    @property
    def matformer_tokenizer(self):
        """The MatformerTokenizer attached to the wrapper, if any."""
        return getattr(self.matformer_model, "tokenizer", None)

    def get_tokenizer(self):
        """The underlying HuggingFace tokenizer (usable exactly like a HF one)."""
        tokenizer = self.matformer_tokenizer
        return getattr(tokenizer, "tokenizer", tokenizer)

    def get_input_embeddings(self):
        node = getattr(self.matformer_model, "model", None)
        while node is not None:
            embed = getattr(node, "embed_tokens", None)
            if embed is not None:
                return embed
            node = getattr(node, "encoder", None)
        return None

    def get_output_embeddings(self):
        node = getattr(self.matformer_model, "model", None)
        while node is not None:
            head = getattr(node, "lm_head", None)
            if head is not None:
                return head
            node = getattr(node, "encoder", None)
        return None

    def resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        raise NotImplementedError(
            "Matformer models are resized through their own configs (model.vocab_size).")

    # ---------------------------------------------------------------- forward
    # Forward kwargs Matformer models understand; anything HF/Trainer injects
    # (num_items_in_batch, return_dict, ...) is filtered out here.
    _matformer_forward_kwargs = ("return_type", "token_type_ids", "is_causal", "sliding",
                                 "block_mask", "document_mask", "recurrence_mask",
                                 "start", "stop")

    def _run(self, input_ids, attention_mask=None, **kwargs):
        """One Matformer forward, returning a plain logits/hidden tensor."""
        if input_ids is None:
            raise ValueError("input_ids is required")
        mat_kwargs = {k: kwargs[k] for k in self._matformer_forward_kwargs if k in kwargs}
        x = _to_tensor_dc(input_ids, attention_mask)
        out = self.matformer_model(x, **mat_kwargs)
        return getattr(out, "tensor", out)

    # ----------------------------------------------------------------- saving
    def save_pretrained(self, save_directory, save_weights=True, save_tokenizer=True, **kwargs):
        """Write a self-contained HF repo: config.json + matformer.yaml + weights.

        Weights are exported in Matformer's stable-state-dict format as
        model.safetensors, which MatformerCheckpointHolder reads natively, so the
        directory is loadable by both frameworks.
        """
        from safetensors.torch import save_file

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        configs = self.matformer_model.configs

        if save_weights:
            save_file(self._clean_state_dict(self.matformer_model.stable_state_dict()),
                      str(save_directory / WEIGHTS_FILENAME))
        save_matformer_yaml(configs, save_directory / MATFORMER_CONFIG_FILENAME)
        self.config.matformer_config_file = MATFORMER_CONFIG_FILENAME
        self.config.checkpoint_file = WEIGHTS_FILENAME
        if not getattr(self.config, "auto_map", None):
            task = self.matformer_task or "masked"
            self.config.auto_map = _auto_map([task])
        # Write config.json directly: PretrainedConfig.save_pretrained would also copy this
        # module next to the model and rewrite auto_map to the copy, which cannot work
        # standalone (it needs the full matformer package). The shim in modeling_matformer.py
        # points at the installed package instead.
        (save_directory / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n")
        from matformer.loading.huggingface_publish import create_modeling_file
        (save_directory / MODELING_FILENAME).write_text(create_modeling_file())
        if save_tokenizer:
            tokenizer = self.get_tokenizer()
            if tokenizer is not None:
                tokenizer.save_pretrained(save_directory)
        return save_directory

    @staticmethod
    def _clean_state_dict(state_dict):
        """Tensors only, on CPU, deduplicated (tied weights share storage in Matformer)."""
        cleaned, seen = {}, set()
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            if value.data_ptr() in seen:
                continue
            seen.add(value.data_ptr())
            cleaned[key] = value.detach().cpu().contiguous()
        return cleaned


class MatformerModel(MatformerPreTrainedModel):
    """Base Matformer model exposed as an HF model returning hidden states."""

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        kwargs.setdefault("return_type", "hidden")
        hidden_states = self._run(input_ids, attention_mask, **kwargs)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class MatformerForMaskedLM(MatformerPreTrainedModel):
    """Masked language modeling head = Matformer's own lm_head, nothing added."""

    matformer_task = "masked"

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        logits = self._run(input_ids, attention_mask, **kwargs)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return MaskedLMOutput(loss=loss, logits=logits)


class MatformerForCausalLM(MatformerPreTrainedModel, GenerationMixin):
    """Causal LM. Generation is delegated to Matformer's own generate
    (transformer_blocks/mixin/text_generation.py): no KV-cache, no beam search,
    but full sampling control (temperature, top-k/top-p, repetition penalty...).
    """

    matformer_task = "causal"

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        logits = self._run(input_ids, attention_mask, **kwargs)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @staticmethod
    def _prompt_len(input_ids):
        if input_ids is None:
            return 0
        if isinstance(input_ids, torch.Tensor):
            return input_ids.shape[-1]
        if len(input_ids) and isinstance(input_ids[0], (list, tuple)):
            return len(input_ids[0])
        return len(input_ids)

    def generate(self, inputs=None, input_ids=None, attention_mask=None,
                 max_new_tokens=None, max_length=None, min_new_tokens=None, min_length=None,
                 do_sample=None, temperature=None, top_k=None, top_p=None,
                 repetition_penalty=None, no_repeat_ngram_size=None,
                 num_return_sequences=None, pad_token_id=None, eos_token_id=None,
                 bos_token_id=None, stopping_criteria=None, stream_callback=None,
                 output_scores=False, return_dict_in_generate=None, **kwargs):
        engine = getattr(self.matformer_model.model, "generate", None)
        if engine is None:
            raise NotImplementedError(
                "The underlying Matformer model does not support generation "
                "(no text-generation mixin in its registry variant).")
        if input_ids is None:
            input_ids = inputs
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor) \
                and (attention_mask == 0).any():
            _warn_once("Matformer generate ignores attention_mask: padded prompts may "
                       "produce incorrect continuations.")
        for name in ("use_cache", "cache_position", "position_ids", "past_key_values"):
            kwargs.pop(name, None)
        num_beams = kwargs.pop("num_beams", 1)
        if num_beams and num_beams > 1:
            _warn_once("Matformer generate does not support beam search: 'num_beams' ignored.")
        for name in ("length_penalty", "early_stopping", "beam_idx", "num_beam_groups"):
            if kwargs.pop(name, None) is not None:
                _warn_once(f"Matformer generate does not support '{name}': ignored.")
        if kwargs:
            _warn_once(f"Ignored generate kwargs unsupported by Matformer: {sorted(kwargs)}")

        gen_kwargs = {}
        for name, value in (("min_new_tokens", min_new_tokens), ("do_sample", do_sample),
                            ("temperature", temperature), ("top_k", top_k), ("top_p", top_p),
                            ("repetition_penalty", repetition_penalty),
                            ("no_repeat_ngram_size", no_repeat_ngram_size),
                            ("num_return_sequences", num_return_sequences)):
            if value is not None:
                gen_kwargs[name] = value
        prompt_len = self._prompt_len(input_ids)
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        elif max_length is not None and input_ids is not None:
            gen_kwargs["max_new_tokens"] = max(max_length - prompt_len, 0)
        if min_new_tokens is None and min_length is not None and input_ids is not None:
            gen_kwargs["min_new_tokens"] = max(min_length - prompt_len, 0)
        for name, explicit in (("pad_token_id", pad_token_id),
                               ("eos_token_id", eos_token_id),
                               ("bos_token_id", bos_token_id)):
            gen_kwargs[name] = explicit if explicit is not None else getattr(self.config, name, None)

        want_scores = bool(output_scores) or bool(return_dict_in_generate)
        result = engine(input_ids=input_ids, output_scores=want_scores,
                        stopping_criteria=stopping_criteria, stream_callback=stream_callback,
                        **gen_kwargs)
        if isinstance(result, tuple):
            sequences = result[0]
            scores = tuple(torch.unbind(result[1], dim=1)) if len(result) > 1 else None
        else:
            sequences, scores = result, None
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=sequences, scores=scores)
        if output_scores:
            return sequences, scores
        return sequences


class MatformerForSequenceClassification(MatformerPreTrainedModel):
    """Sequence classification through Matformer's classification plugin
    (registry variant 'classification', scope 'sequence'): pooling and head
    belong to the framework, so heads saved by Matformer load natively.
    """

    matformer_task = "sequence"

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        logits = self._run(input_ids, attention_mask, **kwargs)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if labels.dtype in (torch.int32, torch.int64) and logits.shape[:-1] == labels.shape:
                loss = F.cross_entropy(logits, labels.long())
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
        return SequenceClassifierOutput(loss=loss, logits=logits)


class MatformerForTokenClassification(MatformerPreTrainedModel):
    """Token classification through Matformer's classification plugin
    (registry variant 'classification', scope 'token').
    """

    matformer_task = "token"

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        logits = self._run(input_ids, attention_mask, **kwargs)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if labels.dtype in (torch.int32, torch.int64):
                loss = F.cross_entropy(
                    logits.transpose(1, 2).reshape(-1, logits.size(-1)),
                    labels.reshape(-1), ignore_index=-100)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
        return TokenClassifierOutput(loss=loss, logits=logits)


def register_matformer():
    """Register Matformer with HF's Auto* classes (safe to call multiple times)."""
    from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                              AutoModelForMaskedLM, AutoModelForSequenceClassification,
                              AutoModelForTokenClassification)

    registrations = (
        (lambda: AutoConfig.register("matformer", MatformerConfig)),
        (lambda: AutoModel.register(MatformerConfig, MatformerModel)),
        (lambda: AutoModelForCausalLM.register(MatformerConfig, MatformerForCausalLM)),
        (lambda: AutoModelForMaskedLM.register(MatformerConfig, MatformerForMaskedLM)),
        (lambda: AutoModelForSequenceClassification.register(MatformerConfig, MatformerForSequenceClassification)),
        (lambda: AutoModelForTokenClassification.register(MatformerConfig, MatformerForTokenClassification)),
    )
    for register in registrations:
        try:
            register()
        except ValueError:
            pass  # already registered in this process
