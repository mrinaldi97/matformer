"""
matformer/huggingface_integration/hf_model_loader.py
    hf_config_to_matformer(config_path)  ->  ModelConfig
    get_hf_repo(model_id)                ->  (config_path, [weight_file_paths])
    load_hf_weights(model, config_path)  ->  (missing, unexpected)
"""
import json, re
from pathlib import Path
import yaml
from matformer.model_config import ModelConfig, LayerConfig

_DEFS = Path(__file__).parent / "adapters"



def expand(template, N):
    """Expand N-templated stable-name patterns into per-layer keys."""
    out = []
    for item in template:
        if "N" in item.split("."):
            for i in range(N): out.append(item.replace("N", str(i)))
        else:
            out.append(item)
    return out

def collapse(_list):
    """Collapse numbered keys back to N-templates (inverse of expand)."""
    return list({".".join(re.sub(r"\d+", "N", p) for p in item.split("."))
                 for item in _list})


def _load_definition(model_type):
    path = _DEFS / f"{model_type}.yaml"
    if not path.exists():
        raise NotImplementedError(
            f"No Matformer definition for '{model_type}'. "
            f"Supported: {[p.stem for p in _DEFS.glob('*.yaml')]}"
        )
    with open(path) as f:
        return yaml.safe_load(f)

def _build_weight_map(defn, num_layers):
    """Expand N-templates from the YAML weight_mapping into a flat dict
    {hf_weight_name: matformer_stable_name}."""
    raw = defn.get("weight_mapping", {})
    result = {}
    for mat_tmpl, hf_tmpl in raw.items():
        if "N" in mat_tmpl:
            for i in range(num_layers):
                result[hf_tmpl.replace("N", str(i))] = mat_tmpl.replace("N", str(i))
        else:
            result[hf_tmpl] = mat_tmpl
    return result



def hf_config_to_matformer(config_path: str) -> ModelConfig:
    """Load a HuggingFace config.json and return a Matformer ModelConfig."""
    with open(config_path) as f:
        hf = json.load(f)
    model_type = hf.get("model_type")
    if not model_type:
        raise ValueError("config.json has no 'model_type' key.")

    defn = _load_definition(model_type)
    kwargs, layer_overrides = {}, {}

    for k, v in defn.get("defaults", {}).items():
        if k == "default_layer": layer_overrides.update(v)
        else: kwargs[k] = v

    for hf_key, mat_key in defn.get("mapping", {}).items():
        if hf_key in hf: kwargs[mat_key] = hf[hf_key]

    for mat_key, expr in defn.get("derived", {}).items():
        value = eval(str(expr), {}, hf)
        if mat_key.startswith("default_layer."): layer_overrides[mat_key[14:]] = value
        else: kwargs[mat_key] = value

    kwargs["default_layer"] = LayerConfig(**layer_overrides)
    return ModelConfig(**kwargs)


def get_hf_repo(model_id: str):
    """Download (or use cached) a HF repo. Returns (config_path, [weight_paths])."""
    from huggingface_hub import snapshot_download
    folder = Path(snapshot_download(model_id))
    config_path = folder / "config.json"
    assert config_path.exists(), f"No config.json found in {folder}"

    index_files = list(folder.glob("*.index.json"))
    if index_files:
        with open(index_files[0]) as f:
            index = json.load(f)
        weight_files = sorted(set(index["weight_map"].values()))
    else:
        weight_files = ([f.name for f in folder.glob("*.safetensors")]
                        or [f.name for f in folder.glob("*.bin")])

    return str(config_path), [str(folder / w) for w in weight_files]


def load_hf_weights(model, config_path: str):
    """Load HuggingFace weights into a Matformer PL_ModelWrapper (or any
    ParametersRenamer subclass) using the stable-name mapping from the YAML.

    Returns (missing, unexpected) lists of stable names."""
    with open(config_path) as f:
        hf = json.load(f)
    defn = _load_definition(hf["model_type"])
    num_layers = hf["num_hidden_layers"]
    hf_to_mat = _build_weight_map(defn, num_layers)   # hf_name => stable_name

    # Collect all weight shards (safetensors preferred)
    folder = Path(config_path).parent
    index_files = list(folder.glob("*.index.json"))
    if index_files:
        with open(index_files[0]) as f:
            shard_map = json.load(f)["weight_map"]   # hf_name => filename
        shards = sorted(set(shard_map.values()))
    else:
        shards = ([f.name for f in folder.glob("*.safetensors")]
                  or [f.name for f in folder.glob("*.bin")])
        shard_map = {k: shards[0] for k in hf_to_mat}   # everything in one file

    loaded_shards = {}
    def _get_tensor(hf_name):
        fname = shard_map.get(hf_name)
        if fname not in loaded_shards:
            path = folder / fname
            if path.suffix == ".safetensors":
                from safetensors.torch import load_file
                loaded_shards[fname] = load_file(path, device="cpu")
            else:
                import torch
                loaded_shards[fname] = torch.load(path, map_location="cpu", weights_only=True)
        return loaded_shards[fname].get(hf_name)

    missing, unexpected = [], []
    for hf_name, stable_name in hf_to_mat.items():
        tensor = _get_tensor(hf_name)
        if tensor is None:
            unexpected.append(hf_name)
            continue
        ptr = model._get_weight_pointer(stable_name)
        if ptr is None:
            missing.append(stable_name)
        else:
            model._set_weight(stable_name, tensor)

    return missing, unexpected
