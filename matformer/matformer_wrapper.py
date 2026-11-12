# matformer/matformer_wrapper.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from matformer.matformer_tokenizers import MatformerTokenizer
from matformer.utils.initialization import init_transformer_weights_
from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor,UnpaddedTensor,PaddedTensor
from matformer.utils.matformer_cache import CachedStuff
from matformer.matformer_module import MatformerModule
from matformer.training.ckpt_manager import CheckpointDirectoryManager
from matformer.dataclasses.model_dataclasses import ModelConfig, TokenizerConfig, MaskingConfig
from matformer.matformer_configs import MatformerConfigs
from matformer.matformer_registry import registry
from matformer.plugins.plugin_manager import PluginManager
from matformer.plugins.plugin_classes import Plugin
from matformer.loading.utils import make_holder
import traceback
import os
import warnings
import torch.nn as nn
from dataclasses import asdict, replace
import gc
from contextlib import nullcontext
import types
#Separare Virtualmodel da qui
from matformer.dataclasses.matformer_config import MatformerConfig
from typing import Optional
from dataclasses import dataclass, field

@registry.register_dataclass("virtual_model","default",default=True)
@dataclass
class VirtualModelConfig(MatformerConfig):
    check_input: bool = True
    input_address: str ="encoder.model_input"
    default_return: Optional[str] = None
    return_multiprobe: bool = True
class VirtualModel(MatformerModule):
    """A virtual model that has sockets for plugins but doesn't execute a real model:
    it replays activations from an activation store instead."""
    def __init__(self, config, cache, store, model_config):
        super().__init__()
        self.model_config = model_config   
        self.init_sockets()
        self.config = config
        self.cache = cache
        self.store = store
        self.register_buffer("_device_probe", torch.empty(0), persistent=False)
        self.sockets_by_address = {a: self.add_nested_socket(a) for a in store.available_addresses()}
        
    def default_plugins(self):
        """Same plugins the live model of this config would mount (probes etc.)."""
        variant = self.model_config._matformer_variant
        cls = registry.registry['model'].variants[variant].implementations['default'].get_implementation()
        if not hasattr(cls, 'default_plugins'):
            return []
        proxy = types.SimpleNamespace(config=self.model_config)   # default_plugins only reads self.config
        return cls.default_plugins(proxy)

    def on_epoch_start(self):
        self.store.position = 0

    def forward(self, ids=None, batch_size=None, **kwargs):
        unpadded = ids is not None and ids.isUnpadded
        batch_size = batch_size or (ids.batch_size if unpadded else len(ids))
        ids_range = [i % len(self.store) for i in range(self.store.position, self.store.position + batch_size)]
        active = [a for a, s in self.sockets_by_address.items() if s.components]
        needed = set(active) | ({self.config.input_address} if self.config.check_input else set())
        data, amount = self.store.fetch_all(ids_range, needed)
        data = {a: (v.unpad() if unpadded else v).to(self.device) for a, v in data.items()}

        if self.config.check_input and ids is not None:    
            stored = data[self.config.input_address]
            if not torch.equal(stored.tensor, ids.tensor):
                print(f"WARNING: inconsistent data between forward and storage (counter={self.store.position}, databank={self.store.databank})")

        self.cache.storage['probe_outputs'] = {}            
        for address in active:                      
            data[address] = self.sockets_by_address[address](data[address])
        self.store.position += amount
        if self.config.return_multiprobe:
            return self.cache.storage['probe_outputs']
        return data[self.config.default_return] if self.config.default_return else data
## Fino a qui
class MatformerModelWrapper(MatformerModule, PluginManager):
    """
    The ModelWrapper contains methods to load checkpoints, weights as well as everything required to train the model.
    The training procedure itself, however, is not hard-coded into the Wrapper but is handled through another class,
    called LearningSignal. The LearningSignal is the one containing the actual preparation of the inputs 
    (ex. masking for Bert-like models) and the selection of targets for the loss function.
    For more complex scenario, it is possible to use a "LearningSignalSelector", another class that select the correct LearningSignal according
    to specific criteria step by step (ex. to randomly mix more LearningSignal(s) or to correctly route the data according
    to info present in the batch's metadata.)
    The LearningSignal, LearningSignalSelector and the model class are all accessed through the [[MatformerRegistry]]
    """

    model: "transparent"

    def __init__(self, configs, device, runtime_mode="live", activation_store=None):
        super().__init__()
        self.configs = configs
        self._init_device = device
        has_model, has_store = hasattr(configs, 'model'), hasattr(configs, 'activation_store')
        assert has_model or has_store, "Configs need a 'model' or an 'activation_store' section."
        self.cache = CachedStuff()
        self._models = {}   #Links to virtual/live

        self.activation_store = activation_store    
        if has_store and self.activation_store is None:
            self.activation_store = configs.instantiate(
                'activation_store', model_config=configs.model if has_model else None, cache=self.cache)
        elif self.activation_store is not None:
            self.activation_store.cache = self.cache   

        self.config = configs.model
        self.runtime_mode = runtime_mode
        self.model = self._get_model(self.runtime_mode)
        self.init_plugins()
        tokenizer = None
        if hasattr(configs, 'tokenizer'):
            try: tokenizer = configs.tokenizer.instantiate()
            except Exception: print(f"WARNING: Impossible to instantiate Tokenizer {configs.tokenizer}")
        if tokenizer is not None:
            self.attach_tokenizer(tokenizer)

        if hasattr(self.configs, 'training'):
            print("Sto iniziando i pesi...")
            self.model.apply(init_transformer_weights_) #Weights are initialized here, before loading weights from a checkpoint 
            self.learning_signal_selector = registry.create(
                "learning_signal_selector", configs.training.learning_signal_selector,
                wrapper=self #Default is "single": only one learning signal.
            )

            self.apply_freeze_config() # Freeze/unfreeze components
            print(self.trainable_param_report())

    def on_epoch_start(self):
        try:
            self.model.on_epoch_start()
        except:
            print("The model doesn't have any epoch start loop. Fine. ")

    def _get_model(self, mode):
        if mode not in self._models:           
            if mode == 'live':
                variant = self.config._matformer_variant
                if not variant or variant == "default":
                    variant = self.config._matformer_variant = "language_model"
                self._models[mode] = registry.create("model", variant, config=self.config,
                    device=self._init_device, cache=self.cache, wrapper=self)
            else:
                self._models[mode] = VirtualModel(self.configs.virtual_model, self.cache, self.activation_store, model_config=self.config)
        return self._models[mode]

    def set_runtime_mode(self, mode):
        assert mode in ('live', 'virtual')
        if mode == self.runtime_mode:
            return
        moving = [(p.socket.address, p) for p in list(self.plugins.values())]
        for address, p in moving:
            self._unmount(p.socket, p.name)
        self.model = self._get_model(mode)   
        self.runtime_mode = mode
        self.index_sockets()
        for address, p in moving:
            socket = self.sockets.get(address)
            if socket is None:
                print(f"WARNING: no socket {address} in {mode} model; plugin {p.id} dropped")
            else:
                self._mount(socket, p, None, p.kind)
        self.device=self.model.device
    def forward(self, _input, *args, **kwargs):
        self.reset_visits()
        if isinstance(_input, torch.Tensor):
            _input = NormalTensor(tensor=_input)
        if isinstance(_input, NormalTensor) and _input.tensor.ndim == 1:
            warnings.warn("1-D input: adding batch dim. Output will be (1, S, ...).")
            _input = replace(_input, tensor=_input.tensor.unsqueeze(0))
        return self.model(_input.to(self.device), *args, **kwargs)
    def params_for_target(self, target):
        """
        Used to parse list of user-friendly parameters/components components; used in "freeze", check freeze's config
        """
        if isinstance(target, dict):
            params = []
            for _, module in self.model.named_modules():
                if getattr(module, 'layer_idx', None) is None:
                    continue
                if not self.config._matches_pattern(module.layer_idx, target['layer']):
                    continue
                for name, child in module.named_children():
                    if target.get('component') and target['component'] not in name:
                        continue
                    params += list(child.parameters(recurse=True))
            return params

        if target == "all":
            return list(self.model.parameters())

        plugin_type = None
        layer_pattern = None
        component_name = None

        if target.startswith("plugin_type:"):
            plugin_type = target[len("plugin_type:"):]
        elif target.startswith("layer:"):
            layer_pattern = target[len("layer:"):]
        elif target.startswith("component:"):
            component_name = target[len("component:"):]
        elif target in (sockets := self.find_socket("*")):
            return [p for c in sockets[target].components
                    if isinstance(c, Plugin) for p in c.parameters()]
        else:
            plugin_type = target


        params = []

        if plugin_type is not None:
            for _, plugin in self.plugins.items():
                kind = getattr(type(plugin), "name", type(plugin).__name__.lower())
                if kind == plugin_type or plugin.name == plugin_type:
                    params += list(plugin.parameters())
            if params:
                return params
            if target.startswith("plugin_type:"):
                return params

        if layer_pattern is not None:
            for _, module in self.model.named_modules():
                if getattr(module, 'layer_idx', None) is None:
                    continue
                if self.config._matches_pattern(module.layer_idx, layer_pattern):
                    params += list(module.parameters(recurse=True))
            return params

        component_name = component_name or target
        for stable_name, actual_name in self.model._mappings.items():
            if component_name in stable_name.split("."):
                params.append(self.model.get_parameter(actual_name))
        if params:
            return params

        stable_param_name = target
        for stable_name, actual_name in self.model._mappings.items():
            if stable_name == stable_param_name or stable_name.startswith(stable_param_name + "."):
                params.append(self.model.get_parameter(actual_name))
        return params
    def apply_freeze_config(self):
        if hasattr(self.configs, 'training'):
            for param in self.model.parameters():
                param.requires_grad = True
            if self.configs.training.freeze:
                if not isinstance(self.configs.training.freeze,list):
                    self.configs.training.freeze=[self.configs.training.freeze]
                for target in self.configs.training.freeze:
                    for param in self.params_for_target(target):
                        param.requires_grad = False
            if self.configs.training.unfreeze:        
                if not isinstance(self.configs.training.unfreeze,list):
                    self.configs.training.unfreeze=[self.configs.training.unfreeze]                                
                for target in self.configs.training.unfreeze:
                    for param in self.params_for_target(target):
                        param.requires_grad = True
    def trainable_param_report(self):
        plugin_params = self.all_plugin_params()
        plugin_ids = {id(p) for p in plugin_params}
        backbone_params = [p for p in self.model.parameters() if id(p) not in plugin_ids]

        rows = {
            "total":       list(self.model.parameters()),
            "no-plugins":  backbone_params,
            "only-plugins": plugin_params,
            "mlp":         self.params_for_target("component:mlp"),
            "attn":        self.params_for_target("component:self_attn"),
            "norms":       self.params_for_target("component:norm"),
            "embedding":   self.params_for_target("component:embed"),
            "head":        self.params_for_target("plugin_type:classification")
        }

        print(f"{'name':<15}{'total':>15}{'frozen':>15}{'trainable':>15}")
        report = {}
        for name, params in rows.items():
            total = sum(p.numel() for p in params)
            trainable = sum(p.numel() for p in params if p.requires_grad)
            frozen = total - trainable
            report[name] = {"total": total, "frozen": frozen, "trainable": trainable}
            print(f"{name:<10}{total:>10,}{frozen:>10,}{trainable:>10,}")

        return report
    def training_step(self, batch, batch_idx=None):
        try:
            if batch.attributes.get('worker_has_finished', False):
                return sum(p.sum() for p in self.parameters()) * 0.0 # Zero loss when batch is de-syncronized across ranks (multi-GPU)

            self.learning_signal_selector.pre_step(batch) #an optional hook
            active_signals = self.learning_signal_selector.pick(batch, batch_idx) #Get the learning signal(s) to be used in this batch
            total_loss = None #Loss of each learning signal's forward pass is summed

            for signal in active_signals:
                input_sequence, metadata = signal.prepare_inputs(batch)
                predictions = self(input_sequence, **signal.model_kwargs(metadata)) #Input to the model
                predictions, targets = signal.select_targets(predictions, batch, metadata)
                loss = signal.loss_function(predictions, targets) #Loss
                total_loss = loss if total_loss is None else total_loss + loss
                signal.post_loss(loss, predictions, targets) #an optional hook

            self.log('train/loss', total_loss, prog_bar=True, batch_size=self.configs.training.batch_size)

            if 'aux_losses' in self.cache.storage: # Add other losses contained in the cache (ex. MoE router loss)
                aux_losses = self.cache.storage['aux_losses']
                if aux_losses:
                    total_aux_loss = torch.stack(aux_losses).sum()
                    total_loss = total_loss + total_aux_loss
                    self.log("train/aux_loss", total_aux_loss.item(), on_step=True)
                self.cache.storage['aux_losses'] = []

            try:
                current_lr = self.lr_schedulers().get_last_lr()[0] # Logging the learning rate
                self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=False,
                         batch_size=self.configs.training.batch_size)
            except Exception:
                pass

            return total_loss

        except Exception as e:
            return self._handle_exception(e)
    def _handle_exception(self,e):
            print("------------------  ALARM ------------------")
            if not any(s in str(e).lower() for s in ("out of memory", "cuda out of memory", "cublas")):
                print("CAUGHT EXCEPTION:")
                print(e)
                print("Training will try to continue but this may be unreliable. Careful check what is happening")
                import traceback
                traceback.print_exc()
                self.log("train/EXCEPTIONS",1,on_step=True,on_epoch=False,batch_size=self.configs.training.batch_size)
            else:
                print(f"OUT OF MEMORY ERROR {e}")
                print("Training is continuing, but be sure that this happens very seldom or reduce the batch size.")
                self.log(
                    "train/skipped_oom", 1,
                    on_step=True, on_epoch=False, batch_size=self.configs.training.batch_size
                )
            print("All the gradients from the current batch and all the previous accumulated gradients were discarded.")
            try:
                opts = self.optimizers()
                for o in opts if isinstance(opts, list) else (opts,):
                    o.zero_grad(set_to_none=True)
            except Exception:
                pass

            if hasattr(self.cache, "storage"):
                self.cache.storage.clear()

            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            return sum(p.sum() for p in self.parameters()) * 0.0
    def setup_logging(self, logger, step_ref):
        self._matformer_logger = logger
        self._step_ref = step_ref
    def log(self, key, value, **kwargs):
        if getattr(self, '_matformer_logger', None) is not None:
            v = value.item() if hasattr(value, 'item') else value
            self._matformer_logger.log(key, v, step=self._step_ref[0], **kwargs)
    @property
    def global_step(self):
        return self._step_ref[0] if getattr(self, '_step_ref', None) else 0        

    
    def _get_weight_pointer(self, stable_name):
        """
        From stable name, returns the associated weight
        """
        try:
            return self.get_parameter(self._mappings[stable_name])
        except:
            return None

    def _set_weight(self, stable_name, weight):
        """
        Allows to set a weight to a specific value, indexed by stable name. Returns False when
        the tensor doesn't fit the parameter (shape mismatch): the caller then counts it as
        missing, so a module left random-initialized can't pass as loaded.
        """
        with torch.no_grad():
            try:
                self._get_weight_pointer(stable_name).copy_(weight)
                return True
            except Exception as e:
                print(f"Impossible to set {stable_name} {e}")
                return False

    def load_stable_state_dict(self, stable_state_dict, include=None, exclude=None):
        """include/exclude: fnmatch patterns on stable keys. "missing" is computed only over the selected keys"""
        IGNORED=['alibi', 'extra_state']
        from fnmatch import fnmatch
        def _key_filter(include, exclude):
            inc = [include] if isinstance(include, str) else (include or [])
            exc = [exclude] if isinstance(exclude, str) else (exclude or [])
            return lambda k: (not inc or any(fnmatch(k, p) for p in inc)) and not any(fnmatch(k, p) for p in exc)
        keep = _key_filter(include, exclude)
        obtained, unexpected = [], []
        for key, tensor in tqdm(stable_state_dict.items()):
            if not keep(key):
                continue
            if self._get_weight_pointer(key) is None:
                unexpected.append(key)
            elif self._set_weight(key, tensor):
                obtained.append(key)
        missing = {k for k in self._mappings if keep(k)} - set(obtained)
        ign = lambda k: any(s in k for s in IGNORED)
        missing, unexpected = {k for k in missing if not ign(k)}, [k for k in unexpected if not ign(k)]
        print(f"MISSING: {missing}\nUNEXPECTED: {unexpected}")
        return missing, unexpected


    def load_additional_weights(self):
        """Load one or more auxiliary weight sources (model merging, autoencoders, classification heads...).
        Each entry may set 'wanted' or 'discard' (mutually exclusive) to choose which stable keys
        to inherit from that checkpoint: the selection is applied lazily at the holder, and the
        missing/unexpected bookkeeping below is scoped to the same selection."""
        if not hasattr(self.configs, 'weights'):
            return []
        results = []
        for _, w in self.configs.weights:
            holder = make_holder(w.path, map_location=self.device)
            missing, unexpected = self.load_stable_state_dict(
                holder.get_weights(w.definition, wanted=w.wanted, discard=w.discard),
                include=w.wanted, exclude=w.discard)
            results.append({"path": w.path, "missing": missing, "unexpected": unexpected + holder.unresolved})
        return results


    def attach_tokenizer(self, tokenizer):
        self.model.tokenizer = self.tokenizer = tokenizer
        if hasattr(self.configs, 'model'):
            self.configs.model.tokenizer = tokenizer.config
        

def load_model(path=None, checkpoint_holder=None, weights_definition=None, configs=None,
               map_location='cpu', dtype=torch.bfloat16, inference=False, inherit_sections=('model', 'tokenizer')):

    # Virtual-only runs: the store replaces the backbone, so its weights are never loaded.
    # Read from the incoming (trainer's) configs, before any merging.
    training = getattr(configs, 'training', None)
    virtual_only = bool(training is not None and training.get('virtual_only', False))
    store = None
    if virtual_only:
        assert hasattr(configs, 'activation_store'), "virtual_only needs an 'activation_store' section."
        store = configs.instantiate('activation_store', model_config=None, cache=None)

    if path is not None:
        assert checkpoint_holder is None,"Either you load from path or from a checkpoint holder"
        checkpoint_holder=make_holder(path, map_location=map_location)
        print(checkpoint_holder)
    if checkpoint_holder:
        ckpt_configs=checkpoint_holder.get_configs() # lazy: no weights are read here
        assert ckpt_configs is not None, "No valid configs found. Hint: if you are loading an HuggingFace model, remember hf:// before its path!"
        if ckpt_configs:
            for s in list(ckpt_configs.list_configs()):
                    if s not in inherit_sections:
                        ckpt_configs.remove_section(s)
        
    if not checkpoint_holder and store is not None:   # the store's model config is injected through the config class
        ckpt_configs=MatformerConfigs.create_configs({"model": store.load_model_config()})
    elif not checkpoint_holder:
        ckpt_configs=MatformerConfigs()   # bare root, populated from `configs` (if given) below

    if configs:
        # Provided configs override what's included in the checkpoing
        ckpt_configs.import_config(configs.asdict(include_defaults=False))
        ckpt_configs.resolve_references()
    configs=ckpt_configs
    
    assert configs is not None,"Either pass a valid configs or a path to a checkpoint containing a valid configuration."
    if hasattr(configs, 'model'):
        print(f"Sto usando come cfg: {configs.model}")
    prev = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)
    try:
        with torch.device(map_location) if map_location is not None else nullcontext():
            model = MatformerModelWrapper(configs=configs, device=map_location,
                                          runtime_mode='virtual' if virtual_only else 'live',
                                          activation_store=store)
    finally:
        torch.set_default_dtype(prev)
    
    """APPOGGIO"""
    from collections import Counter

    def memory_report(model, tag=""):
        by_dtype = Counter()
        for t in [*model.parameters(), *model.buffers()]:
            by_dtype[str(t.dtype)] += t.numel() * t.element_size()
        gb = lambda b: f"{b / 1024**3:.2f} GB"
        print(f"[{tag}] model tensors: " + ", ".join(f"{d}: {gb(b)}" for d, b in by_dtype.items())
              + f" | total: {gb(sum(by_dtype.values()))}")
        if torch.cuda.is_available():
            print(f"[{tag}] cuda allocated: {gb(torch.cuda.memory_allocated())} | "
                  f"reserved: {gb(torch.cuda.memory_reserved())} | "
                  f"peak allocated: {gb(torch.cuda.max_memory_allocated())}")
    try:
        memory_report(model)
    except:
        print("Memory report failed.")
    if map_location:
        model = model.to(map_location)
    if dtype:
        model = model.to(dtype)
    for m in model.modules():
        if getattr(m, "alibi_slopes", None) is not None:
            m.alibi_slopes = m.alibi_slopes.to(torch.float32)
    if inference:
        model.eval()
    if checkpoint_holder:
        # In virtual mode _mappings only holds plugin params (probes): ask the holder for
        # exactly those, so the backbone weights are never fetched nor transformed.
        weights = checkpoint_holder.get_weights(
            weights_definition, wanted=model._mappings if virtual_only else None)
        missing, unexpected = model.load_stable_state_dict(weights)
        memory_report(model)
        checkpoint_holder.release()
       
    else:
        missing, unexpected = None, None
    memory_report(model)
    gc.collect()
    torch.cuda.empty_cache()
    memory_report(model)
    additional_weights_results=None
    if hasattr(model.configs,'weights'):
        additional_weights_results=model.load_additional_weights()  

    return {
        "model": model,
        "configs": configs,
        "missing": missing,
        "unexpected": unexpected,
        "additional_weights_results":additional_weights_results
        }
