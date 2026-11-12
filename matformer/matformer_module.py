# matformer/matformer_module.py
from matformer.matformer_registry import registry
import torch
import torch.nn as nn
from typing import Optional
from itertools import chain
from matformer.plugins.socket_mixin import SocketMixin
from tqdm import tqdm #TEMPORANEO
"""
    Definitions:
        "Stable_name(s)" => Names of the parameters defined at annotation level, for example 
            class TransformerWithClassificationHead(MatformerModule): 
            encoder: "param_name:encoder"
            classification_head: "param_name:classifier"
            dropout: "param_name:dropout"
        "Actual_name(s)" => The name used internally by matformer, and dependent from the way in which modules are coded and nested.
        "External_name(s)" => A possibility to load any state dict, also of models not originally trained with Matformer, with a dictionary that converts the names used by those external state dicts into the "stable_names"
"""            
class ParametersRenamer:
    @property
    def _mappings(self): 
        """
        Mapping from stable name to actual names
        """
        if not hasattr(self, '_parameters_mappings'):
            self._parameters_mappings = {}
            self._scan_tree(self, "", "")
        return self._parameters_mappings
    
    def _scan_tree(self,module,stable_prefix,actual_prefix):
        """
        Scan tree 
        """
        # "Annotations" are the "param_name" string under class definition, ex encoder: "param_name:encoder". They must start with param_name to be recognized.
        # Collect annotations from the full MRO (base to derived), then instance annotations
        class_annotations = {}
        for cls in reversed(module.__class__.__mro__): 
            if hasattr(cls, '__annotations__'):
                class_annotations.update(cls.__annotations__)
        # Instance-level annotations take final priority
        class_annotations.update(getattr(module, '__instance_annotations__', {}))

        # Registry annotations are the _params_names present in the registry module (ex. params_names={'inner.weight': 'weight'}); it is a dict where keys are the "actual_names" and values the "stable_names"
        registry_annotations=getattr(module,'_params_names',{})
        
        # Cycle into the module's parameters and buffers
        for actual_name, _ in chain(module.named_parameters(recurse=False), module.named_buffers(recurse=False)):
            stable_name=registry_annotations.get(actual_name,actual_name) # Get the stable name; if missing, regression to actual name
            stable_path = '.'.join(filter(None, [stable_prefix, stable_name]))
            actual_path = '.'.join(filter(None, [actual_prefix, actual_name]))
            self._parameters_mappings[stable_path]=actual_path
        
        for actual_child_name, child_module in module.named_children():
            ann=class_annotations.get(actual_child_name, actual_child_name)
            #if ann == "transparent": <= It should be like this, but instead I put if transparent in ann WIP because from future import annotations in some files messes with the annotations adding a single quote instead the string "transparent" ("'transparent'"). To fix better. 
            if "transparent" in ann:
                stable_child_name = None
            elif ann.startswith('param_name:'):
                stable_child_name = ann.split(':', 1)[1]
            else:
                stable_child_name = actual_child_name
            child_stable_path = '.'.join(filter(None, [stable_prefix, stable_child_name]))
            child_actual_path = '.'.join(filter(None, [actual_prefix, actual_child_name]))
            self._scan_tree(child_module,child_stable_path,child_actual_path)
 

    @property
    def _module_mappings(self):
        """Mapping from stable module path to actual module path (and the module itself)."""
        if not hasattr(self, '_module_mappings_cache'):
            self._module_mappings_cache = {}
            self._scan_modules(self, "", "")
        return self._module_mappings_cache

    def _scan_modules(self, module, stable_prefix, actual_prefix):
        self._module_mappings_cache[stable_prefix] = (actual_prefix, module)

        class_annotations = {}
        for cls in reversed(module.__class__.__mro__):
            class_annotations.update(getattr(cls, '__annotations__', {}))
        class_annotations.update(getattr(module, '__instance_annotations__', {}))

        for name, child in module.named_children():
            ann = class_annotations.get(name, name)
            if "transparent" in ann:
                child_stable = stable_prefix
            elif ann.startswith('param_name:'):
                child_stable = '.'.join(filter(None, [stable_prefix, ann.split(':', 1)[1]]))
            else:
                child_stable = '.'.join(filter(None, [stable_prefix, name]))
            self._scan_modules(child, child_stable, '.'.join(filter(None, [actual_prefix, name])))    
    def stable_state_dict(self):
        # register_state_dict_post_hook(hook)
        mapping = self._mappings
        reverse = {v: k for k, v in mapping.items()}
        raw_dict = nn.Module.state_dict(self)
        return {reverse.get(k, k): v for k, v in raw_dict.items()}
    """TEMPORANEO! INCOLLO QUI I METODI DEL WRAPPER PER CARICARE I PESI DEI PLUGIN, DECIDERE COSA FARE IN VIA DEFINITIVA"""
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

class ParameterEditor:
    """Weight-space interventions on a plugin's parameters, with backup and restore.

    Edit:    edit, project_out
    Freeze:  freeze (gradient masking, values untouched)
    Inspect: directions, leak, delta, stats
    Undo:    restore

    Mask convention everywhere: True = entry stays untouched / trainable.
    """

    def _store(self):
        if not hasattr(self, "_param_backup"):
            self._param_backup, self._param_hooks = {}, {}
        return self._param_backup

    def _p(self, name):
        """The LIVE parameter (never a state_dict copy: state_dict() detaches, and gradient
        hooks registered on a detached view never fire, which made freeze_grad a silent no-op)."""
        params = dict(self.named_parameters())
        if name in params:
            return params[name]
        raw = self._mappings.get(name, name)
        if raw in params:
            return params[raw]
        raise KeyError(f"'{name}' is not a parameter of {type(self).__name__}. "
                       f"Stable names available: {sorted(self._mappings)}")

    def snapshot(self, name):
        return self._p(name).detach().clone()

    def _remember(self, name):
        s = self._store()
        if name not in s:
            s[name] = self.snapshot(name)

    def _hook(self, name, fn):
        """Register a gradient hook on `name`; restore() removes it."""
        p = self._p(name)
        self._remember(name)
        if p.requires_grad:
            self._param_hooks.setdefault(name, []).append(p.register_hook(fn))

    @staticmethod
    def _proj(x, d, side):
        """Remove span(d) from x's input (last) dim or output (first) dim. Linear, so also valid for grads."""
        return x - (x @ d.T) @ d if side == "in" else x - d.T @ (d @ x)

    @staticmethod
    def _basis(p, direction):
        d = torch.as_tensor(direction, device=p.device, dtype=p.dtype)
        d = d.unsqueeze(0) if d.dim() == 1 else d
        return torch.linalg.qr(d.T).Q.T                   # (k, D), orthonormal

    def freeze(self, name, train=None):
        """Zero the gradient everywhere except where `train` is True (None = freeze all). Values untouched.

        Sparse probing: freeze(name, train=mask). Create the optimizer afterwards, with weight_decay=0.
        """
        p = self._p(name)
        if train is None:
            frozen = torch.ones_like(p, dtype=torch.bool)
        else:
            frozen = ~torch.as_tensor(train, device=p.device, dtype=torch.bool)
        self._hook(name, lambda g, f=frozen: g.masked_fill(f, 0))
        return p

    def edit(self, name, value=0, mask=None, freeze_grad=True):
        """Overwrite `name` with `value`; with `mask`, only where mask is False (True entries are kept).

        If freeze_grad, the overwritten entries are frozen and the kept ones stay trainable.
        """
        p = self._p(name)
        self._remember(name)
        keep = None if mask is None else torch.as_tensor(mask, device=p.device, dtype=torch.bool)
        with torch.no_grad():
            if keep is None:
                p.copy_(torch.as_tensor(value, device=p.device, dtype=p.dtype))
            else:
                p.masked_fill_(~keep, value)
        if freeze_grad:
            self.freeze(name, train=keep)
        return p

    def project_out(self, name, direction, side="in", freeze_grad=True):
        """Remove `direction` ((D,) or (k, D)) from the input ("in") or output ("out") space of 2D `name`.

        If freeze_grad, gradients are projected the same way, so training cannot reintroduce it.
        """
        p = self._p(name)
        self._remember(name)
        d = self._basis(p, direction)
        with torch.no_grad():
            p.copy_(self._proj(p, d, side))
        if freeze_grad:
            self._hook(name, lambda g, D=d: self._proj(g, D, side))
        return p

    def directions(self, name, k=1, side="in"):
        """Top-k singular vectors of `name`: input space ("in") or output space ("out"), shape (k, D)."""
        U, S, Vh = torch.linalg.svd(self._p(name).detach().float(), full_matrices=False)
        return Vh[:k] if side == "in" else U[:, :k].T

    def leak(self, name, direction, side="in"):
        """Fraction of ||W||² lying in span(direction). ~0 after a successful project_out."""
        p = self._p(name).detach()
        d = self._basis(p, direction)
        part = p @ d.T if side == "in" else d @ p
        return (part.float().norm() ** 2 / p.float().norm() ** 2).item()

    def delta(self, name):
        """Relative change vs. backup (0 if never edited)."""
        b = self._store().get(name)
        if b is None:
            return 0.0
        p = self._p(name).detach()
        return ((p - b).float().norm() / b.float().norm().clamp_min(1e-12)).item()
    def l1(self, name, lam):
        """Add lam*sign(w) to the gradient (L1 half of elastic net)"""
        p = self._p(name)
        self._hook(name, lambda g, p=p: g + lam * p.detach().sign())

    def stats(self, name):
        """Mean/std/abs_mean; for 2D also top singular value and effective rank (exp of spectral entropy)."""
        p = self._p(name).detach().float()
        out = {"mean": p.mean().item(), "std": p.std().item(), "abs_mean": p.abs().mean().item()}
        if p.dim() == 2:
            S = torch.linalg.svdvals(p)
            q = S / S.sum()
            out.update(top_sv=S[0].item(),
                       eff_rank=torch.exp(-(q * q.clamp_min(1e-12).log()).sum()).item())
        return out

    def restore(self, name=None):
        """Undo edits and remove gradient hooks; `name=None` restores everything."""
        for k in list(self._store()):
            if name is None or k == name:
                for h in self._param_hooks.pop(k, []):
                    h.remove()
                with torch.no_grad():
                    self._p(k).copy_(self._param_backup.pop(k))
    def get_direction(self, name, dir1=1, dir2=0):
        """ Return the direction between two arbitrary dimension of the vector """
        w=self._p(name)        
        assert max(dir1,dir2)<=w.numel()
        return w[dir1]-w[dir2]

class MatformerModule(nn.Module, ParametersRenamer, SocketMixin, ParameterEditor):
        """
        Base class for Matformer modules when Lightning is NOT available.
        Provides stable checkpointing + manual device/dtype management.
        Probably going to be deprecated
        """
        def __init__(self):
            super().__init__()
            self._matformer_device = None
            self._matformer_dtype = None
            self.has_lightning = False
            self.tokenizer=None
        def to(self, *args, **kwargs):
            """Override to track device/dtype"""
            result = super().to(*args, **kwargs)
            
            # Parse arguments to extract device/dtype
            for arg in args:
                if isinstance(arg, torch.device):
                    self._matformer_device = arg
                elif isinstance(arg, torch.dtype):
                    self._matformer_dtype = arg
                elif isinstance(arg, str):
                    # Could be device string like 'cuda' or 'cpu'
                    try:
                        self._matformer_device = torch.device(arg)
                    except:
                        pass
            
            if 'device' in kwargs:
                self._matformer_device = kwargs['device']
            if 'dtype' in kwargs:
                self._matformer_dtype = kwargs['dtype']
            
            return result

        @property
        def device(self):
            """Property to mimic Lightning's device"""
            if self._matformer_device is not None:
                return self._matformer_device
            
            # Infer from first parameter
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device('cpu')
        
        @property
        def dtype(self):
            """Property to access dtype"""
            if self._matformer_dtype is not None:
                return self._matformer_dtype
            
            # Infer from first parameter
            try:
                return next(self.parameters()).dtype
            except StopIteration:
                return torch.float32
        
        def save_checkpoint(self, path):
            """Manual checkpoint saving (mimics Lightning)"""
            checkpoint = {
                'state_dict': self.stable_state_dict(),
                'hyper_parameters': {
                    'config': getattr(self, 'config', None)
                }
            }
            torch.save(checkpoint, path)
        
        @classmethod
        def load_from_checkpoint(cls, path, **kwargs):
            """Manual checkpoint loading (mimics Lightning)"""
            checkpoint = torch.load(path, map_location=kwargs.get('map_location', 'cpu'))
            
            # Extract config if saved
            if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
                kwargs.setdefault('config', checkpoint['hyper_parameters']['config'])
            
            # Instantiate model
            model = cls(**kwargs)
            
            # Load state
            model.load_stable_state_dict(checkpoint['state_dict'], strict=False)
            
            return model
        def configure_sockets(self):
            """ Here the """
            return None

