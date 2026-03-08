# matformer/base.py
import torch
import torch.nn as nn
from typing import Optional
from itertools import chain
try:
    import pytorch_lightning as pl
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    

#class ParametersRenamer:
"""
    The system for automatically renaming parameters is being rewritten.

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
        class_annotations=getattr(module.__class__,'__annotations__',{})
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
 
    
    def stable_state_dict(self):
        # register_state_dict_post_hook(hook)
        mapping = self._mappings
        reverse = {v: k for k, v in mapping.items()}
        raw_dict = nn.Module.state_dict(self)
        return {reverse.get(k, k): v for k, v in raw_dict.items()}
        


if HAS_LIGHTNING:
    class MatformerModule(pl.LightningModule, ParametersRenamer):
        def __init__(self):
            super().__init__()
            self.has_lightning = True
            self.strict_loading = False
        def on_save_checkpoint(self, checkpoint: dict) -> None:
            """Transform state dict before saving."""
            checkpoint['state_dict'] = self.stable_state_dict()
            print(f"Saved checkpoint with {len(checkpoint['state_dict'])} keys")
        
        def load_state_dict(self, state_dict, strict=True):
            """Translate stable keys to actual keys before loading."""
            mapping = self._mappings  # stable -> actual
            translated = {mapping.get(k, k): v for k, v in state_dict.items()}
            self._state_dict_translated = True  
            return super().load_state_dict(translated, strict=strict)
        
        def on_load_checkpoint(self, checkpoint: dict) -> None:
            """Fallback: translate if load_state_dict didn't run."""
            if not getattr(self, '_state_dict_translated', False):
                mapping = self._mappings
                checkpoint['state_dict'] = {mapping.get(k, k): v for k, v in checkpoint['state_dict'].items()}
            self._state_dict_translated = False 
else:
    class MatformerModule(nn.Module, ParametersRenamer):
        """
        Base class for Matformer modules when Lightning is NOT available.
        Provides stable checkpointing + manual device/dtype management.
        """
        def __init__(self):
            super().__init__()
            self._matformer_device = None
            self._matformer_dtype = None
            self.has_lightning = False
        
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


# Utility to check which mode we're in
def is_lightning_available():
    return HAS_LIGHTNING
