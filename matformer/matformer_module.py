# matformer/base.py
import torch
import torch.nn as nn
from typing import Optional

try:
    import pytorch_lightning as pl
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False


class ParametersRenamer:
    """
    Mixin class that handles parameter name mapping/translation.
    Allows defining stable parameter names independent of internal implementation.
    """
    
    def get_parameters_name_mapping(self, external_mapping: Optional[dict] = None):
        """
        Get the mapping from stable parameter names to actual parameter paths.
        
        Args:
            external_mapping: Optional dict for translating external checkpoint names
                             (e.g., from pretrained models) to our stable names.
                             Format: {checkpoint_name: stable_name}
        
        Returns:
            Dict mapping stable/external names to actual parameter paths in the model
        """
        if not hasattr(self, '_parameters_mappings'): #_parameters_mapping, once initialized, store the mapping of each class
            self._build_parameters_mappings()
        
        # If external mapping provided (for loading pretrained models)
        if external_mapping:
            # Create a new composed mapping: external -> stable -> actual
            composed = {}
            
            # For each entry in the external mapping
            for checkpoint_name, parameters_name in external_mapping.items():
                # Look up what actual path this stable name maps to
                actual_name = self._parameters_mappings.get(parameters_name)
                
                # If we found a mapping for this stable name
                if actual_name:
                    # Map external checkpoint name to actual internal path
                    composed[checkpoint_name] = actual_name
                else:
                    # No mapping found, use the parameters_name as-is
                    # (might be already an actual path)
                    composed[checkpoint_name] = parameters_name
            
            # Return the composed mapping
            return composed
        
        # No external mapping, just return our internal stable->actual mapping
        return self._parameters_mappings
    
    def _build_parameters_mappings(self):
        """
        Build the internal mapping from stable parameter names to actual paths.
        This is called once when first accessing the mapping.
        
        Process:
        1. Scan class annotations for "param_name:XXX" markers
        2. For each marked attribute, recursively map all its parameters
        3. Store the mapping as {stable_name: actual_path}
        """
        self._parameters_mappings = {} #The variable that contains all the mappings
        annotations = getattr(self.__class__, '__annotations__', {}) # (annotations are where we put "param_name:XXX" markers)
        
        for attr, ann in annotations.items():
            if not isinstance(ann, str):
                continue      
            if not ann.startswith('param_name:'):
                continue
            
            parameters_prefix = ann.split(':', 1)[1] #"param_name:transformer" -> "transformer"
            
            # Get the actual module instance from the attribute
            module = getattr(self, attr, None)
            
            #if not isinstance(module, nn.Module):
            #    continue
                        
            #if module is None:
            #    continue

                        
        for param_name, _ in module.named_parameters(recurse=True):
            raw = ".".join(p for p in param_name.split(".") if p not in ("model", "module", "inner")) #Strips model, module and inner.
            if raw in self._parameters_mappings:
                raw = param_name #If there is a collision, avoid the stripping
            self._parameters_mappings[f"{parameters_prefix}.{raw}"] = f"{attr}.{param_name}"

    
    def parameters_state_dict(self):
        """
        Get the model's state dict with stable parameter names. 
        """   
        mapping = self.get_parameters_name_mapping()     
        reverse = {v: k for k, v in mapping.items()}
        return {reverse.get(k, k): v for k, v in self.state_dict().items()}
    
    def load_parameters_state_dict(self, state_dict, strict=True, external_mapping: Optional[dict] = None):
        """
        Load a state dict that uses stable (or external) parameter names.
        Used when LOADING checkpoints.
        
        Args:
            state_dict: State dict with stable/external names
            strict: Whether to require exact key matching
            external_mapping: Optional mapping from external names to stable names
        
        Returns:
            Result from PyTorch's load_state_dict
        """
        # Get mapping (stable/external -> actual)
        # If external_mapping provided, this creates a composed mapping
        mapping = self.get_parameters_name_mapping(external_mapping)
        
        # Translate all keys in state_dict from stable/external to actual
        # For each key in the incoming state dict:
        #   - Look it up in mapping to get actual internal name
        #   - If not in mapping, keep the key as-is
        translated = {mapping.get(k, k): v for k, v in state_dict.items()}
        
        # Load the translated state dict using PyTorch's standard method
        return nn.Module.load_state_dict(self, translated, strict=strict)
    
    def stable_state_dict(self):
        return self.parameters_state_dict()

    def load_stable_state_dict(self, state_dict, strict=True, external_mapping: Optional[dict] = None):
        return self.load_parameters_state_dict(state_dict, strict=strict, external_mapping=external_mapping)


# Choose base class based on Lightning availability and config
if HAS_LIGHTNING:
    class MatformerModule(pl.LightningModule, ParametersRenamer):
        def __init__(self):
            super().__init__()
            # Indicate we are running under Lightning
            self.has_lightning = True

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
