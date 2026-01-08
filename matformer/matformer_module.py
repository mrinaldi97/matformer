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
    def get_parameters_name_mapping(self, external_mapping: Optional[dict] = None):
        if not hasattr(self, '_modules') or len(self._modules) == 0:
            self._parameters_mappings = {}
            return
        if not hasattr(self, '_parameters_mappings'):
            self._build_parameters_mappings()
        
        if external_mapping:
            composed = {}
            for checkpoint_name, parameters_name in external_mapping.items():
                actual_name = self._parameters_mappings.get(parameters_name)
                if actual_name:
                    composed[checkpoint_name] = actual_name
                else:
                    composed[checkpoint_name] = parameters_name
            return composed
        
        return self._parameters_mappings
    
    def _build_parameters_mappings(self):
        """Recursively scan entire module hierarchy."""
        self._parameters_mappings = {}
        self._scan_tree(self, "", "")
        print(f"Mappings: {len(self._parameters_mappings)}")
        if not self._parameters_mappings:
            print("WARNING: No mappings found!")

    def _scan_tree(self, module: nn.Module, stable_prefix: str, actual_prefix: str):
        """
        Recursively scan module hierarchy, building mappings.
        
        Args:
            module: Current module to scan
            stable_prefix: Clean path (strips model/module/inner)
            actual_prefix: Real attribute path
        """
        # Get annotations from current module's class (these annotate CHILDREN)
        annotations = getattr(module.__class__, '__annotations__', {})
        
        # Get registry custom mappings for this module's parameters
        custom_maps = getattr(module, '_params_names', {})
        if custom_maps:
            print(f"   {module.__class__.__name__} => {custom_maps}")
        
        # Process this module's direct parameters (respect custom maps)
        for param_name, _ in module.named_parameters(recurse=False):
            if param_name in custom_maps:
                stable_name = custom_maps[param_name]
                print(f" {param_name} => {stable_name}")
            else:
                # Strip generic prefixes from parameter name itself
                parts = param_name.split('.')
                stable_name = '.'.join(p for p in parts if p not in ('model', 'module', 'inner'))
            
            # Build full paths
            full_stable = f"{stable_prefix}.{stable_name}" if stable_prefix else stable_name
            full_actual = f"{actual_prefix}.{param_name}" if actual_prefix else param_name
            
            self._parameters_mappings[full_stable] = full_actual
            print(f"{full_stable} => {full_actual}")
        # Process BUFFERS (add this block)
        for buffer_name, _ in module.named_buffers(recurse=False):
            # Use the same renaming logic as parameters
            clean_name = '.'.join(p for p in buffer_name.split('.') if p not in ('model', 'module', 'inner'))
            
            if buffer_name in custom_maps:  # Also respect _params_names for buffers
                stable_name = custom_maps[buffer_name]
                print(f"{buffer_name} => {stable_name}")
            else:
                stable_name = clean_name
            
            full_stable = f"{stable_prefix}.{stable_name}" if stable_prefix else stable_name
            full_actual = f"{actual_prefix}.{buffer_name}" if actual_prefix else buffer_name
            
            self._parameters_mappings[full_stable] = full_actual
            print(f"Mapped buffer: {full_stable} => {full_actual}")        
        # Recurse into children
        for child_name, child_module in module.named_children():
            # Check for annotation override (e.g., module: "param_name:mlp")
            override_name = None
            if child_name in annotations:
                ann = annotations[child_name]
                if isinstance(ann, str) and ann.startswith('param_name:'):
                    override_name = ann.split(':', 1)[1]
                    print(f"  {module.__class__.__name__}.{child_name} => {override_name}")
            
            # Determine stable child name:
            # - If annotation override exists: use it
            # - If child_name is generic (model/module/inner): skip this level
            # - Otherwise: use child_name
            if override_name:
                stable_child_name = override_name
            elif child_name in ('model', 'module', 'inner'):
                stable_child_name = None  # Skip generic names in stable path
            else:
                stable_child_name = child_name
            
            # Build stable prefix (skip if None)
            if stable_child_name:
                child_stable = f"{stable_prefix}.{stable_child_name}" if stable_prefix else stable_child_name
            else:
                child_stable = stable_prefix  # Pass through current prefix
            
            # Build actual prefix (always keep real names)
            child_actual = f"{actual_prefix}.{child_name}" if actual_prefix else child_name
            
            # Continue recursion
            self._scan_tree(child_module, child_stable, child_actual)
    
    def stable_state_dict(self):
        """Get renamed state dict without recursion issues."""
        mapping = self.get_parameters_name_mapping()
        reverse = {v: k for k, v in mapping.items()}
        raw_dict = nn.Module.state_dict(self)
        return {reverse.get(k, k): v for k, v in raw_dict.items()}

    def load_stable_state_dict(self, state_dict, strict=True, external_mapping=None):
        """Load renamed state dict."""
        mapping = self.get_parameters_name_mapping(external_mapping)
        translated = {mapping.get(k, k): v for k, v in state_dict.items()}
        return nn.Module.load_state_dict(self, translated, strict=strict)


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
            mapping = self.get_parameters_name_mapping()  # stable -> actual
            translated = {mapping.get(k, k): v for k, v in state_dict.items()}
            self._state_dict_translated = True  
            return super().load_state_dict(translated, strict=strict)
        
        def on_load_checkpoint(self, checkpoint: dict) -> None:
            """Fallback: translate if load_state_dict didn't run."""
            if not getattr(self, '_state_dict_translated', False):
                mapping = self.get_parameters_name_mapping()
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
