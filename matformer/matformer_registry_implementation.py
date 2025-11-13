import os
import importlib
from typing import Dict, List, Optional, Any, Callable
import pathlib
import sys
class MatformerRegistry:
    def __init__(self):
        self._registry = {}
        self._availability_cache = {}
        self._global_preferences = {}
        self._env_overrides = {}
        self._selected_implementations = {}  # Cache selected impl per (category, variant)
         
    def load_modules(self):
        base_dir = pathlib.Path(__file__).resolve().parent
        for folder in ("matformer/attention", "matformer/modules", "user-modules","attention","modules"):
            folder_path = base_dir / folder
            if not folder_path.exists():
                continue
            for py_file in folder_path.glob("*.py"):
                module_name = f"matformer.{folder.replace('/', '.')}.{py_file.stem}"
                
                importlib.import_module(module_name)                        
    def register(self, category: str, variant: str, name: str, *, 
                 requires: Optional[List[str]] = None, 
                 priority: Optional[int] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Register implementation in a two-level hierarchy.
        
        Args:
            category: Top-level category (e.g., 'attention', 'mlp', 'norm')
            variant: Semantic variant (e.g., 'flash', 'linear', 'sdpa' for attention)
            name: Implementation name (e.g., 'torch', 'triton', 'nvidia', 'liger')
            requires: Import/version requirements
            priority: Priority within variant (lower = higher priority)
            metadata: Additional info (tensor_order, causal_support, etc.)
        """
        def decorator(cls):
            if category not in self._registry:
                self._registry[category] = {}
            if variant not in self._registry[category]:
                self._registry[category][variant] = {'implementations': {}, 'priorities': []}
            
            # Store numeric priority in the implementation info
            self._registry[category][variant]['implementations'][name] = {
                'cls': cls,
                'requires': requires or [],
                'metadata': metadata or {},
                'priority': priority if priority is not None else 0
            }
            
            # Optionally keep simple registration order
            if name not in self._registry[category][variant]['priorities']:
                self._registry[category][variant]['priorities'].append(name)
            
            return cls
        return decorator
    
    def set_preferences(self, preferences: Dict[str, Dict[str, str]]):
        self._global_preferences = preferences.copy()
        
    def _resolve_preference(self, category: str, variant: str) -> Optional[str]:
        """Resolve preference with environment override > config > None."""
        # Example: MATFORMER_ATTENTION_FLASH=triton
        env_key = f"MATFORMER_{category.upper()}_{variant.upper()}"
        if env_key in os.environ:
            return os.environ[env_key]
        
        if category in self._global_preferences:
            if variant in self._global_preferences[category]:
                return self._global_preferences[category][variant]
        
        return None
    
    def create(self, category: str, variant: str, preferred: Optional[str] = None, **kwargs):
        """
        Create instance with preference resolution and fallback.
        Once an implementation is selected for a (category, variant) pair, it's cached
        and reused without re-testing.
        
        Args:
            category: e.g., 'attention'
            variant: e.g., 'flash'
            preferred: Override preference for this call
            **kwargs: Args passed to implementation __init__
        """
        if category not in self._registry or variant not in self._registry[category]:
            raise ValueError(f"Unknown {category}.{variant}")
        
        cache_key = (category, variant)
        
        # Check if we've already selected an implementation
        if cache_key in self._selected_implementations:
            selected_name = self._selected_implementations[cache_key]
            impl_info = self._registry[category][variant]['implementations'][selected_name]
            instance = impl_info['cls'](**kwargs)
            self._attach_metadata(instance, impl_info, variant, selected_name)
            return instance
        
        variant_info = self._registry[category][variant]
        
        # Build candidate list with numeric priority
        all_candidates = list(variant_info['implementations'].items())
        
        # Explicit preferred override has top priority
        if preferred and preferred in variant_info['implementations']:
            all_candidates.sort(key=lambda x: 0 if x[0] == preferred else 1)
        else:
            # Sort by numeric priority (higher = higher precedence)
            all_candidates.sort(key=lambda x: x[1].get('priority', 0), reverse=True)
        
        candidates = [name for name, _info in all_candidates]
        
        # Try each candidate
        last_error = None
        for name in candidates:
            if not self._is_available(category, variant, name):
                continue
            
            impl_info = variant_info['implementations'][name]
            try:
                instance = impl_info['cls'](**kwargs)
                self._attach_metadata(instance, impl_info, variant, name)
                
                # Cache this selection for future creates
                self._selected_implementations[cache_key] = name
                print(f"Implementation {name} was chosen for {category}.{variant}")
                return instance
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(
            f"No available implementation for {category}.{variant}. "
            f"Tried: {candidates}. Last error: {last_error}"
        )
    
    def _attach_metadata(self, instance, impl_info: Dict, variant: str, name: str):
        """Attach metadata to instance for introspection."""
        instance._matformer_metadata = impl_info['metadata']
        instance._matformer_variant = variant
        instance._matformer_implementation = name
    
    def get_metadata(self, category: str, variant: str, name: str) -> Dict[str, Any]:
        """Retrieve metadata for an implementation."""
        return self._registry[category][variant]['implementations'][name]['metadata']
    
    def _is_available(self, category: str, variant: str, name: str) -> bool:
        """
        Check availability with caching.
        Supports both declarative requirements and custom test() method.
        """
        cache_key = (category, variant, name)
        if cache_key in self._availability_cache:
            return self._availability_cache[cache_key]
        
        impl_info = self._registry[category][variant]['implementations'][name]
        cls = impl_info['cls']
        
        # First check declarative requirements
        for requirement in impl_info['requires']:
            try:
                self._check_requirement(requirement)
            except (ImportError, RuntimeError):
                self._availability_cache[cache_key] = False
                return False
        
        # Then check custom test method if present
        if hasattr(cls, 'is_available'):
            try:
                result = cls.is_available()
                self._availability_cache[cache_key] = result
                return result
            except Exception:
                self._availability_cache[cache_key] = False
                return False
        
        # If no custom test and requirements passed, it's available
        self._availability_cache[cache_key] = True
        return True
    
    def _check_requirement(self, requirement: str):
        """Parse and check a requirement string."""
        # Handle version specs: 'torch>=2.0', 'flash_attn'
        if '>=' in requirement or '==' in requirement or '<=' in requirement:
            # Extract module name before version operator
            for op in ['>=', '==', '<=', '>', '<']:
                if op in requirement:
                    module_name = requirement.split(op)[0].strip()
                    break
            importlib.import_module(module_name)
            # TODO: actual version comparison
        else:
            importlib.import_module(requirement)
    
    def reset_selections(self):
        """Reset cached implementation selections. Useful for testing."""
        self._selected_implementations = {}
    
    def get_selected_implementation(self, category: str, variant: str) -> Optional[str]:
        """Get the currently selected implementation for a (category, variant) pair."""
        return self._selected_implementations.get((category, variant))
        
