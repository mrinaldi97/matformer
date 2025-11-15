import os
import importlib
from typing import Dict, List, Optional, Any, Callable
import pathlib
import sys
class MatformerRegistry:
    """
    =The Matformer Registry: customization of modules =
    The Matformer architecture tries to avoid every direct instantiation of hard-coded classes of neural network components.
    Instead, every implementation of the core components is managed by the MatformerRegistry. This means that it is possible
    to define modules for every component of the model, such as MLP, attention implementations, layer normalizations...
    In this way the model is not bound to specific implementation (such as vanilla pyTorch) but it is immediately possible
    to adopt different kernels without touching a single line of the core transformer implementation
    This gives extreme flexibility to the developer: just by putting python files in the "user-modules" folder and decorating 
    the class with a decorator it is possible to substitute every module of the network.
    By using the registry it is also possible to import customized experimental modules that can be hooked to many different entry points of
    the model, and, if necessary, they can be trained together with the rest of the network!
    Adding custom components to a transformer model has never been so easy.
    
    The registry has the following hierarchy:
        _registry (dict)
            category (dict) <= example "attention","norm","linear","embedding","mlp","loss"
            variant (dict) <= example "geglu","swiglu","rmsnorm","flash","gated_deltanet","mamba"...
            name (dict) <= example "torch","liger","nvidia-transformer-engine"... this is the name of the actual implementation
                name contains: 
                    * requires => List of required modules, useful for auto-fallback if not supported
                    * priority => The priority of different variants, ex torch 0, liger 1, nvidia 2 and so on
                    * metadata => Free dictionary of metadata
                    * params_names => *IMPORTANT* A mapping that renames the parameters to create universal checkpoints compatible also
                                        with different implementations
     
     == Add a custom implementation ==
			Put your script in the external_modules folder.
            Then, you don't have to do anything by hand, as the registration is handled by an easy to use decorator, example:
            from matformer.matformer_registry import registry
            @registry.register("mlp", "swiglu", "a_name_for_your_swiglu_kernel",
                   params_names={'w13.weight': 'gate_proj.weight', 
                                  'w2.weight': 'down_proj.weight'}, priority=100,requires=["torch","my_custom_library"])
            class MySwigluImplementation(nn.Module):
				(your code here)
				def _is_available():
					#Test logic here, for auto-fallback
				
			
			
            The MatformerRegister works in tandem with the CachedStuff object, thus it's very simple to define a module and let Matformer handle
            the picking of correct implementation for you:
            cache.registry.create("mlp", "swiglu", hidden_size=768, ffn_factor=4)
    These are the only requirements to seamless integrate different custom implementation of basic modules into a Matformer model! Just be careful to 
    match conventions for param_names if you want that your model correctly loads also with other implementation.
    
    == Adding custom modules to the model  ==
    
    The registry can also be used to add hooked modules/functions to any part of the model.
    The main transformer block exposes the following entry points:
			* pre_attn
			* pre_mlp
			* post_mlp
			* pre_output
	Let's say you want to add a custom module before the mlp. What you need to do is just to insert into the "external modules" directory a file
	my_module.py
		from matformer.matformer_registry import registry
		@register.registry("hook","name_of_my_hook","default")
		class ACustomModule(nn.Module):
			(your code...)
	
	Then, in the model config's JSON file the hook should be attached in this way:
		    "default_layer": { (or custom layer)
					[other params...]
				  "hooks": {"pre_mlp":"name_of_my_hook"}
				},
	If the modules contains trainable parameters, it will be trained togheter with the model.
	That's it! 
            
    """

    def __init__(self):
        self._registry = {}
        self._availability_cache = {} # 
        self._global_preferences = {} # It is possible to force preferences using set_preferences(preferences)
        self._env_overrides = {} # It is possible to force preferences using environment variables
        self._selected_implementations = {}  # Chosen implementation (category, variable) is cached so that search is performed only once
         
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
                 metadata: Optional[Dict[str, Any]] = None,
                 params_names:Optional[Dict]=None):
        """
        Register implementation in a two-level hierarchy.
        
        Args:
            category: Top-level category (e.g., 'attention', 'mlp', 'norm')
            variant: Semantic variant (e.g., 'flash', 'linear', 'sdpa' for attention)
            name: Implementation name (e.g., 'torch', 'triton', 'nvidia', 'liger')
            requires: Import/version requirements
            priority: Priority within variant (higher = higher priority)
            metadata: Additional info (tensor_order, causal_support, etc.)
        """
        def decorator(cls):
            if category not in self._registry:
                self._registry[category] = {}
            if variant not in self._registry[category]:
                self._registry[category][variant] = {'implementations': {}, 'priorities': []}
            
            # Store priorites in the implementation info
            self._registry[category][variant]['implementations'][name] = {
                'cls': cls,
                'requires': requires or [],
                'metadata': metadata or {},
                'priority': priority if priority is not None else 0,
                'params_names': params_names or {}
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
        Once an implementation is selected for a (category, variant) pair, it is cached
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

        
        def instantiate(name: str):
            impl = self._registry[category][variant]['implementations'][name]
            obj = impl['cls'](**kwargs)
            self._attach_metadata(obj, impl, variant, name)
            obj._params_names = impl.get('params_names', {})
            return obj

        # If already selected, instantiate
        if cache_key in self._selected_implementations:
            return instantiate(self._selected_implementations[cache_key])

        variant_info = self._registry[category][variant]

        # Sort candidates implementations: first explicit preferences then follows priority
        items = list(variant_info['implementations'].items())
        if preferred and preferred in variant_info['implementations']:
            items.sort(key=lambda x: 0 if x[0] == preferred else 1)
        else:
            items.sort(key=lambda x: x[1].get('priority', 0), reverse=True)

        candidates = [name for name, _ in items]
        last_error = None

        # Test each candidate in order until a viable one is found
        for name in candidates:
            if not self._is_available(category, variant, name):
                continue
            try:
                obj = instantiate(name)
                self._selected_implementations[cache_key] = name
                print(f"Implementation {name} was chosen for {category}.{variant}")
                return obj
            except Exception as e:
                last_error = e

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
        
