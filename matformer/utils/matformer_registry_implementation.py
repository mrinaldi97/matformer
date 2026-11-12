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
				
			
			
            It's very simple to define a module and let Matformer handle the picking of correct implementation for you:
            registry.create("mlp", "swiglu", hidden_size=768, ffn_factor=4)
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

    == Preferences and global uniqueness ==

    Implementation choice can be steered without touching code: registry.set_preferences({"attention":{"flash":"triton"}})
    sets it programmatically, or MATFORMER_ATTENTION_FLASH=triton sets it via environment (env wins over set_preferences).
    Once a (category, variant) pair is resolved, the choice is cached for the process and won't be re-tested.

    There is exactly one registry per process. matformer/matformer_registry.py builds it once at import time; every
    `from matformer.matformer_registry import registry` elsewhere in the codebase (or in your own modules) gets that
    same instance, already populated. Never call MatformerRegistry() directly.

    == Linking config dataclasses ==

    A dataclass can be bound to a registered implementation with register_dataclass, instead of carrying its own
    bookkeeping fields:

        @registry.register("saver", "JSONLSaver", "default")
        class JSONLSaver: ...

        @registry.register_dataclass("saver", "JSONLSaver", default=True)
        @dataclass
        class JSONLSaverConfig(SaverConfig):
            filename: str = "output.jsonl"

    Every YAML section is keyed by category and picks its variant with a "type" field
    (saver: {type: JSONLSaver, ...}); a section with no "type" falls back to whichever
    variant was registered with default=True. load_config.instantiate() then resolves a config
    straight through the registry, so the config dataclass itself carries no bookkeeping fields
    and no build() logic.


"""
import ast
import importlib
from collections import defaultdict
from functools import total_ordering
from pathlib import Path
from typing import Any, Optional
from matformer.utils import env_variables
import traceback

@total_ordering
class RegisteredImplementation:
    """
    This class represents a single implementation, ex. a kernel for a neural network operator (plain pyTorch, liger, transformer-engine, custom kernels...)
    Each implementation has a priority, defined with the decorator's call (see module-docs for details) and RegisteredImplementation istances
    can be ordered by priority, as they implement ordering operators __eq__ and __lt__
    get_implementations returns the package, if it's available, or None if any error occurr (ex, an ImportError)
    """
    def __init__(self, implementation_name: str, pkgname: str, registry: "MatformerRegistry",  _kwargs:dict):
        self.implementation_name = implementation_name
        self.pkgname = pkgname
        self.registry = registry
        self.priority = int(_kwargs.get('priority', 0))
        self.metadata = _kwargs.get('metadata', {})
        self.requires = _kwargs.get('requires', [])
        self.params_names = _kwargs.get('params_names', {})

    def __eq__(self, other):
            return self.priority == other.priority
    def __lt__(self, other):
        return self.priority < other.priority

    def get_implementation(self, requires: Optional[Any] = None):
        if requires is not None:
            raise NotImplementedError
        return self.registry.import_package(self.pkgname)  

    def instantiate(self, requires: Optional[Any] = None, **kwargs):
        package = self.get_implementation(requires)
        if package is None:
            return package
        # Attaching metadata and params_names to the instantiated object. This may be changed/deprecated in future
        if package:
            try:
                obj = package(**kwargs)
            except Exception as e:
                return e
            obj._matformer_metadata = self.metadata
            obj._params_names = self.params_names
            return obj            
class RegisteredVariant:
    """
    This class represents a variant, ex "cross_entropy_loss" for the category "loss".
    A variant can optionally have a "definition", that is, its own dataclass compatible with the
    MatformerConfig's logic. 
    A variant can have none, one or more implementations. In case more than one implementations are present,
    the preferred one is resolved through a priority decision (more info in module-level docs)
    """
    def __init__(self, category: str, registry: "MatformerRegistry", name: Optional[str] = None):
        self.name=name # The variant's name              
        self.category=category #The category to which this variant belongs
        self.registry=registry
        self.implementations=dict()
        self.definition=None # This stores the pkgname of the variant's dataclass

    def register_definition(self, pkgname: str):
        if self.definition is not None:
            print(f"WARNING: variant {self.name} (category {self.category}) already has a definition ({self.definition}); overwriting with {pkgname}.")
        self.definition=pkgname
    def register_implementation(self, implementation_name: str, pkgname: str, _kwargs: dict):
        self.implementations[implementation_name]=RegisteredImplementation(implementation_name=implementation_name, pkgname=pkgname, registry=self.registry, _kwargs=_kwargs)
    def get_instance(self, preferred: Optional[str] = None, requires: Optional[Any] = None, **kwargs):
        """
        Implementation priority: 
        1. implementation directly asked for with the parameter "preferred" (ex. 'torch')
        2. os environment variable MATFORMER_<CATEGORY>_<VARIANT> (ex. MATFORMER_ATTENTION_NORMAL=sdpa)
        3. priority queue, skipping implementations whose `requires` packages are unimportable
           or whose class-level is_available() returns False
        """
        if requires is not None:
            raise NotImplementedError
        if len(self.implementations)==0:
            print(f"Variant {self.name} doesn't have any registered implementation.")
        #1. Preferred parameter
        if preferred is not None:
            if preferred in self.implementations.keys():
                result=self.implementations[preferred].instantiate(requires, **kwargs)
                if result is not None:
                    if env_variables.registry_verbose:
                        print(f"Implementation {preferred} was chosen for {self.category}.{self.name} (preferred)")
                    return result
                else:
                    print(f"WARNING: preferred implementation {preferred} is not available.")
        #2. Environment variable
        import os
        env_key = f"MATFORMER_{self.category.upper()}_{self.name.upper()}"
        env_preferred = os.environ.get(env_key)
        if env_preferred and env_preferred in self.implementations:
            result = self.implementations[env_preferred].instantiate(requires, **kwargs)
            if result is not None and not isinstance(result, Exception):
                if env_variables.registry_verbose:
                    print(f"Implementation {env_preferred} was chosen for {self.category}.{self.name} ({env_key})")
                return result
            print(f"WARNING: environment-selected implementation '{env_preferred}' failed; falling back to the priority queue.")
        #3. Priority queue
        last_exception=""
        for impl in sorted(self.implementations.values(), reverse=True): #From highest priority to lowest
            if not self.registry.implementation_available(impl):
                continue
            result = impl.instantiate(requires=requires, **kwargs)
            if isinstance(result,Exception):
                last_exception=result
                continue
            else:
                if env_variables.registry_verbose:
                    print(f"Implementation {impl.implementation_name} was chosen for {self.category}.{self.name}")
                return result
        # X. Everything failed :(
        print(f"All the implementations of variant {self.name} category {self.category} failed. Tested implementations: {self.implementations}")
        raise result
        raise Exception(f"All the implementations of variant {self.name} category {self.category} failed. Tested implementations: {self.implementations}")
        return None

    def get_definition(self):
        if not self.definition:
            print(f"Variant {self.name} doesn't have an associated dataclass.")
            return None
        cls = self.registry.import_package(self.definition)
        if cls and hasattr(cls, '_matformer_category'):
            cls._matformer_category = self.category
            cls._matformer_variant = self.name
        return cls

class RegisteredCategory:
    """
    This class stores information about a specific category, the top-level hierarchy of the register.

    """
    def __init__(self, registry: "MatformerRegistry", name: str):
        self.name=name
        self.registry=registry
        self.variants = defaultdict(lambda: RegisteredVariant(category=self.name, registry=registry))
        self.default_variant=None
        self.has_definition=False
        self.is_multi=None
    def register_definition(self, variant: str, pkgname: str, default: bool = False, is_multi: bool = False):
        if self.is_multi is None:
            self.is_multi=is_multi
        else:
            assert is_multi == self.is_multi,f"Error: be sure that all the dataclasses definitions for the category {self.name} are of the same type (MatformerConfig OR MatformerMultiConfig)"
        self.variants[variant].name = variant
        self.variants[variant].register_definition(pkgname)
        if default:
            if self.default_variant is not None:
                print(f"Warning: you've set more than one default variant for category {self.name}. Variant {variant} is overriding {self.default_variant} as default for this category.")
            self.default_variant=variant
        self.has_definition=True
    def register_implementation(self, variant: str, implementation: str, pkgname: str, _kwargs: dict):
        self.variants[variant].name = variant 
        self.variants[variant].register_implementation(implementation_name=implementation, pkgname=pkgname, _kwargs=_kwargs)
    def get_definition(self, variant: Optional[str] = None):
        if variant is None:
            assert self.default_variant is not None,f"FATAL: Category {self.name} doesn't have a default variant. Please specify the desired variant. Available variiants: {self.variants}"
            variant=self.default_variant
        if variant not in self.variants.keys():
            raise ValueError(f"Category {self.name} doesn't have variant {variant}. Available variants: {self.variants.keys()}")
        return self.variants[variant].get_definition()
    def get_instance(self, variant: str, preferred: Optional[str] = None, requires: Optional[Any] = None, **kwargs):
        if variant not in self.variants.keys():
            raise ValueError(f"Category {self.name} doesn't have variant {variant}. Available variants: {self.variants.keys()}")
        return self.variants[variant].get_instance(preferred, requires, **kwargs)


class MatformerRegistry:
    def __init__(self):
        self.root_modname='matformer'
        reg_path=Path(__file__) 
        #reg_path=Path("/home/matteo/Ricerca/miei_progetti/matformer/matformer/utils/reg.py")
        if reg_path.parent.name!='utils':
            print(f"WARNING: it seems that the {reg_path.name} file, which implements the registry, is not in the 'utils' folder. Execution will continue, but correctness is not guaranteed.")
        self.root_path=reg_path.parent.parent
        self.root_modname=self.root_path.name
        self.modname_filepath={} #k=module's dotted path (modname), v=python file path
        self.pkgname_modname={} #k=pkgname v=modname
        self.register_types=['register', 'register_dataclass'] #The decorators for which the registration will be invoked
        self.registry = defaultdict(lambda: RegisteredCategory(registry=self, name=None))
        self.allow_overwrite=True #
        # The internal cache of imported modules and packages
        self.imported_modules={}
        self.failed_modules=[]
        self.imported_packages={}
        self.failed_packages=[]
        # To reconstruct the inheritance graph
        self.class_bases={}
        self.class_lines={}          # class name -> (lineno, end_lineno)

    # (mostly) internal functions
    def populate_registry(self):
        """
        Scans all the py files in matformer's root and populate the registry
        """
        self.walk_py_files()
        self.ast_scan()

    def register_package(self, pkgname: str, modname: str):
        """
        Associate a pkgname to its modname and write this association in the lookup table "self.pkgname_modname", indexed by pkgname
        Note that register-wide (at a higher level than single variants or categories) packages must have all distinct names,
        as the resolve is at registry level, not at single variant/category one. 
        """
        if pkgname in self.pkgname_modname.keys():
            previous_module=self.pkgname_modname[pkgname]
            previous_file=self.modname_filepath[previous_module]
            current_file=self.modname_filepath[modname]
            if self.allow_overwrite is False:
                raise ValueError(f"Package {pkgname} is already present in module {self.pkgname_modname[pkgname]}. Please use another name. ")
            else:
                print(f"WARNING: {pkgname} is defined both in {previous_file} (previous) and {current_file} (current). Current is overwriting previous. ")
        self.pkgname_modname[pkgname]=modname

    def import_module(self, modname: str):
        """
        A specific module is imported. A cache system ensures that if a modules' import fails, no more attempt will be done (self.failed_modules),
        conversely, modules are imported only once and then cached (self.imported_modules)
        Accept modname (str) as argument, returns the module with the specified name.
        """
        if modname in self.failed_modules:
            return None
        if modname in self.imported_modules:
            return self.imported_modules[modname]
        try:
            self.imported_modules[modname] = importlib.import_module(modname)
            return self.imported_modules[modname]
        except Exception:
            print(f"Import of {modname} in {self.modname_filepath[modname]} failed:")
            traceback.print_exc()
            self.failed_modules.append(modname)
            return None

    def import_package(self, pkgname: str):
        """
        A specific package is imported. A cache system ensures that if a package' import fails, no more attempt will be done (self.failed_packages),
        conversely, packages are imported only once and then cached (self.imported_packages). It works in tandem with import_module to import the module 
        containing the packages. On import_module's failure, also import package will fail. 
        Accept pkgname (str) as argument, returns the pkgname with the specified name.
        """
        if pkgname in self.failed_packages:
            return None
        if pkgname in self.imported_packages.keys():
            return self.imported_packages[pkgname]
        modname=self.pkgname_modname[pkgname]
        _module=self.import_module(modname)
        if not _module:
            self.failed_packages.append(pkgname)
        else:
            try:
                self.imported_packages[pkgname]=getattr(_module, pkgname)
            except:
                self.failed_packages.append(pkgname)
        return self.import_package(pkgname)

    def walk_py_files(self):
        """Get dotted_modname and filepath for every .py file under root_path, without importing anything """
        for py in self.root_path.rglob("*.py"):
            parts = py.relative_to(self.root_path).with_suffix("").parts
            modname = ".".join((self.root_modname, *parts)) if parts else self.root_modname
            self.modname_filepath[modname]=py

    def ast_scan(self):
        """ Perform an AST scan of all the discovered python files
        Store the decorators of modules that use the registry
        """
    def ast_scan(self):
        """Two-pass AST scan.

        Pass 1: collect class bases and line numbers for every class in every file,
                so the inheritance graph is complete.
        Pass 2: process @register / @register_dataclass decorators, which may
                need to walk the (now fully-populated) inheritance graph.
        """
        # ---- Pass 0: parse every file once and keep the trees around ----
        trees = {}
        for modname, filepath in self.modname_filepath.items():
            try:
                trees[modname] = ast.parse(filepath.read_text())
            except Exception as e:
                print(f"File {filepath} not loaded, exception: {e}")

        # ---- Pass 1: build the inheritance graph ----
        for modname, tree in trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    base_names = {
                        b.id if isinstance(b, ast.Name) else getattr(b, 'attr', None)
                        for b in node.bases
                    }
                    # Drop the None that comes from non-Name/Attribute bases
                    base_names.discard(None)
                    self.class_bases[node.name] = base_names
                    self.class_lines[node.name] = (node.lineno, node.end_lineno)

        # ---- Pass 2: invoke the registration logic ----
        for modname, tree in trees.items():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    if not isinstance(decorator.func, ast.Attribute):
                        continue
                    if decorator.func.attr not in self.register_types:
                        continue
                    self.register_package(pkgname=node.name, modname=modname)
                    self._register(node=node, decorator_call=decorator, modname=modname)

    def _register(self, node: ast.ClassDef, decorator_call: ast.Call, modname: str):
        pkgname=node.name
        args=[a.value for a in decorator_call.args if isinstance(a,ast.Constant)]
        _kwargs={}
        for kw in decorator_call.keywords:
            if kw.arg is not None:
                try:
                    _kwargs[kw.arg]=ast.literal_eval(kw.value)
                except Exception as e:
                    print(f"Exception in {modname} kwarg evaluation: {kw.arg} {e}")  
        _type=decorator_call.func.attr
        if _type=='register':
            assert len(args)==3
            category=args[0]
            variant=args[1]
            implementation=args[2]
            self.registry[category].name = self.registry[category].name or category 
            self.registry[category].register_implementation(variant=variant, implementation=implementation, pkgname=pkgname, _kwargs=_kwargs)
        elif _type == 'register_dataclass':
            assert len(args)==2
            category=args[0]
            variant=args[1]
            is_multi = self._class_inherits_from(pkgname, 'MatformerMultiConfig')
            self.registry[category].name = self.registry[category].name or category
            self.registry[category].register_definition(variant=variant, pkgname=pkgname, default=_kwargs.get('default', False), is_multi=is_multi)
        else:
            print(f"WARNING: wrong {_type} registry call in {modname}")  

    def _class_inherits_from(self, clsname: str, target: str, visited: Optional[set] = None) -> bool:
        """
        Walk up the inheritance chain of 'clsname' (using base-class names collected via AST) to check whether it inherits from 'target'.
        """
        if visited is None:
            visited = set()
        if clsname in visited:
            return False  # avoid infinite loops on cyclic/self-referential bases
        visited.add(clsname)

        direct_bases = self.class_bases.get(clsname, set())
        if target in direct_bases:
            return True

        for base_name in direct_bases:
            if self._class_inherits_from(base_name, target, visited):
                return True

        return False
    def register(self, *args, **kwargs):
        def decorator(cls):
            return cls  # The decorator is not actually used, but parsed through AST
        return decorator
        
    def register_dataclass(self, *args, **kwargs):
        def decorator(cls):
            return cls  # The decorator is not actually used, but parsed through AST
        return decorator


    # Main API
    def implementation_available(self, impl: RegisteredImplementation) -> bool:
        """True if every package in impl.requires imports AND the class-level
        is_available() (if defined) returns True. Used by the priority-queue fallback
        so implementations that merely defer missing imports to forward() are skipped."""
        for pkg in impl.requires:
            try:
                importlib.import_module(pkg)
            except Exception:
                return False
        cls = impl.get_implementation()
        if cls is None:
            return False
        if hasattr(cls, "is_available") and not cls.is_available():
            return False
        return True

    def get_definition(self, category: str, variant: Optional[str] = None):
        if category not in self.registry.keys():
            raise ValueError(f"{category} is not a category registered in the Register. Available categories: {self.registry.keys()}")
        return self.registry[category].get_definition(variant)
    def get_definition_defaults(self,category:str, variant: Optional[str]=None):
        return self.get_definition(category,variant).from_dict({}).asdict()
    def create(self, category: str, variant: str, preferred: Optional[str] = None, requires: Optional[str] = None, **kwargs):
        if category not in self.registry.keys():
            raise ValueError(f"{category} is not a category registered in the Register. Available categories: {self.registry.keys()}")        
        return self.registry[category].get_instance(variant, preferred, requires, **kwargs)

    def list_categories(self, with_definition=True, without_definition=True, listed=True, single=True):
        return [c for c, cat in self.registry.items()
                if (with_definition if cat.has_definition else without_definition)
                and (not cat.has_definition or (listed if cat.is_multi else single))]
    # Documentation helpers
    def _get_filepath(self, pkgname: str, relative: bool = True) -> Optional[Path]:
        """Return the file path for a given package name, if known.
        
        Args:
            pkgname: The package/class name to look up
            relative: If True, return path relative to self.root_path; 
                     if False, return absolute path
        """
        modname = self.pkgname_modname.get(pkgname)
        if modname:
            path = self.modname_filepath.get(modname)
            if path and relative:
                try:
                    return path.relative_to(self.root_path)
                except ValueError:
                    # If path is not under root_path, return as-is
                    return path
            return path
        return None

    def get_registry_info(
        self,
        category: Optional[str] = None,
        variant: Optional[str] = None,
        implementation: Optional[str] = None,
        include_definitions: bool = True,
        include_implementations: bool = True,
        as_json: bool = False,
        relative_filepath: bool = False
    ):
        """
        Return a nested dictionary describing the registry contents.
        Optionally filter by category, variant, or implementation name.
        If `as_json` is True, return a JSON string instead of a dict.
        """
        info = {}
        for cat_name, cat_obj in self.registry.items():
            if category is not None and cat_name != category:
                continue

            cat_info = {
                "default_variant": cat_obj.default_variant,
                "variants": {},
            }

            for var_name, var_obj in cat_obj.variants.items():
                if variant is not None and var_name != variant:
                    continue

                var_info = {}

                if include_definitions and var_obj.definition:
                    filepath = self._get_filepath(var_obj.definition, relative=relative_filepath)
                    line_start, line_end = self.class_lines.get(var_obj.definition) or (None, None)
                    var_info["definition"] = {
                        "pkgname": var_obj.definition,
                        "filepath": str(filepath) if filepath else None,
                        "line_start": line_start,
                        "line_end": line_end
                    }

                if include_implementations:
                    impls = {}
                    for impl_name, impl_obj in var_obj.implementations.items():
                        if implementation is not None and impl_name != implementation:
                            continue
                        filepath = self._get_filepath(impl_obj.pkgname, relative=relative_filepath)
                        line_start, line_end = self.class_lines.get(impl_obj.pkgname) or (None, None)
                        impls[impl_name] = {
                            "pkgname": impl_obj.pkgname,
                            "filepath": str(filepath) if filepath else None,
                            "line_start": line_start,
                            "line_end": line_end,                            
                            "priority": impl_obj.priority,
                            "metadata": impl_obj.metadata,
                        }
                    if impls:  # only include if non‑empty after filtering
                        var_info["implementations"] = impls

                if var_info:
                    cat_info["variants"][var_name] = var_info

            if cat_info["variants"]:
                info[cat_name] = cat_info

        if as_json:
            import json
            return json.dumps(info, indent=2, default=str)
        return info

    def print_registry(
        self,
        category: Optional[str] = None,
        variant: Optional[str] = None,
        implementation: Optional[str] = None,
        relative_filepath: Optional[bool] = False
    ):
        """Print a human‑readable tree of the registry (with optional filters)."""
        info = self.get_registry_info(category, variant, implementation, relative_filepath)
        for cat_name, cat_info in info.items():
            print(f"Category: {cat_name}")
            if cat_info.get("default_variant"):
                print(f"  Default variant: {cat_info['default_variant']}")
            for var_name, var_info in cat_info["variants"].items():
                print(f"  Variant: {var_name}")
                if "definition" in var_info:
                    d = var_info["definition"]
                    print(f"    Definition: {d['pkgname']} (file: {d['filepath']}, lines {d['line_start']}-{d['line_end']})")
                if "implementations" in var_info:
                    for impl_name, impl_info in var_info["implementations"].items():
                        print(f"    Implementation: {impl_name} (priority={impl_info['priority']})")
                        print(f"      Package: {impl_info['pkgname']}")
                        print(f"      File: {impl_info['filepath']} (lines {impl_info['line_start']}-{impl_info['line_end']})")
                        if impl_info.get("metadata"):
                            print(f"      Metadata: {impl_info['metadata']}")    
