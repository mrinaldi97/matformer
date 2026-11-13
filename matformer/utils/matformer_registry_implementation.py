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
                    * requires => List of required libraries, useful for auto-fallback if not supported
                    * priority => The priority of different variants, ex torch 0, liger 1, nvidia 2 and so on
                    * constraints => Constraints (ex. ["is_training"]) this implementation is compatible with;
                                    see the "Constraints" section below
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
				@staticmethod
				def _test():
					# Return True (or simply don't raise) if this kernel really works on this machine
				
			
			
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

    == Preferences, availability tests and constraints ==

    Implementation choice can be steered without touching code: the environment variable
    MATFORMER_<CATEGORY>_<VARIANT> (ex. MATFORMER_ATTENTION_NORMAL=sdpa) forces a specific
    implementation, and create(..., preferred='liger') does the same programmatically. Otherwise the
    priority queue decides. Once an implementation is chosen it is instantiated; if it cannot be
    built, the registry simply tries the next one.

    Availability tests: an implementation may define a class-level `_test()` static method, a few
    lines that really exercise the kernel (import it and run the smallest possible call). It is run
    by the registry at instantiation time - never during the AST discovery, which stays completely
    lazy - and its outcome is stored globally, so a whole model pays the cost only once and a
    failing kernel (ex. flash-attn installed but useless on a CPU-only machine, or compiled for
    another CUDA version) is skipped forever after without repeating the test. Any raised exception
    or a False return means "don't use me here".

    Constraints describe what an implementation can be used for, ex. an hypothetical FlashInfer
    kernel usable for inference but not for training:

        @registry.register("attention", "normal", "flashinfer", priority=20, constraints=["is_inference"])

    registry.create("attention", "normal", constraints=["is_training"], ...) then skips it, even if it
    has the highest priority; the same happens with registry.set_default_constraints(["is_training"]),
    which activates a constraint for every subsequent create() call. An implementation that declares
    no constraints is considered universal and is never filtered out; a kernel can also declare the
    constraints it does NOT support, ex. constraints={'is_training': False}, in which case everything
    else stays compatible.

    == requires, constraints, metadata: three different things ==

    They are easy to mix up, so the rule of thumb is:
        * requires     -> external LIBRARIES the implementation needs to exist at all
                         (ex. ['flash_attn']); unimportable => the implementation is not a candidate.
        * constraints  -> whether the implementation MAY BE SELECTED in the current context
                         (ex. is_training, softcap); not satisfied => skip it and take the next one.
        * metadata     -> how to CALL an implementation that has already been selected (tensor order,
                         packed-qkv support...); it is attached to the instance as _matformer_metadata
                         and it never influences the choice.
    A capability whose absence forces a fallback (ex. flash-attn not doing tanh softcapping) is a
    constraint, not a metadata: declaring it as metadata would leave the caller to discover, at
    forward time, that the chosen kernel cannot do the job.

    == Errors ==

    Every error raised by the registry is a RegistryError subclass: ImplementationNotAvailable
    (missing dependency or failed _test), ImplementationInitError (the constructor of the chosen
    implementation raised; the original traceback is preserved, so it points at the user's module and
    not at the registry) and NoImplementationAvailable (every candidate failed: the message lists the
    reason of each attempt). A failing implementation never silently ends the search and never
    returns an object that isn't the requested module.

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
import inspect
import os
from collections import defaultdict
from functools import total_ordering
from pathlib import Path
from typing import Any, Optional
from matformer.utils import env_variables
import traceback


class RegistryError(Exception):
    """Base class of every error raised by the registry itself."""


class ImplementationNotAvailable(RegistryError):
    """An implementation cannot run on this system (missing dependency, broken import or failed _test())."""


class ImplementationInitError(RegistryError):
    """The implementation was selected, but its constructor raised.
    The original exception (with its original traceback) is preserved as __cause__, so the
    traceback the user reads always starts inside their own implementation, not in the registry."""


class NoImplementationAvailable(RegistryError):
    """Every candidate implementation of a variant failed or was filtered out."""


def _keep_class(cls):
    """The registry decorators are pure annotations: the registration itself is done by the AST
    scan of the package, so at import time they must leave the class untouched."""
    return cls


def normalize_constraints(constraints: Any) -> dict:
    """Canonical form of a constraint specification: a {name: bool} dict.

    Accepts None, a single name (str), a list/tuple of names or an existing mapping.
    On a registered implementation a name mapped to False explicitly declares its incompatibility
    with that constraint (ex. constraints={'is_training': False}); on create() it simply means that
    the constraint is not active.
    """
    if constraints is None:
        return {}
    if isinstance(constraints, str):
        return {constraints: True}
    if isinstance(constraints, dict):
        return {str(name): bool(value) for name, value in constraints.items()}
    return {str(name): True for name in constraints}


@total_ordering
class RegisteredImplementation:
    """
    This class represents a single implementation, ex. a kernel for a neural network operator (plain pyTorch, liger, transformer-engine, custom kernels...)
    Each implementation has a priority, defined with the decorator's call (see module-docs for details) and RegisteredImplementation istances
    can be ordered by priority, as they implement ordering operators __eq__ and __lt__
    get_implementation returns the package, if it's available, or None if any error occurr (ex, an ImportError);
    instantiate runs the availability _test() of the class (see MatformerRegistry.run_implementation_test),
    builds the object and raises a RegistryError explaining why it cannot be used here, if that's the case.
    """
    def __init__(self, implementation_name: str, pkgname: str, registry: "MatformerRegistry",
                 _kwargs: dict, category: Optional[str] = None, variant: Optional[str] = None):
        self.implementation_name = implementation_name
        self.pkgname = pkgname
        self.registry = registry
        self.category = category # Set by the owning RegisteredVariant, only used for readable messages
        self.variant = variant
        self.priority = int(_kwargs.get('priority', 0))
        self.metadata = _kwargs.get('metadata', {})
        self.requires = _kwargs.get('requires', [])
        self.params_names = _kwargs.get('params_names', {})
        self.constraints = normalize_constraints(_kwargs.get('constraints', None))

    def __eq__(self, other):
        if not isinstance(other, RegisteredImplementation):
            return NotImplemented
        return self.priority == other.priority
    def __lt__(self, other):
        if not isinstance(other, RegisteredImplementation):
            return NotImplemented
        return self.priority < other.priority

    def __repr__(self):
        return f"{self.describe()}"

    def describe(self) -> str:
        """Human-readable identification of this implementation, used in warnings and errors."""
        where = f"{self.category}.{self.variant}" if self.category else "the registry"
        location = self.registry._get_filepath(self.pkgname, relative=True)
        at = f", {location}" if location else ""
        return f"implementation '{self.implementation_name}' of {where} (class {self.pkgname}{at})"

    def get_implementation(self):
        return self.registry.import_package(self.pkgname)  

    def instantiate(self, **kwargs):
        package = self.get_implementation()
        if package is None:
            raise ImplementationNotAvailable(f"{self.describe()} could not be imported")
        self.registry.run_implementation_test(self)
        # Attaching metadata and params_names to the instantiated object. This may be changed/deprecated in future
        try:
            obj = package(**kwargs)
        except Exception as e:
            raise ImplementationInitError(
                f"{self.describe()} raised {type(e).__name__} while being constructed: {e}"
            ) from e
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
        self.implementations[implementation_name]=RegisteredImplementation(
            implementation_name=implementation_name, pkgname=pkgname, registry=self.registry, _kwargs=_kwargs,
            category=self.category, variant=self.name)
    def get_instance(self, preferred: Optional[str] = None, constraints: Optional[Any] = None, **kwargs):
        """
        Implementation priority: 
        1. implementation directly asked for with the parameter "preferred" (ex. 'torch')
        2. os environment variable MATFORMER_<CATEGORY>_<VARIANT> (ex. MATFORMER_ATTENTION_NORMAL=sdpa)
        3. priority queue, skipping implementations whose `requires` packages are unimportable,
           whose _test() failed (see MatformerRegistry.run_implementation_test) or that don't
           support the active constraints

        A candidate that fails never ends the search: it is just recorded, and the next one is
        tried. Only when every candidate failed NoImplementationAvailable is raised, listing the
        reason of each attempt (the last construction error, if any, is chained as __cause__).
        """
        active_constraints = self.registry.active_constraints(constraints)
        attempts = []            # [(implementation name, reason of the failure)]
        attempted = set()
        last_error: Optional[Exception] = None

        def try_implementation(impl: RegisteredImplementation, source: str):
            nonlocal last_error
            attempted.add(impl.implementation_name)
            reason = self.registry.implementation_unavailable_reason(impl)
            if reason is None:
                reason = self.registry.constraints_unavailable_reason(impl, active_constraints)
            if reason is not None:
                attempts.append((impl.implementation_name, reason))
                return None
            try:
                instance = impl.instantiate(**kwargs)
            except RegistryError as error:
                last_error = error
                attempts.append((impl.implementation_name, str(error)))
                return None
            if env_variables.registry_verbose:
                self.registry.announce(self.category, self.name, impl.implementation_name, source, active_constraints)
            return instance

        if len(self.implementations)==0:
            raise NoImplementationAvailable(f"Variant {self.name} (category {self.category}) doesn't have any registered implementation.")
        #1. Preferred parameter
        if preferred is not None:
            if preferred not in self.implementations:
                print(f"WARNING: preferred implementation '{preferred}' is not registered for "
                      f"{self.category}.{self.name}. Available implementations: {list(self.implementations)}")
            else:
                result = try_implementation(self.implementations[preferred], 'preferred')
                if result is not None:
                    return result
                print(f"WARNING: preferred implementation {preferred} is not available for "
                      f"{self.category}.{self.name}; falling back to the priority queue.")
        #2. Environment variable
        env_key = f"MATFORMER_{self.category.upper()}_{self.name.upper()}"
        env_preferred = os.environ.get(env_key)
        if env_preferred and env_preferred not in attempted: # "attempted": already tried as preferred
            if env_preferred in self.implementations:
                result = try_implementation(self.implementations[env_preferred], env_key)
                if result is not None:
                    return result
                print(f"WARNING: environment-selected implementation '{env_preferred}' failed; falling back to the priority queue.")
            else:
                print(f"WARNING: {env_key}='{env_preferred}' is not a registered implementation for "
                      f"{self.category}.{self.name}. Available implementations: {list(self.implementations)}")
        #3. Priority queue
        for impl in sorted(self.implementations.values(), key=lambda i: (-i.priority, i.implementation_name)): #From highest priority to lowest
            if impl.implementation_name in attempted: # Already tried (preferred or env var), don't report it twice
                continue
            result = try_implementation(impl, 'priority queue')
            if result is not None:
                return result
        # X. Everything failed :(
        details = "\n".join(f"  - {name}: {reason}" for name, reason in attempts) or "  (no implementation was eligible)"
        constraint_note = f" (active constraints: {sorted(active_constraints)})" if active_constraints else ""
        raise NoImplementationAvailable(
            f"All the implementations of variant '{self.name}' category '{self.category}' failed{constraint_note}.\n"
            f"Attempted implementations:\n{details}"
        ) from last_error

    def get_definition(self):
        if not self.definition:
            print(f"Variant {self.name} doesn't have an associated dataclass.")
            return None
        cls = self.registry.import_package(self.definition)
        if cls is not None: # Stamped so that a config knows which category/variant it belongs to
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
    def get_instance(self, variant: str, preferred: Optional[str] = None, constraints: Optional[Any] = None, **kwargs):
        if variant not in self.variants.keys():
            raise ValueError(f"Category {self.name} doesn't have variant {variant}. Available variants: {self.variants.keys()}")
        return self.variants[variant].get_instance(preferred=preferred, constraints=constraints, **kwargs)


class MatformerRegistry:
    def __init__(self):
        reg_path = Path(__file__)
        if reg_path.parent.name!='utils':
            print(f"WARNING: it seems that the {reg_path.name} file, which implements the registry, is not in the 'utils' folder. Execution will continue, but correctness is not guaranteed.")
        self.root_path=reg_path.parent.parent
        self.root_modname=self.root_path.name
        self.modname_filepath={} #k=module's dotted path (modname), v=python file path
        self.pkgname_modname={} #k=pkgname v=modname
        self.register_types=['register', 'register_dataclass'] #The decorators for which the registration will be invoked
        self.allowed_register_kwargs={'priority','metadata','requires','params_names','constraints','default'}
        self.registry = defaultdict(lambda: RegisteredCategory(registry=self, name=None))
        self.allow_overwrite=True #
        self.default_constraints={} # Constraint names active for every create() call (see set_default_constraints)
        # The internal cache of imported modules and packages
        self.imported_modules={}
        self.failed_modules=[]
        self.imported_packages={}
        self.failed_packages=[]
        self.importable_packages={} # k=requirement name (ex. 'flash_attn'), v=bool: is it importable here?
        # Global outcome of the _test() of the implementations: k=pkgname, v=None if the test passed,
        # error message if it failed. A test is never run twice in the same process, so a whole model
        # pays the cost of (ex.) the flash-attention smoke test only once.
        self.test_results={}
        self.announced=set() # (category, variant, implementation, source, constraints) already reported, to avoid printing the same line for every layer
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
            previous_file=self.modname_filepath.get(previous_module, previous_module)
            current_file=self.modname_filepath.get(modname, modname)
            if self.allow_overwrite is False:
                raise ValueError(f"Package {pkgname} is already present in module {previous_module}. Please use another name. ")
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
            print(f"Import of {modname} in {self.modname_filepath.get(modname, '<unknown file>')} failed:")
            traceback.print_exc()
            self.failed_modules.append(modname)
            return None

    def import_package(self, pkgname: str):
        """
        A specific package is imported. A cache system ensures that if a package' import fails, no more attempt will be done (self.failed_packages),
        conversely, packages are imported only once and then cached (self.imported_packages). It works in tandem with import_module to import the module 
        containing the packages. On import_module's failure, also import package will fail. 
        Accept pkgname (str) as argument, returns the pkgname with the specified name, or None if it cannot be resolved.
        """
        if pkgname in self.failed_packages:
            return None
        if pkgname in self.imported_packages:
            return self.imported_packages[pkgname]
        modname=self.pkgname_modname.get(pkgname)
        if modname is None:
            print(f"WARNING: '{pkgname}' is not a package known to the registry, so it cannot be imported.")
            self.failed_packages.append(pkgname)
            return None
        _module=self.import_module(modname)
        if not _module:
            self.failed_packages.append(pkgname)
            return None
        try:
            package = getattr(_module, pkgname)
        except AttributeError:
            print(f"WARNING: module {self.modname_filepath.get(modname, modname)} has no '{pkgname}' attribute.")
            self.failed_packages.append(pkgname)
            return None
        self.imported_packages[pkgname] = package
        return package

    def package_importable(self, requirement: str) -> bool:
        """True if `requirement` (an external library declared in `requires`) can be imported here.
        The outcome is cached, so a missing library costs one import attempt per process."""
        if requirement not in self.importable_packages:
            try:
                importlib.import_module(requirement)
                self.importable_packages[requirement] = True
            except Exception:
                self.importable_packages[requirement] = False
        return self.importable_packages[requirement]

    def walk_py_files(self):
        """Get dotted_modname and filepath for every .py file under root_path, without importing anything """
        for py in self.root_path.rglob("*.py"):
            parts = py.relative_to(self.root_path).with_suffix("").parts
            modname = ".".join((self.root_modname, *parts)) if parts else self.root_modname
            self.modname_filepath[modname]=py

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
        location = f"{self.modname_filepath.get(modname, modname)}:{node.lineno}"
        args=[a.value for a in decorator_call.args if isinstance(a,ast.Constant)]
        _kwargs={}
        for kw in decorator_call.keywords:
            if kw.arg is None:
                print(f"WARNING: unpacked argument in the registry call of {pkgname} ({location}): **kwargs in a registry decorator cannot be parsed, it is ignored.")
                continue
            try:
                _kwargs[kw.arg]=ast.literal_eval(kw.value)
            except Exception as e:
                print(f"WARNING: cannot evaluate the '{kw.arg}' argument of the registry call of {pkgname} ({location}); it is ignored: {e}. "
                      f"Registry decorator arguments must be literals (strings, numbers, lists, dicts).")
        unknown = set(_kwargs) - self.allowed_register_kwargs
        if unknown:
            print(f"WARNING: unknown argument(s) {sorted(unknown)} in the registry call of {pkgname} ({location}). "
                  f"Supported arguments: {sorted(self.allowed_register_kwargs)}")
        _type=decorator_call.func.attr
        if _type=='register':
            if len(args)!=3:
                raise ValueError(f"The registry call of {pkgname} ({location}) must be @registry.register(category, variant, implementation). Parsed arguments: {args}")
            category=args[0]
            variant=args[1]
            implementation=args[2]
            self.registry[category].name = self.registry[category].name or category 
            self.registry[category].register_implementation(variant=variant, implementation=implementation, pkgname=pkgname, _kwargs=_kwargs)
        elif _type == 'register_dataclass':
            if len(args)!=2:
                raise ValueError(f"The registry call of {pkgname} ({location}) must be @registry.register_dataclass(category, variant). Parsed arguments: {args}")
            category=args[0]
            variant=args[1]
            is_multi = self._class_inherits_from(pkgname, 'MatformerMultiConfig')
            self.registry[category].name = self.registry[category].name or category
            self.registry[category].register_definition(variant=variant, pkgname=pkgname, default=_kwargs.get('default', False), is_multi=is_multi)
        else:
            print(f"WARNING: wrong {_type} registry call in {location}")  

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
        return _keep_class

    def register_dataclass(self, *args, **kwargs):
        return _keep_class


    # Main API
    def implementation_unavailable_reason(self, impl: RegisteredImplementation) -> Optional[str]:
        """Reason (str) why an implementation cannot run here, or None if it looks usable.
        Checks, in order: the `requires` libraries, the outcome of a previously run _test() and the
        importability of the class. The _test() itself is NOT run here: availability tests happen at
        instantiation time (see run_implementation_test), so the registry stays completely lazy."""
        for pkg in impl.requires:
            if not self.package_importable(pkg):
                return f"the required library '{pkg}' is not installed"
        previous_test = self.test_results.get(impl.pkgname)
        if previous_test is not None:
            return f"availability test failed: {previous_test}"
        if impl.get_implementation() is None:
            return "the implementation class could not be imported"
        return None

    def constraints_unavailable_reason(self, impl: RegisteredImplementation, active: dict) -> Optional[str]:
        """Reason (str) why an implementation doesn't satisfy the active constraints, else None.
        An implementation declaring no constraints at all is universal and is never filtered out.
        Otherwise two styles can be used, also together:
          * constraints=['is_inference'] is a list of what the kernel supports (an hypothetical
            FlashInfer usable only for inference): it is skipped as soon as an active constraint is
            not among them;
          * constraints={'is_training': False} declares an explicit incompatibility: everything not
            mentioned stays compatible.
        """
        declared = impl.constraints
        if not declared:
            return None
        declares_capabilities = any(declared.values())
        for name in active:
            if not active[name]: # A constraint mapped to False is simply not active
                continue
            if declared.get(name) is False:
                return f"is incompatible with the required constraint '{name}'"
            if declares_capabilities and name not in declared:
                return f"does not support the required constraint '{name}'"
        return None

    def set_default_constraints(self, constraints: Any):
        """Globally activate constraint names, ex. registry.set_default_constraints(['is_training'])
        in a training entry point. Every later create() call that doesn't pass its own constraints
        will skip the implementations not supporting them."""
        self.default_constraints = normalize_constraints(constraints)
        return self.default_constraints

    def announce(self, category: str, variant: str, implementation_name: str, source: str, active_constraints: dict):
        """Report the chosen implementation once per (variant, implementation, constraints)"""
        key = (category, variant, implementation_name, source, tuple(sorted(active_constraints.items())))
        if key in self.announced:
            return
        self.announced.add(key)
        constraint_note = f" [constraints: {sorted(active_constraints)}]" if active_constraints else ""
        print(f"Implementation {implementation_name} was chosen for {category}.{variant} ({source}){constraint_note}")

    def active_constraints(self, constraints: Any) -> dict:
        """Constraints of a single create() call, falling back to the global default ones."""
        if constraints is None:
            return dict(self.default_constraints)
        return normalize_constraints(constraints)

    def run_implementation_test(self, impl: RegisteredImplementation):
        """Run the class-level _test() of an implementation, once per process.

        Called at instantiation time (never during the registry's lazy discovery): the outcome is
        stored in self.test_results. _test() is a @staticmethod returning True; any raised exception or a
        False return means the implementation cannot be used here. Raises ImplementationNotAvailable
        on failure, so the caller goes on with the next implementation of the variant.
        """
        if impl.pkgname in self.test_results:
            failure = self.test_results[impl.pkgname]
            if failure is not None:
                raise ImplementationNotAvailable(f"availability test previously failed: {failure}")
            return
        cls = impl.get_implementation()
        test = getattr(cls, '_test', None) if cls is not None else None
        if test is None:
            self.test_results[impl.pkgname] = None # No test defined: nothing to check, remember not to look again
            return
        failure = None
        try:
            if self.call_implementation_test(test, cls) is False:
                failure = "_test() returned False"
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        self.test_results[impl.pkgname] = failure
        if failure is not None:
            print(f"NOTE: {impl.describe()} failed its availability test and will be skipped: {failure}")
            raise ImplementationNotAvailable(f"availability test failed: {failure}")

    @staticmethod
    def call_implementation_test(test, cls):
        """Invoke an implementation's _test(). A @staticmethod is expected, but a plain
        `def _test(cls)` is tolerated"""
        positional = [p for p in inspect.signature(test).parameters.values()
                      if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        if positional and positional[0].default is inspect.Parameter.empty:
            return test(cls)
        return test()

    def get_definition(self, category: str, variant: Optional[str] = None):
        if category not in self.registry.keys():
            raise ValueError(f"{category} is not a category registered in the Register. Available categories: {self.registry.keys()}")
        return self.registry[category].get_definition(variant)
    def get_definition_defaults(self,category:str, variant: Optional[str]=None):
        return self.get_definition(category,variant).from_dict({}).asdict()
    def create(self, category: str, variant: str, preferred: Optional[str] = None,
               constraints: Optional[Any] = None, **kwargs):
        """Instantiate a module through the registry, e.g. registry.create('attention', 'normal', ...).
        `constraints` (ex. ['is_training']) skips the implementations that don't support it; when it
        is omitted the global default constraints are used (see set_default_constraints)."""
        if category not in self.registry.keys():
            raise ValueError(f"{category} is not a category registered in the Register. Available categories: {self.registry.keys()}")        
        return self.registry[category].get_instance(variant, preferred=preferred, constraints=constraints, **kwargs)

    def list_categories(self, defined_only: bool = True, multi_only: bool = False) -> list:
        """Category names, as used to validate the sections of a config file. By default only the
        categories that have a dataclass definition are returned; `multi_only` keeps the categories
        that accept more than one section (a listed group in the YAML)."""
        defined = [name for name, category in self.registry.items() if not defined_only or category.has_definition]
        if not multi_only:
            return defined
        return [name for name in defined if self.registry[name].is_multi]
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

    def test_status(self, impl: RegisteredImplementation) -> str:
        """Outcome of the availability test of an implementation, for introspection: 'passed',
        'failed: <reason>' or 'not tested' (no _test() yet, or nothing instantiated so far).
        Purely introspective: it never imports anything, keeping the registry lazy."""
        failure = self.test_results.get(impl.pkgname, 'not tested')
        if failure == 'not tested':
            return failure
        return f"failed: {failure}" if failure else "passed"

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
                            "requires": impl_obj.requires,
                            "constraints": impl_obj.constraints,
                            "test": self.test_status(impl_obj),
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
        info = self.get_registry_info(category=category, variant=variant, implementation=implementation,
                                      relative_filepath=relative_filepath)
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
                        if impl_info.get("requires"):
                            print(f"      Requires: {impl_info['requires']}")
                        if impl_info.get("constraints"):
                            print(f"      Constraints: {impl_info['constraints']}")
                        if impl_info.get("test") and impl_info["test"] != "not tested":
                            print(f"      Test: {impl_info['test']}")
                        if impl_info.get("metadata"):
                            print(f"      Metadata: {impl_info['metadata']}")    
