from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, ClassVar, Literal, Any
from matformer.dataclasses.matformer_config import MatformerConfig
from matformer.matformer_registry import registry

@registry.register_dataclass("training", "default", default=True)
@dataclass
class TrainingConfig(MatformerConfig):
    ref_data: str = field(default="train", kw_only=True) # Default train the 'train' dataset
    _ref_fields: ClassVar[dict] = {"ref_data": "data"}    
    # Learning objective
    learning_signal_selector: str = "single"
    learning_signals: List[str] = field(default_factory=lambda: ["masked"])
    selector_config: dict = field(default_factory=dict) # This dict is passed to the selector, check its own documentation



    # Optimizer 
    optimizer: str = "adamw"
    lr: float = 5e-4
    final_lr: float = 2e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-10
    no_decay_for_embedding: bool = False
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_nesterov: bool = True
    clip_threshold: float = 50.0
    clip_alpha: float = 0.5

    # LR scheduling
    lr_scheduling: bool = True
    scheduler: str = "cosine"
    warmup_steps_fraction: float = 0.05 # Fraction of total steps spent warming up (not an absolute step count)
    hold_steps_fraction: float = 0.2    # Fraction of total steps held at peak LR before decay
    polynomial_lr_power: int = 2
    annealing_start_step: Optional[int] = None
    annealing_lr: Optional[float] = None
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1

    # Checkpointing & logging
    save_every_n_steps: int = 5000
    save_every_n_epochs: int = 0   # 0 = disabled; saves at the end of every n-th epoch
    save_what: Literal['model', 'plugins', 'both'] = 'both'   # what periodic saves (n steps / n epochs) write
    validate_every_n_steps: int = -1
    max_epochs: int = 1
    checkpoint_name: str = "checkpoint"
    save_dir: str = "./checkpoints"
    logger: Union[List[str], str] = "wandb"
    project_name: str = "default_project"
    run_name: str = "default_run"

    # Runtime
    seed: int = 27
    varlen_strategy: str = "unpadding"
    batch_size: Optional[int] = None
    token_per_batch: Optional[int] = None # Automatically populated: max_seq_len * batch_size * accumulate_grad_batches
    total_steps: Optional[int] = 0 # Will be automatically populated
    total_batches: Optional[int] = 0 # Will be automatically populated
    precision: Literal['16-mixed', 'bf16-mixed', '32','16','bf16','32-true','bf16-true','16-true','64-true','transformer-engine'] = "bf16-mixed"
    num_nodes: Optional[int] = 1
    device_count: Optional[int] = 1
    checkpoint: Optional[str] = None
    start_from_scratch: Optional[bool] = False
    start_from_step: Optional[int] = None
    what_to_reset: Optional[Any] = None
    max_steps: Optional[int] = None
    version: Optional[int] = 1
    compile: Optional[bool] = False
    parallel_strategy: Optional[str] = 'ddp'
    virtual_only: Optional[bool] = False #Use only the virtual model without loading weights
    # Freezing
    """
    Resolves one freeze/unfreeze target to a list of parameters.

    target is either a string or a dict:

      dict: {"layer": "<pattern>", "component": "<name>"}
          -> that component, on layers matching <pattern> only.
             "component" is optional (omit it to match the whole layer).
             Example: {"layer": "8", "component": "mlp"}

      string, matched in this order:
        "all"                    -> every parameter in the model
        "plugin_type:<name>"     -> every plugin registered under that name, any address
        "layer:<pattern>"        -> every layer matching the pattern (every_N, odd/even, S_E, ...)
        "component:<name>"       -> that named component, on every layer
        a socket address, e.g. "encoder.final_output"
                                  -> every plugin mounted there
        a bare name, tried until one matches:
          - plugin_type match       "classification"
          - component match         "mlp"
          - stable parameter/module name (prefix match)   "encoder.norm"

    Examples:
        freeze: ["all"]
        unfreeze: ["classification"]
        # ---
        freeze: ["mlp"]
        # ---
        freeze: ["all"]
        unfreeze: ["layer:even"]
        # ---
        freeze:
          - {layer: "8", component: "mlp"}
          - {layer: "12", component: "self_attn"}
    """
    freeze: Optional[List[str]] = None
    unfreeze: Optional[List[str]] = None
    def __post_init__(self):
        if isinstance(self.logger, str):
            self.logger = [self.logger]
