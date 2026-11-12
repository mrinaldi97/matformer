from dataclasses import dataclass, field
from typing import Optional, Union
from matformer.matformer_module import MatformerModule
from matformer.dataclasses.matformer_config import MatformerConfig, MatformerMultiConfig
from matformer.matformer_registry import registry
import torch
class Plugin(MatformerModule):
    def __init__(self, config, model_config, cache):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.cache = cache
        self.socket = None # Assigned by PluginManager at mount time
        self.id = None # Same as above                                  
        self.kind = None # Same as above
    @property
    def address(self):   return self.socket.address
    @property
    def visit(self):     return self.socket.visits
    @property
    def layer_idx(self): return getattr(self.socket.owner, "layer_idx", None)
    @property
    def index(self):
        #Sporca per ora
        for i, plug in enumerate(self.socket.components):
            if plug==self:
                return i
    def save(self, path):
        import torch
        torch.save(
            {
                "plugin_kind":self.kind,
                "address":self.socket.address,
                "id":self.id,
                "state_dict":self.stable_state_dict(),
                "index":0, #Come prenderlo?
                "plugin_config":self.config.asdict()
            }, path)
    def load(self, path=None, sd=None):
        if sd is not None:
            assert path is None,"Either restore from state dict or from path!"
            self.load_stable_state_dict(sd)
        elif path is not None:
            ckpt=torch.load(path,weights_only=False)
            self.load(sd=ckpt['state_dict'])

@dataclass
class PluginConfig(MatformerMultiConfig):
    targets: Optional[Union[list,str]] = None
    name: str = None
