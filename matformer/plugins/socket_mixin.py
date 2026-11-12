import torch.nn as nn
from contextlib import contextmanager
from matformer.matformer_registry import registry          
class Socket:
    def __init__(self, name, owner):
        self.name = name
        self.owner = owner        
        self.address = name                        
        self.components, self.visits = [], 0

    def index(self, ref):                          
        if isinstance(ref, int):
            return ref
        return next(i for i, c in enumerate(self.components) if getattr(c, "name", None) == ref)

    def add(self, component, index=None):
        self.components.insert(len(self.components) if index is None else index, component)

    def add_first(self, component): self.add(component, 0)
    def add_end(self, component):   self.add(component)

    def get(self, ref=0):                return self.components[self.index(ref)]
    def remove(self, ref=0):             return self.components.pop(self.index(ref))
    def replace(self, component, ref=0): self.components[self.index(ref)] = component

    def swap(self, a, b):
        i, j = self.index(a), self.index(b)
        self.components[i], self.components[j] = self.components[j], self.components[i]

    def reorder(self, order):
        self.components = [self.components[i] for i in order]

    def inspect(self):
        if not self.components:
            return "(identity)"
        return " -> ".join(f"[{i}] {getattr(c, 'name', type(c).__name__)}"
                           for i, c in enumerate(self.components))
    __repr__ = inspect

    def __call__(self, x, **kw):
        for component in self.components:
            out = component(x, **kw)
            if out is not None:                    
                x = out
        self.visits += 1                           
        return x



class SocketMixin:
    def init_sockets(self, stream_type="residual_stream"):
        self.sockets = {}
        self.stream = registry.create("stream", stream_type)
        self.stream.bind(self.sockets)

    def add_socket(self, name):
        self.sockets[name] = Socket(name, owner=self)

    def register_socket(self, name, component, attr=None):
        """Structural component. attr is the attribute name on self, i.e. the checkpoint key."""
        if name not in self.sockets:
            self.add_socket(name)
        if attr:
            setattr(self, attr, component)
        self.sockets[name].add(component)

    def inspect_sockets(self):
        return "\n".join(f"{name:15s} {socket.inspect()}" for name, socket in self.sockets.items())

    def add_nested_socket(self, address):
        """Create a socket at a dotted address ('encoder.layers.0.model_input'),
        building empty intermediate modules as needed. Returns the Socket."""
        *path, name = address.split(".")
        owner = self
        for p in path:
            if p not in owner._modules:
                owner.add_module(p, SocketNode())
            owner = owner._modules[p]
        owner.add_socket(name)
        return owner.sockets[name]
        
class SocketNode(nn.Module, SocketMixin):
    """Empty module that only holds sockets, so named_modules() yields dotted addresses."""
    def __init__(self):
        super().__init__()
        self.sockets = {}