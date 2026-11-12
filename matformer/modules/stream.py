from contextlib import contextmanager
from matformer.matformer_registry import registry

class Stream:
    def __init__(self):
        self.sockets, self.x = None, None
        self.visits = {}                  
        self.context = {}                 

    def bind(self, sockets): self.sockets = sockets
    def init(self, x):       self.x, self.visits = x, {}
    def exit(self):          return self.x

    def apply(self, name, **kw):
        self.visits[name] = self.visits.get(name, 0) + 1
        self.x = self.sockets[name](self.x, **kw)

    def merge(self, branch):
        raise NotImplementedError

    @contextmanager
    def branch(self):
        branch = type(self)()
        branch.bind(self.sockets)
        branch.init(self.x)
        yield branch
        self.merge(branch)


@registry.register("stream", "residual_stream", "default")
class ResidualStream(Stream):
    def merge(self, branch):
        self.x = self.x + branch.x



