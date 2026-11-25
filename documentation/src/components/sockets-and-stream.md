---
title: Sockets, plugins and the stream
sources:
  - matformer/plugins/socket_mixin.py
  - matformer/modules/stream.py
---
# Sockets, plugins and the stream
In Matformer, any neural network can be defined with having one or more "Socket" where components ("Plugins") are "mounted" and integrated in the computational graph. This is particularly handy for several reasons: it allows for being very DRY when extending the implementation to support very different model families, it permit to insert extra components into the computational graph natively, as if they were hard-coded into the architecture, and allow for inspection and intervention at any stage of the computation, which is a very useful feature for Interpretability experimentation. Finally, the Socket/Plugin system allows for the easy replacement of any components, for development or experimental purposes.
While this system may seem confusign at a first glance, it is actually very simple.
Each [[components/matformer-module | MatformerModule]], for example the [[file:matformer/transformer_blocks/transformer.py:NakedTransformer | default modules of transformer models]] inherit from [[file:matformer/plugins/socket_mixin.py:SocketMixin | SocketMixin]] and, at initialization time, should call "init_sockets" with a specific stream type (default to residual stream, but potentially could be AttnRes, mHC or whatever else). Any useful position in the network's computational graph should be added as a socket, for example "attention" or "normalization" or "mlp". Adding sockets is inexpensive, because if unpopulated they default to identity. For this reason, it is suggested to add sockets to any position that can be useful not only for default components to sit, but also where inspection or additional plugins could be mounted, wheter in training or during inference. [[file:matformer/plugins/socket_mixin.py:Socket | Socket]], indeed, contain inside an ordered list of components, which can be empty, that can be reordered, inspected and where components can be easily removed, inserted or replaced.

## The Socket object

A `Socket` has the following methods:

| Method | Effect |
|--------|--------|
| `add(component, index=None)` | insert a component, at the end by default |
| `add_first(component)` / `add_end(component)` | insert at position 0 / at the end |
| `get(ref=0)` / `remove(ref=0)` / `replace(component, ref=0)` | access, remove or replace by index or by name |
| `swap(a, b)` / `reorder(order)` | change the order |
| `inspect()` | a one-line rendering of the chain |
| `__call__(x, **kw)` | run every component in order |

Each component receives the output of the previous one. A component that returns `None` leaves the value unchanged; an empty socket is the identity. A probe can therefore be mounted without changing the computation, and a component that returns a new value is an active plugin.

## The Stream object

A `Stream` owns the value that is flowing and the sockets it applies to. The API is four methods:

```python
stream.init(x)                 # start a forward: current value is x
stream.apply("pre_attn")       # run the socket on the current value
with stream.branch() as s:     # open a child stream
    s.apply("self_attn")       # work on a copy
stream.apply("post_add_attn")  # back on the trunk
stream.exit()                  # return the current value
```

`branch()` is a context manager. It creates a child of the same stream class, binds it to the same sockets, initialises it with the current value, yields it, and on exit calls `merge(child)`. The default `ResidualStream.merge` adds the child's value to the parent's:

```python
@registry.register("stream", "residual_stream", "default")
class ResidualStream(Stream):
    def merge(self, branch):
        self.x = self.x + branch.x
```

The residual connection is the merge. A different stream type can subtract, gate, concatenate or route; the block code does not change.

`stream.visits` counts how many times each socket was applied in the current forward. Plugins read it as `self.visit`, which is how a probe in a looped model distinguishes the first pass from the third.

## Example block

```python
class MyBlock(MatformerModule):
    def __init__(self, config):
        super().__init__()
        self.init_sockets(stream_type="residual_stream")
        self.register_socket("pre_attn", registry.create("norm", "rmsnorm", normalized_shape=config.hidden_size))
        self.register_socket("self_attn", registry.create("sequence_mixer", "attention", hidden_size=config.hidden_size))
        self.add_socket("post_attn")
        self.add_socket("post_add_attn")

    def forward(self, x, **kwargs):
        self.stream.init(x)
        with self.stream.branch() as s:
            s.apply("pre_attn")
            s.apply("self_attn", **kwargs)
            s.apply("post_attn")
        self.stream.apply("post_add_attn")
        return self.stream.exit()
```

`register_socket(name, component, attr=None)` also sets the component as an attribute when `attr` is given, which is what makes it appear in the state dict and under the stable name declared by the class annotation. `add_socket(name)` creates an empty socket that is the identity until something is mounted in it.

## The plugin layer

Sockets hold components; [[components/plugins | plugins]] are the components mounted from the configuration or at runtime. The `PluginManager` indexes sockets by their dotted address, matches patterns with `fnmatch`, and provides `add_first`, `add_end`, `attach`, `replace`, `swap`, `reorder`, `detach`, `inspect` and `plugins`.

A plugin receives and returns the same kind of object the socket carries: a [[components/tensors | TensorDC]] in the tensor sockets, and the type documented by the socket's owner elsewhere.
