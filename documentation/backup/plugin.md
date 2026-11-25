Plugins and sockets: user guide

I haven’t run this. It describes the design as we built it tonight.

1. The model

Every structural piece of the model sits in a socket: a named slot holding an ordered list of components. A call runs the components in order, each receiving the output of the previous one. An empty socket is the identity. If a component returns None, the input passes through unchanged, which is how passive plugins (probes) work.

Attaching to a socket therefore means “run this before or after what’s already there”, and replacing means “run this instead”.

2. Which sockets exist

A socket’s address is its module path plus its name. Exact paths depend on the model class, so check with wrapper.inspect("*") once. For the classification model they look like this:

Address (suffix)	What it holds	Notes
lm_head	LM head	
encoder	embedding + transformer	
embed_tokens	token embedding	add_first = old pre_embed, add_end = old post_embed
embed_positions	learnable positions, or empty	
blocks	the whole NakedTransformer	
layer_<i>	the whole block i	one visit per pass in looped models
final_norm	final norm	add_first = old before_final_norm, add_end = old before_unembed
layers.<i>.pre_attn	norm before attention	empty unless the norm style uses it
layers.<i>.self_attn	the attention (sequence mixer)	
layers.<i>.post_attn	norm after attention, inside the branch	
layers.<i>.post_add_attn	norm after the residual add	the trunk, i.e. the residual stream
layers.<i>.pre_mlp / mlp / post_mlp / post_add_mlp	same, for the MLP	

Inside a block, the flow is:

branch: pre_attn → self_attn → post_attn   (added to the trunk)
trunk:  post_add_attn
branch: pre_mlp → mlp → post_mlp           (added to the trunk)
trunk:  post_add_mlp

The pre_*/post_* sockets always exist, even when no norm is in them. A probe on post_add_mlp reads the residual stream at the end of a layer, whatever the norm style.

3. Targeting sockets

A target is a glob pattern on the full address, and * crosses dots.

python
"*.final_norm"                       # the final norm
"*.layers.*.post_add_mlp"            # end of every layer
"*.layers.3.*"                       # everything in layer 3
"*.layers.[0-2].pre_attn"            # layers 0 to 2 (single digits only)
["*.layers.0.post_attn", "*.layers.5.post_attn"]   # a list of patterns
"*"                                  # everything

If a pattern matches nothing, you get a ValueError. Nothing is silently skipped. wrapper.inspect("*.layers.3.*") shows what a pattern hits and what each socket currently holds.

In config files, target is treated the same way. Only if you kept my _target_patterns helper, all and bare names such as pre_attn also work.

4. Writing a plugin
python
@registry.register_dataclass("plugin", "myprobe", default=True)
@dataclass
class MyProbeConfig(PluginConfig):
    scale: float = 1.0                    # config fields; `alias` comes from PluginConfig

@registry.register("plugin", "myprobe", "default")
class MyProbe(Plugin):
    name = "myprobe"                      # class-level: used by "plugin_type:myprobe"

    def __init__(self, config, model_config, cache):
        super().__init__(config, model_config, cache)
        ...                               # build modules here: they enter the model's state dict

    def forward(self, x, **kwargs):
        ...
        return None                       # passive: x continues unchanged
        # return new_x                    # active: replaces x

What a plugin knows about its position once mounted:

self.address is the full socket address.
self.visit is the 0-based pass index within the current forward (it is >0 only in looped models).
self.layer_idx is the layer index, or None outside layers.
self.cache and self.model_config are available.

Rules:

Names: a mounted instance is named after, in order, the name= argument, the config’s alias, the class name, and the lowercased class name. If the name is already taken in that socket, it becomes name_1, name_2, and so on. Names identify a plugin for detach, swap and replace, and for freezing.
State dict: plugin modules are always part of the model, under keys like ...plug_<socket>_<name>.*. A base checkpoint loaded into a model with plugins reports them as missing keys, which is expected. Use strict=False or filter plug_*.
Plain callables work too. For a quick hook, pass lambda x, **kw: ... or a function with no config. It is not an nn.Module, so it adds no parameters.
5. Config-time

Two routes, both applied by the wrapper at construction.

a) The experiment config, with the mount_plugin sections: target is a pattern and plugin is a dict with a mandatory type plus the plugin config fields. Use an alias to control the name.

target = "*.layers.*.post_add_mlp"
plugin = { type = "myprobe", scale = 0.5, alias = "mine" }

The syntax above is illustrative. Use whatever format your config files already use. The default target in MountPluginConfig is still "final_output", which matches no socket, so set it explicitly.

b) The model class, via default_plugins(), which returns (pattern, plugin_config) pairs. TransformerWithClassificationHead uses it to put the classifier on *.final_norm, and MultiProbeClassifier uses it to put probes on every probe_targets pattern. Results of probes come back as {address: logits}, with address@n on later passes.

Because plugins are in the state dict and the module tree, anything mounted at config time is trainable, freezable and saved with the model, so it can be a permanent part of the architecture.

6. Runtime

All on the wrapper, and all take a pattern first.

python
w.add_end("*.layers.*.post_add_mlp", cfg)            # after what's there
w.add_first("*.layers.3.self_attn", my_hook)         # before attention
w.attach("*.layers.3.pre_attn", cfg, at=1, name="x") # explicit index and name
w.replace("*.layers.0.pre_attn", affine_cfg)         # swap the norm for something else (ref=0 by default)
w.replace("*.layers.0.pre_attn", other, ref="x")     # by name
w.swap("*.layers.3.pre_attn", 0, 1)                  # indices or names
w.reorder("*.layers.3.pre_attn", [1, 0])
w.detach("*.layers.*.post_add_mlp", "myprobe")       # by name (or index)
w.inspect("*.layers.3.*")                            # {address: "[0] norm -> [1] myprobe"}
w.plugins()                                          # [(address, plugin), ...], plugins only

Notes:

Config vs instance: pass a config when mounting on several sockets, and each socket gets its own instance. Passing one instance to several sockets shares it, and it will only remember the last socket as its .socket.
Ablation: structural components are removable too. For example, w.detach("*.layers.3.self_attn", 0) removes attention in layer 3. Its weights stay in the module tree, so checkpoints still load.
Order: inside a socket, index 0 runs first. add_first and add_end are the “before” and “after” hooks, for example before and after attention on its own socket.
7. Freezing and reporting

In the training config’s freeze and unfreeze:

plugin_type:classification matches plugins by class name or by instance name.
A socket address matches the plugins mounted there.
component:mlp, layer:3 and the older forms are unchanged.

trainable_param_report() prints the plugin and backbone splits.

8. Things to watch
Norm style: normalization_position in LayerConfig is still typed Literal["pre", "post"]. Extend it if you want inner and sandwich. The names (checkpoint keys) of every style are in the NormStyle classes.
Layer schedule: in ModelConfig, layer_schedule= None has no type annotation, so the dataclass does not treat it as a field and it cannot be set from a config file. Write layer_schedule: Optional[List[int]] = None. Also, NakedTransformer.forward accepts start and stop.
Visits: wrapper.forward must call reset_visits() at its start, otherwise visit keeps counting across batches.
Virtual models: a virtual model must use SocketsMixin for addresses to resolve. find raises if the model has no sockets at all.
