from matformer.matformer_registry import registry

class NormStyle:
    active = ()                                  
    names = {}                                   

    def __call__(self, slot): return slot in self.active
    def attr(self, slot):     return self.names.get(slot, f"{slot}_norm")


@registry.register("norm_style", "pre", "default")
class Pre(NormStyle):
    active = ["pre_attn", "pre_mlp"]
    names = {"pre_attn": "attn_norm", "pre_mlp": "mlp_norm"}

@registry.register("norm_style", "post", "default")
class Post(NormStyle):
    active = ["post_add_attn", "post_add_mlp"]
    names = {"post_add_attn": "attn_norm", "post_add_mlp": "mlp_norm"}

@registry.register("norm_style", "inner", "default")
class Inner(NormStyle):
    active = ["post_attn", "post_mlp"]
    names = {"post_attn": "attn_norm", "post_mlp": "mlp_norm"}

@registry.register("norm_style", "pre_no_attn", "default")
class PreNoAttn(NormStyle):
    """Pre-norm without the attention-side norm: the pre_attn socket stays empty
    (identity). E.g. HF ModernBERT, whose layer 0 applies the attention branch raw."""
    active = ["pre_mlp"]
    names = {"pre_mlp": "mlp_norm"}

@registry.register("norm_style", "sandwich", "default")
class Sandwich(NormStyle):
    active = ["pre_attn", "post_attn", "pre_mlp", "post_mlp"]
    names = {"pre_attn": "attn_norm", "post_attn": "attn_post_norm",
             "pre_mlp": "mlp_norm",   "post_mlp": "mlp_post_norm"}
