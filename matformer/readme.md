
!Section MatformerModule/class-annotations

If you look at the predefined neural-net implementations in Matformer, for example [[TransformerBlock]], you will immediately notice that at the top of the class some non-standard annotations are present.
For example:

```
    attn_norm: "param_name:attn_norm"
    mlp_norm: "param_name:mlp_norm"
    self_attn: "param_name:attn"
    mlp: "param_name:mlp"

    available_hooks = [
        'pre_attn', 'post_norm_pre_attn', 'post_attn',
        'pre_mlp',  'post_norm_pre_mlp',  'post_mlp',
        'post_attn_norm', 'post_mlp_norm',
        'residual_stream_output'
    ]

Those annotations are automatically interpreted by the base [[MatformerModule]] class. For this reason, it is useful for consistency that if you write new modules for Matfomer, either permanent or temporary, such as [[Hooks]], you inherit MatformerModule. 
MatformerModule inherits from two important base classes:
* [[ParameterRenamer]]
ParameterRenamer determines the actual names of the weights as they will be included in the checkpoint. PyTorch, by default, saves parameter inside the checkpoint by expanding their entire path inside the model. This would lead to very aestethically unpleasant and potentially confusing weights' keys in a complex architecture like matformer such as, for example, "model.model.encoder.layer.4.mlp.inner.weight". To avoid this, Matformer leaves at disposal an handy system for renaming parameters' name and make them "stable", that is, potentially independent from the actual modules' names used inside the library.
It seems hard, but is actually very easy to use:
1) Any 
2) "transparent" means
* [[HooksMixin]]

