This folder contains the low-level kernels used by Matformer.

Matformer modules, indeed, are not bound to specific implementation such as torch.nn.linear.
In Matformer's modules, for example in [[transformer_blocks]], you will never see a direct instantation such as:
    self.embedding = nn.Linear(...)
instead, models are instantiated using the [[MatformerRegistry]] which, in a correctly instantiated model, should live inside the [[MatformerCache]].
The previous line, so, should be rewritten like:
    self.embedding = self.cache.registry.create("linear")

The registry will look inside this folder to find the best available implementation, such as Nvidia's transformer-engine, Liger kernels, vanilla PyTorch and so on.
