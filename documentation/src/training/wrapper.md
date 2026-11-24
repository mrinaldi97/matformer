---
title: MatformerWrapper
sources:
  - matformer/matformer_wrapper.py
  - matformer/plugins/plugin_manager.py
---
# MatformerWrapper

The ModelWrapper contains methods to load checkpoints, weights as well as everything required to train the model.
The training procedure itself, however, is not hard-coded into the Wrapper but is handled through another class,
called LearningSignal. The LearningSignal is the one containing the actual preparation of the inputs
(ex. masking for Bert-like models) and the selection of targets for the loss function.
For more complex scenario, it is possible to use a "LearningSignalSelector", another class that select the correct LearningSignal according
to specific criteria step by step (ex. to randomly mix more LearningSignal(s) or to correctly route the data according
to info present in the batch's metadata.)
The LearningSignal, LearningSignalSelector and the model class are all accessed through the [[components/registry | MatformerRegistry]].

MatformerModelWrapper is one of the most important components of the architecture: any MatformerModel must be wrapped inside the ModelWrapper to enable frameworks' functionalities. MatformerModelWrapper inherits from [[components/matformer-module | MatformerModule]] and [[components/plugins | PluginManager]], which means that [[components/plugins | Plugins]], a fundamental feature of Matformer, are managed by the Wrapper.
Any model executed or trained by matformer is initialized inside the wrapper, and its reference is stored as `self.model`.
The Wrapper is also responsible for the handling of [[components/virtual-models | VirtualModels]], that is, models that are not executed live but whose activations are rather retrieved from an activation store, which is a useful feature for interpretability, distillation or model's debugging. To understand VirtualModels, the activation store and how the wrapper is able to switch from "live" to "virtual" `runtime_mode`, please see the [[components/virtual-models | VirtualModels]] section.
In the `__init__`, after having initialized plugins and the eventual virtualmodel, the model is created and the [[components/tokenizers | tokenizer]] attached. If the wrapper is initialized in training mode (the [[components/configuration | MatformerConfigs]] contains a "training" section), the [[training/learning-signals | LearningSignal]] is initialized, and models' weight are [[file:matformer/utils/initialization.py | initialized]] and, if necessary, some of them may be frozen according to the "freeze" config (see: [[training/wrapper#freezing | Freezing]]).
Because any forward pass to a Matformer model will pass from the MatformerModelWrapper makes sure that any tensor passed to the forward will be wrapped in a [[components/tensors | TensorDC]] before getting inside the model. Thanks to this, matformer's neural network modules don't have to worry about the tensor being a TensorDC or not: plain `torch.Tensor` will never pass the barrier of the MatformerModelWrapper's forward.
The MatformerModelWrapper is responsible for the inner part of the training step, the one called by the [[training/trainer | MatformerTrainer]].

## Training step

1. Check if the batch's [[components/tensors | TensorDC]] has the special attribute "worker_has_finished" set to True. In this case, loss is immediately set to 0.0; this is necessary when ranks get out of sync during distributed training, for example when a GPUs exhaust the training data while other GPUs are still busy.
2. LearningSignalSelector is asked for the list of active learning signals, and they are used to compute the loss. Check [[training/learning-signals | the learning signals]] for detail!
3. Loss is logged
4. The wrapper now will scan the dictionary "aux_losses" contained in the [[components/cache | CachedStuff]]. If any additional loss is present there (ex. a Moe router loss), it is added to the loss and logged separetely.
5. Learning rate is logged
6. Total loss is returned.

The whole training step is guarded by a try except block. If an Exception is found, particularly if this step got OOM, the current batch is skipped and gradients cleared. It is a dirty workaround, but it may be helpful to avoid a catasthrophic training run loss if just few batch errors out. Clearly, it must be thought as a workaround that breaks training correctness: it is duty of the user to decide how many failed training steps can be tolerated.

`strict_steps` disables the workaround. The trainer's memory probe enables it while measuring, so a failure reaches the probe.

## Model loading

The wrapper is also the entry point for loading models, either for inference or for finetuning / inheriting of weights.
The loading system is very flexible and will be explained in two steps, here and in the [[loading/checkpoint-holder | CheckpointHolder]] documentation page.
the [[file:matformer/matformer_wrapper.py:load_model | load_model function]] builds a model from scratch and returns it to the caller.

This function can create a model in different ways:

- If you provide just a [[components/configuration | MatformerConfigs]] object, it creates a fresh model ready for training
- If you provide a path OR a [[loading/checkpoint-holder | CheckpointHolder]], the config and models are loaded from the checkpoint (with the term 'checkpoints' we intend also published weights, such as HuggingFace repositories);
- If you provide BOTH (path/checkpointholder) + configs, what happens is that the config is at first initialized from the path and then overriden with the fields provided with the configs argument. This is particularly useful to train derived models, for example a classification fine-tuning starting from a pre-trained language model. The sections that are inherited from the base checkpoint can be chosen with the `inherit_sections` argument.

After the weights are loaded or initialized, the wrapper checks for the presence of the "weights" section in the [[components/configuration | MatformerConfigs]]; if this section is set, it allows to load full or partial weights from arbitrary many other models. This is helpful for experimental settings, as it allows to perform model stiching.

`load_model` returns a dictionary:

```python
result = load_model(path="repo/or/file", configs=configs, map_location="cpu")
model = result["model"]
configs = result["configs"]
missing, unexpected = result["missing"], result["unexpected"]
```

The trainer uses the returned configs because a checkpoint can add sections. The missing and unexpected keys are part of the load report. `additional_weights_results` contains one entry per extra weight source.

<a id="freezing"></a>
## Freezing

`freeze` and `unfreeze` in the training config resolve a target to a list of parameters, then set `requires_grad`. A target is either a string or a dict:

* `{"layer": "<pattern>", "component": "<name>"}`: that component, on layers matching the pattern only. `component` is optional.
* `"all"`: every parameter.
* `"plugin_type:<name>"`: every plugin of that kind.
* `"layer:<pattern>"`: every layer matching the pattern (`every_N`, `odd`, `even`, `S_E`, ...).
* `"component:<name>"`: that named component on every layer.
* a socket address: every plugin mounted there.
* a bare name: tried as a plugin type, then a component, then a stable parameter/module name prefix.

Freezing is applied after the weights are initialised and before the optimiser is built. `trainable_param_report()` prints the split; use it to check that a pattern matched the intended parameters.

## Runtime

`forward` converts a plain `torch.Tensor` to a `NormalTensor`, adds a batch dimension to a 1-D input, and resets the per-socket visit counters. The model is stored in `self._models` for each runtime mode. Switching between live and virtual reuses the mounted plugins.
