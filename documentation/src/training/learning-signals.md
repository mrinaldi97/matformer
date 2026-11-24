---
title: Learning signals
sources:
  - matformer/training/learning_signals/learning_signals.py
  - matformer/training/learning_signals_selectors/learning_signals_selectors.py
category_docs: [learning_signal, learning_signal_selector]
---
# Learning signal & Learning signal selector
## Learning signals
Because Matformer should support even experimental settings, the [[training/wrapper | MatformerWrapper]] doesn't hard-wire any specific learning signal. Model being trained for autoregressive objective, classification, masked language model, flow matching has no direct impact on how the training step is written in the wrapper: this stays generic and the actual signal used is provided by the registry. Moreover, being Matformer a framework designed with experimentation in mind, there is the possibility to have more than one learning signal in the training run: the correct learning signal could be chosen stochastically, or data-wise or even more than one learning signal can be mixed together.
Each LearningSignal is based on the abstract class [[file:matformer/training/learning_signals/learning_signals.py:LearningSignal | LearningSignals]] whose methods are executed on a specific order by the wrapper.
1. prepare_inputs receives the batch and return a tuple, (input_sequence, metadata). The wrapper will then make a forward pass to the model passing the returned input_sequence to it. The model is eventually run with the kwargs specified in the learning signal's model_kwargs dict. 
2. select_targets receives predictions - the raw model's output -, the original batch and the metadata object. It returns (predictions, targets), already flattened and ready for the loss function.
3. The wrapper then use the loss function defined in learning signal's loss_function and sum this loss to the total loss of the batch, ready for the optimizer
4. As a final, optional, step a callback on the learning signal is called, post_loss, which passes (loss, predictions, targets) for other things, such as additional logging.
An easy good example of a learning signal, that can be used as a basis if you need to write a new one, is [[file:matformer/training/learning_signals/masked.py:MaskedLearningSignal | Masked]], used for bert-like models.
Learning signals are indexed by the registry in the category [[registry_category:learning_signals | learning_signals]]. Click to explore them!
## Learning signals selector
The most common case is that the learning signal selector acts as a no op and just a single learning signal (ex. autoregressive) is use for the training. However, Matformer adds an extra layer of flexibility by allowing to:
- Switch learning signal automatically depending on data or stochastically
- Mix learning signals together
To understand how to write a Learning signal selector, is again useful to refer to the [[training/wrapper | MatformerWrapper]] which handles all the dialogue with the LearningSignal selector in the training step.
0. The selector is initialized in the wrapper's init, and the selector is populated with all the configured learning signals, by directly accessing the [[components/configuration | MatformerConfigs]] object. 
1. ls' pre_step is called with the raw batch, an optional callback for anything not foreseen
2. The wrapper sends (batch, batch_idx) to the ls' "pick" function. The Learning Signal Selector will return to the wrapper the list of LearningSignals active for this step; the wrapper will iterate on them executing the pipeline already explained in the [[training/learning-signals#learning-signals | LearningSignals]] section. If for a specific batch more than one learning signal is returned, their losses are summed.
Matformer provides by default "single" (the no-op for a single ls),"multitask" (to sum several ls together) and a stochastic selector. Other selectors can be explored through the registry: [[registry_category:learning_signal_selector | learning_signal_selector]]. The stochastic selector is a good starting point to write a custom selector. 
The stochastic selector can be used in this way:
```yaml
training:
  learning_signal_selector: stochastic
  learning_signals: [autoregressive, masked]
  selector_config:
    curriculum: true
    weights_start: [0.2, 0.8]      # at step 0: 20% GPT, 80% BERT
    weights_end:   [0.8, 0.2]      # at the end of the curriculum: 20% BERT, 80% GPT
    final_steps: 10000             # the last 10000 steps ignore the curriculum
    final_weights: [0.5, 0.5]      # ... and alternate them evenly, to avoid forgetting either objective
```

The weights are sampling probabilities. At each step `pick` draws one signal with `torch.multinomial`. During the curriculum phase the weights move linearly from `weights_start` to `weights_end` over `total_steps - final_steps` steps. In the final phase `final_weights` is used unchanged. If `curriculum` is false, `weights` is used for the whole run. The final phase keeps both objectives active, so the run does not end on the objective favoured by the last curriculum step.

## Which signals are available

The registry lists them under [[registry_category:learning_signals | learning_signal]]. At the time of writing:

* `autoregressive`: shifts the targets by one and masks padding; the model is called with `is_causal=True`.
* `masked`: the [[components/maskerator | Maskerator]] hides a fraction of the tokens, the cloze mask becomes both the target selector and a `following_attribute`, and the model is called with `is_causal=False`.
* `sequence_classification` and `token_classification`: cross-entropy on the labels. The token-level variant drops `IGNORE_INDEX` positions (subword continuations, special tokens, padding) instead of masking the pad id, so class 0 remains a trainable class.
* `multiprobe_ls`: the model returns `{address: logits}`; every probe contributes its own cross-entropy, the losses are summed, and `post_loss` logs each one separately.

A learning signal can prepare inputs, compute one loss, or combine several losses. The interface is `prepare_inputs`, `model_kwargs`, `select_targets` and `post_loss`, plus the `loss_function` attribute.

## Writing a selector

A selector subclass implements `populate_learning_signals()` and `pick(batch, batch_idx)`. `populate_learning_signals` is called once, at construction, and should create every signal it may later return through `self._create_signal(name)`. `pick` returns a list; the wrapper runs each returned signal in order and sums the losses. `pre_step(batch)` is the optional per-step callback, called before `pick`.

The `single` selector ignores the rest of the list and returns the first signal, which is the no-op case. `multitask` returns every signal, so the losses are summed. `stochastic` returns one, sampled with the weights described above.

## The selector in the config

`training.learning_signals` is a list, but the `single` selector only looks at the first element. A config that pairs a list with `single` is not an error; it just means the other signals are unused. If you want more than one, use `multitask` or `stochastic`, or write a selector that routes on `batch.attributes`.
