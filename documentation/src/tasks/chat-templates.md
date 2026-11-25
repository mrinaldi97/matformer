---
title: Chat templates
sources:
  - matformer/utils/chat_template.py
  - matformer/tasks/generation.py
  - matformer/data/lazy_dataset.py
---
# Chat templates

A chat model is trained on a conversation rendered with control tokens that mark who is speaking.
The recipe each model expects is its **chat template**, a Jinja program shipped beside the
tokenizer. Matformer renders that template itself, with Jinja, and never through the HuggingFace
tokenizer's `apply_chat_template`: `matformer.utils.chat_template.ChatTemplate` is the only engine.
The template source is read from the checkpoint (`tokenizer_config.json`, `chat_template.jinja`, or
the `.ckpt` itself), so Qwen, Llama, Mistral, ChatML and Zephyr templates all work unchanged.

## Where it lives

* The checkpoint holder returns the raw template through `get_chat_template()`. A HuggingFace repo
  reads `tokenizer_config.json` (a Jinja string), a `chat_template.jinja` sidecar, or the mapped
  variant; a Matformer checkpoint reads an embedded `chat_template` key or a sidecar file.
* `load_model` builds it once and exposes it as `wrapper.chat_template` (a `ChatTemplate`), also
  attached to `tokenizer.chat_template`. Datasets and tasks reach it through the tokenizer.
* A repo published with `huggingface_publish` writes `chat_template.jinja`, and a hub repo whose
  `config.json` says `model_type: matformer` is loaded with the Matformer holder directly.

## Special tokens and boundaries

The template owns BOS/EOS: when a chat template is applied, `add_bos`/`add_eos` are ignored so a
boundary token is never doubled. Generation stops on the tokenizer's own end token, and the stop
token is trimmed from the displayed answer.

`add_generation_prompt` ends a prompt with the assistant header so the model answers instead of
continuing the input; `continue_final_message` keeps the last message open so the model continues
it. They are mutually exclusive, as in the reference implementation.

## Interactive generation

`GenerationTask` gains `use_chat_template`, `system_prompt`, `add_generation_prompt`,
`continue_final_message` and `chat_options`. With the template on, each interactive request is one
turn of a conversation: history is kept as `{role, content}` messages, re-rendered every turn (so
editing the system prompt or a feature applies to the next turn). The exact templated prompt is
returned by the RPC (`task.feed` → `chat_prompt`) and can be shown or hidden in the GUI transcript
(the **Chat** menu). Model-specific toggles such as `enable_thinking` are discovered from the
template itself (`ChatTemplate.features`) and surfaced through the RPC (`task.chat`) and the GUI.

```yaml
task:
  - name: chat
    type: generation
    max_new_tokens: 128
    use_chat_template: true
    add_generation_prompt: true
    system_prompt: "You are a concise assistant."
    chat_options: {enable_thinking: false}
```

Shell: `template on`, `system "You are concise"`, then type a message; `template off` returns to
plain continuation.

## Batch annotation

Annotation renders each row through the same template. A lazy dataset reads three schemas:

* a plain text column (every row is a `user` message);
* a text column plus an optional `role` column (a missing role defaults to `user`; set
  `MISSING_TEMPLATE_ROLE_WARNING=TRUE` to warn);
* a multi-turn `chat` column holding a list of `{role, content}` dicts.

`system_prompt` and `prompt_prefix` add cheap instructions before the row. The **Annotate** wizard
exposes all of this. By convention the template's control tokens are kept separate from the
content's tokens (tokenized once and cached, concatenated per row, so `encode` stays fast);
`store_template_in_input_ids: true` instead tokenizes the whole rendered string, exact at every
boundary.

The RPC surface is one place: `task.chat` reads and sets `use`, `system` and template `options`,
and reports `available`/`features`; `task.feed` reports `used_chat_template` and `chat_prompt`.
