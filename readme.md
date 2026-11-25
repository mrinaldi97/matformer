# Matformer

> **WARNING:** Matformer hasn't been released yet. It works, but it's still Work In Progress.

Matformer is a library designed to train, finetune and execute state-of-the-art transformer models.

Unlike common implementations, such as HuggingFace's transformers or Nvidia's Megatron, it is designed from scratch with simplicity and modularity in mind but without renouncing to all the possible improvements necessary to make transformers' training more efficient.

As the research goes on, several novelties are introduced in the libraries. However, this often leads to an explosion of their complexity, where new functions are introduced abruptely in the codebase, for example by introducing many conditional branches or by repeating many times entire part of the code, breaking the DRY principles.

The goal of Matformer is to produce a very compact and clear code, with the smallest possible amount of repetition, in line with the principles of software design. Matformer tries to hard-code as few elements as possible: it is easily extensible with additional modules that allows important modification to the models' definitions without altering the core parts of the library.

Matformers' core is so easy to read that it is ready to be used also for didactical purposes: teachers are given with the possibility to present to their student an easy to understand codebase and guide them in the understanding of how a modern transformer model works.

Matformer's main focus is on elegancy and readability of the code, an aspect often ignored in the world of codebases for deep-learning models' training.

This allows a twofold use of Matformer: on the one hand, it is a library ready for the development of models with many different possible configurations ecompassing the most common scenario adopted in 2025 for training models; on the other hand, it is a library designed from the Academia to the Academia: this means that it gives to researchers from all over the world an easily hackable transformer implementation ready to embrace many possible experimental scenarios. If, let's say, a particular custom module has to be added or modifed with respect to the base Transformer design, it is not necessary to alter even a single row of code of the library itself, but the module can be easily imported and used through the configuration files. Matformer will also take care to pack the custom code in the models'repository ready to be uploaded in platforms such as HuggingFace.

This doesn't come at expenses of speed: it is very important to exploit as much as possible the capabilities of the hardware when training complex models such as transformers. That's why Matformer already contains the code necessary to maximize training's performances, such as FlashAttention, sequence packing and high-performance fused kernels. Currently Data Parallelism is fully supported thanks to Pytorch Lightning while model parallelism is still WIP.

Altough now available only for text, its modularity allows extension to multimodality scenario. This is one of the top priorities in the library's development.

## Documentation

The [registry reference](documentation/src/components/registry.md) lists the categories and implementations. The [training tutorial](documentation/src/tutorials/train-and-run.md) trains a model from a YAML file.
