This folder contains the actual implementation of the Neural Networks components of Matformer.

[!WarningBox]Help: What are the annotation at the top of each class? Read more here: [[MatformerModule#class-annotations]]
[!WarningBox]Help: Where are torch.nn components? What is that cache.registry.create(...) used to instantiate modules? Read more here [[MatformerRegistry]]
* attention.py => Contains an implementation of a MultiHeadAttention, but it doesn't include the actual kernel. Using the [[registry]] it is indeed possible to load the attention kernel you prefer (or the best performing for your system). This selection is done automatically. The attention.py file also include a "Mechanistic-intepretability-friendly" attention implementation, that exposes the virtual weights of each single head in order to do studies and experiments dividing the QK circuit from the OV circuit. More info here: [[interpretability]]. For further info, refer to the [[class documentation]]

* transformer.py => Contains the classes [[NakedTransformer]] and [[TransformerBlock]]. NakedTransformer just stacks together as many TransformerBlocks as required by the [[config]]. TranformerBlock is a single complete transformer block supporting both pre-normalization and post-normalization styles.

* heads.py => Contains head and tail of the model. [[TransformerWithEmbeddingHead]] adds a linear embedding head to the [[NakedTransformer]] module, for example in order to embed tokens into latent. It also supports learnable positional embedding, to support legacy models. [[TransformerWithLMHead]] initialize a [[TransformerWithEmbeddingHead]] but adds an LMHead at the end of the [[NakedTransformer]]. Tying the weights of the embedding head and of the LM head is supported.

* mixin/
    Folder containing final classes that enable models' functions, such as token generation, masked language models inference and so on.
    * mixin/text_generation.py => Adds to the [[TransformerWithLMHead]] module a method to autoregressively generate tokens;

* models/
    Folder containing the final wrapper for base model (ex. TransformerWithLMHead) and useful "mixin" classes, such as GenerationMixin.
    For example, [[LanguageModel]] class is the more general, enabling a generic GPT-like and/or BERT-like model to be trained and used.
