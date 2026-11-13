"""Built-in splitters used by PretokenizationStrategy.

Custom splitters can still be dropped in the mdat's "functions" folder: the
dynamic lookup in PretokenizationStrategy._find_splitter_class checks this
package's module globals first, then scans the functions folder.
"""
from .nltk_sentence import split_and_tokenize_by_nltk_sentences
from .nltk_sentence_aligned import split_and_tokenize_by_nltk_sentences_aligned
from .small_splitter import SmallSplitter

__all__ = [
    'SmallSplitter',
    'split_and_tokenize_by_nltk_sentences_aligned',
    'split_and_tokenize_by_nltk_sentences',
]
