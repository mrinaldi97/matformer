"""Sentence-aligned splitter: NLTK Punkt sentence spans aligned to token offsets."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from nltk.tokenize import PunktTokenizer

class split_and_tokenize_by_nltk_sentences_aligned:
    def __init__(self, language, chunk_size, tokenizer):
        from typing import List, Tuple
        import time
        import difflib
        import re
        from tqdm import tqdm
        import numpy as np
        self.punkt_tokenizer = PunktTokenizer(language) 
        self.tokenizer = tokenizer.tokenizer #Accessing the huggingface tokenizer inside MatformerTokenizer
        self.language = language
        self.max_tokens = chunk_size
    
    def get_sentence_spans(self, document: str):
        spans = [x for x in self.punkt_tokenizer.span_tokenize(document)]
        starts = [s[0] for s in spans]
        ends = [s[0] for s in spans[1:]] + [len(document)]
        spans = list(zip(starts, ends))
        return spans

    def align(self, sentencespans, encoding):
        tokenspans = []
        try:
            mapping = np.array(encoding["offset_mapping"])
            starts = mapping[:, 0]
            ends = mapping[:, 1]

            for sent_start, sent_end in sentencespans:
                tokenstart = int(np.searchsorted(starts, sent_start, side="right")-1)
                tokenend = int(np.searchsorted(ends, sent_end, side="left")+1)
                tokenspans.append((max(tokenstart, 0), min(tokenend, len(mapping))))
            if len(tokenspans)>0:
                tokenspans[-1]=(tokenspans[-1][0],len(mapping))
                starts = [s[0] for s in tokenspans]
                ends = [s[0] for s in tokenspans[1:]] + [len(mapping)]
                tokenspans = list(zip(starts, ends))            
            else:
                print("WARNING: Empty sequence")
        except Exception as e:
            print(f"Error. {e}")
            print(encoding)
        return tokenspans
    def trim_long_sequences(self, aligned_spans, maxlen):
        #print(maxlen)   
        new_aligned_spans = []
        
        for x in aligned_spans:
            span_length = x[1] - x[0]
            
            if span_length > maxlen:
                # Split the long span into multiple chunks of maxlen
                start = x[0]
                while start < x[1]:
                    end = min(start + maxlen, x[1])
                    new_aligned_spans.append((start, end))
                    start = end
            else:
                new_aligned_spans.append(x)
        
        return new_aligned_spans


    def lenspan(self, start, end):
        return end[1] - start[0]

    def create_final_spans(self, aligned_spans, maxlen):
        span_idx_start = 0
        span_idx_end = 0
        finalspans = []

        while span_idx_end < len(aligned_spans):
            if self.lenspan(aligned_spans[span_idx_start], aligned_spans[span_idx_end]) > maxlen:
                finalspans.append(
                    (aligned_spans[span_idx_start][0], aligned_spans[span_idx_end - 1][1])
                )
                span_idx_start = span_idx_end
            else:
                span_idx_end += 1
                if span_idx_end >= len(aligned_spans):
                    finalspans.append(
                        (aligned_spans[span_idx_start][0], aligned_spans[span_idx_end - 1][1])
                    )

        return finalspans

    def __call__(self, document, batch_idx=None, batched_encoding=None):
        if isinstance(document, bytes):
            document = document.decode("utf-8")
        if not isinstance(document, str):
            raise Exception("Document must be a string or bytes")

        if batch_idx is not None and batched_encoding is not None:
            encoding = batched_encoding[batch_idx]
        else:
            encoding = self.tokenizer(document, return_offsets_mapping=True, add_special_tokens=False)

        if hasattr(encoding, "ids"):  # it's a tokenizers.Encoding
            all_tokens = encoding.ids
            offset_mapping = encoding.offsets
        else:  # it's a BatchEncoding
            all_tokens = encoding["input_ids"]
            offset_mapping = encoding["offset_mapping"]
        try:
            if len(all_tokens) == 0:
                print("WARNING: Empty sequence")
            elif len(all_tokens) == 1:
                print("WARNING: Sequence with just one token!")
            else:
                pass
        except:
            print(f"DEBUG: Conteggio token fallito! {all_tokens}")
        spans = self.get_sentence_spans(document)
        aligned_spans = self.align(spans, {"offset_mapping": offset_mapping})
        aligned_spans = self.trim_long_sequences(aligned_spans, self.max_tokens)
        finalspans = self.create_final_spans(aligned_spans, self.max_tokens)

        chunk_ranges = [(span[0], span[1]) for span in finalspans]

        return {
            "tokens": all_tokens,
            "chunks": chunk_ranges,
        }


    def batched(self, batch):
        batched_encoding = self.tokenizer(batch, return_offsets_mapping=True, add_special_tokens=False)
        results = []
        for i, document in enumerate(batch):
            results.append(self(document, batch_idx=i, batched_encoding=batched_encoding))
        return results
