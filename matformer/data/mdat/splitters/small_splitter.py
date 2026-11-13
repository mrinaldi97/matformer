"""SmallSplitter: sentence-aware chunking via NLTK Punkt + offset mappings."""
from __future__ import annotations

from typing import Any, Dict, List

from nltk.tokenize import PunktTokenizer

class SmallSplitter:
    def __init__(self, language: str, chunk_size: int, tokenizer: Any):
        self.punkt = PunktTokenizer(language)
        self.tokenizer = getattr(tokenizer, "tokenizer", tokenizer) 
        self.chunk_size = chunk_size

    def _get_encoding_data(self, encoding) -> tuple:
        if hasattr(encoding, "ids"): 
            if len(encoding.ids)<=10:
               print(f"WARNING: Sequence (in batch) has just {len(encoding.ids)} tokens.")
               try:
                  print(self.tokenizer.decode(encoding.ids))
               except:
                  pass
            return encoding.ids, encoding.offsets
        if len(encoding['input_ids'])<=10:
            print(f"WARNING: Sequence has just {len(encoding['input_ids'])} tokens.")
            try:
               print(self.tokenizer.decode(encoding['input_ids']))
            except:
               pass
        return encoding["input_ids"], encoding["offset_mapping"]

    def _process(self, text: str, encoding) -> Dict[str, Any]:
        ids, offsets = self._get_encoding_data(encoding)
        sent_spans = self.punkt.span_tokenize(text)
        
        chunks, curr_chunk = [], []
        tok_idx, n_tokens = 0, len(ids)

        for _, sent_end in sent_spans:
            sent_toks = []
            while tok_idx < n_tokens and offsets[tok_idx][0] < sent_end:
                sent_toks.append(ids[tok_idx])
                tok_idx += 1
            
            # For very long sequences
            if len(sent_toks) > self.chunk_size:
                if curr_chunk: chunks.append(curr_chunk); curr_chunk = []
                for i in range(0, len(sent_toks), self.chunk_size):
                    sub = sent_toks[i : i + self.chunk_size]
                    if len(sub) == self.chunk_size: chunks.append(sub)
                    else: curr_chunk.extend(sub)
                continue

            # Normal length
            if len(curr_chunk) + len(sent_toks) > self.chunk_size:
                chunks.append(curr_chunk)
                curr_chunk = list(sent_toks)
            else:
                curr_chunk.extend(sent_toks)

        if curr_chunk: chunks.append(curr_chunk)

        ranges, cursor = [], 0
        flat_tokens = []
        for c in chunks:
            flat_tokens.extend(c)
            ranges.append((cursor, cursor + len(c)))
            cursor += len(c)
            
        return {"tokens": flat_tokens, "chunks": ranges}

    def __call__(self, document: str) -> Dict[str, Any]:
        encoding = self.tokenizer(document, return_offsets_mapping=True, add_special_tokens=False)
        return self._process(document, encoding)

    def batched(self, batch: List[str]) -> List[Dict[str, Any]]:
        encodings = self.tokenizer(
            batch, return_offsets_mapping=True, add_special_tokens=False, 
            verbose=False, padding=False, truncation=False
        )
        
        return [
            self._process(doc, {
                'input_ids': encodings['input_ids'][i],
                'offset_mapping': encodings['offset_mapping'][i]
            })
            for i, doc in enumerate(batch)
        ]
