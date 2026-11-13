"""Sentence splitter that tokenizes span by span (no offset alignment)."""
from __future__ import annotations

import math

class split_and_tokenize_by_nltk_sentences:
    def __init__(self,language,chunk_size, tokenizer):
        from typing import List, Tuple
        import time
        import difflib
        import re
        from tqdm import tqdm
        from nltk.tokenize import PunktTokenizer        
        self.punkt_tokenizer = PunktTokenizer(language) 
        self.tokenizer=tokenizer
        self.language=language
        self.max_tokens=chunk_size
    def __call__(self,document):
        if isinstance(document,bytes):
            document=document.decode('utf-8')
        if not isinstance(document,str):
            raise Exception
        """
        Divide un documento in chunk di token rispettando i confini delle frasi.
        
        Args:
            document: Il testo da processare
            max_tokens: Numero massimo di token per chunk
            punkt_tokenizer: Tokenizer per le sentence di NLTK
            tokenizer: Tokenizer di HuggingFace
            language: Lingua del testo (default: italiano)
        
        Returns:
            all_tokens: Lista di tutti i token del documento (appiattita)
            token_chunks: Lista di chunk di token
            chunk_ranges: Lista di tuple (start, end) per ogni chunk
        """
        # Ottengo con NLTK gli span di ciascuna frase
        spans = [x for x in self.punkt_tokenizer.span_tokenize(document)]
        
        # Riallineo gli span in modo che la fine del primo coincida con l'inizio del secondo e così via
        starts = [s[0] for s in spans]
        ends = [s[0] for s in spans[1:]] + [len(document)]
        spans = list(zip(starts, ends))
        
        # Tokenizzo span per span fin quando non ho finito gli span, dividendo a gruppi da max_tokens token
        spans.reverse()
        token_chunks = []  # La lista con i blocchi di token divisi per segmento
        current_tokens_list = []  # Lista token di appoggio

        while len(spans) != 0:
            current_span = spans.pop()
            stringa_span = document[current_span[0]:current_span[1]]
            tokens_span = self.tokenizer.encode(stringa_span, add_special_tokens=False)
            
            if len(tokens_span) + len(current_tokens_list) <= self.max_tokens:
                # Questo span entra, lo aggiungo
                current_tokens_list.extend(tokens_span)
            else:
                # Questo span NON entra in max token. Salvo gli appoggi precedenti e passo a un nuovo segmento
                if current_tokens_list:  # Solo se non è vuoto
                    token_chunks.append(current_tokens_list)
                
                if len(tokens_span) > self.max_tokens:
                    # Siamo nella situazione in cui una singola "sentence" supera il numero di token massimi.
                    num_chunks = math.ceil(len(tokens_span) / self.max_tokens)
                    token_chunks_split = []
                    
                    for i in range(num_chunks):
                        start_idx = i * self.max_tokens
                        end_idx = min((i + 1) * self.max_tokens, len(tokens_span))
                        token_chunks_split.append(tokens_span[start_idx:end_idx])
                    
                    # Aggiungi tutti i chunk tranne l'ultimo
                    token_chunks.extend(token_chunks_split[:-1])
                    current_tokens_list = token_chunks_split[-1]
                else:
                    current_tokens_list = tokens_span
        
        if current_tokens_list:  # Salvo eventuali residui
            token_chunks.append(current_tokens_list)

        # Crea la lista di tutti i token
        all_tokens = []
        for chunk in token_chunks:
            all_tokens.extend(chunk)
        
        # Crea le tuple (start, end) per ogni chunk
        chunk_ranges = []
        current_idx = 0
        for chunk in token_chunks:
            chunk_length = len(chunk)
            chunk_ranges.append((current_idx, current_idx + chunk_length - 1))
            current_idx += chunk_length
        
        return {
            "tokens":all_tokens,
            "chunks":chunk_ranges
        }       
        pass                
    def _batched():
        raise NotImplementedError
