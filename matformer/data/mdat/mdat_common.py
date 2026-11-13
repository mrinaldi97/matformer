"""Shared utilities and exceptions for the mdat package.

Extracted from mdat.py. The exception classes used to be defined three
separate times inside mdat.py (with duplicates); they are consolidated here
so that every module of the package raises and catches the same class objects.
"""
from __future__ import annotations

import math
import random

import numpy as np

__all__ = [
    # Exceptions
    'MDatNotFound', 'MDatAlreadyPresent', 'SubmDatNotFound', 'SubMdatMissesStrategy',
    'SubMdatAlreadyExists', 'MDatIsReadOnly', 'MdatIsWriteOnly', 'TooManySubmDats',
    'MDatAlreadyShuffled', 'NameAlreadyUsed', 'StrategyNotRegistered', 'MDatNotShuffled',
    'MissingStrategyKey', 'StrategyNotFound', 'SplitterClassNotFound', 'MissingDataKey',
    'MissingSplitterOutput', 'InvalidChunksFormat',
    # Utilities
    'get_datatype_for_numpy', 'sanify_name', 'RandomPermutator',
]

# Exceptions


class MDatNotFound(Exception):
    """Raised when dataset is not found."""
    pass

class MDatAlreadyPresent(Exception):
    """Raised when dataset already exists and overwrite is False."""
    pass

class SubmDatNotFound(Exception):
    """Raised when sub-dataset is not found."""
    pass

class SubMdatMissesStrategy(Exception):
    """Raised when a submdat has not been pretokenized with a given strategy."""
    pass

class SubMdatAlreadyExists(Exception):
    """Raised when a submdat already exists and overwrite is False."""
    pass

class MDatIsReadOnly(Exception):
    """Raised when trying to modify readonly dataset."""
    pass

class MdatIsWriteOnly(Exception):
    pass

class TooManySubmDats(Exception):
    """Raised when too many submdats."""
    #print("FATAL: You've reached the maximum amount of Sub-dataset allowed in the Mdat. Please, increase the DSMAP_BITS in the manifest.")
    pass

class MDatAlreadyShuffled(Exception):
    """Raised when mdat is already shuffled."""
    pass

class NameAlreadyUsed(Exception):
    """Raised when name is already used."""
    pass

class StrategyNotRegistered(Exception):
    """Raised when strategy is not registered."""
    pass

class MDatNotShuffled(Exception):
    """Raised when an operation requires a shuffled view but it is not shuffled."""
    pass

class MissingStrategyKey(Exception):
    pass

class StrategyNotFound(Exception):
    pass

class SplitterClassNotFound(Exception):
    pass

class MissingDataKey(Exception):
    pass

class MissingSplitterOutput(Exception):
    pass

class InvalidChunksFormat(Exception):
    pass

def get_datatype_for_numpy(datatype):
    """Map a datatype descriptor ('8'/'uint8'/'float32'/...) to a numpy dtype."""
    if isinstance(datatype, int):
        datatype = str(datatype)
    mapping = {
        '8': np.uint8, 'uint8': np.uint8,
        '16': np.uint16, 'uint16': np.uint16,
        '32': np.uint32, 'uint32': np.uint32,
        'float16': np.float16,
        'float32': np.float32,
        'float64': np.float64,
    }
    try:
        return mapping[datatype]
    except KeyError:
        raise ValueError(f"Unknown datatype descriptor: {datatype!r}")

def sanify_name(_input):
    return ''.join(c if c.isalnum() or c in '_-' else '_' for c in _input)
    
class RandomPermutator:
    """Deterministic pseudo-random permutation of range(max_len) via an affine
    map (a*i + b mod max_len, with a coprime to max_len). Uses its own RNG so
    it never disturbs the global random state."""

    def __init__(self, max_len, seed=27):
        if max_len < 1:
            raise ValueError("RandomPermutator requires max_len >= 1")
        self.max_len = max_len
        self._rng = random.Random(seed)
        if max_len == 1:
            self.a = 1
            self.b = 0
        else:
            self.a = self._create_coprime()
            self.b = self._rng.randint(0, max_len - 1)

    def _create_coprime(self):
        while True:
            a = self._rng.randint(1, self.max_len - 1)
            if math.gcd(a, self.max_len) == 1:
                return a

    def __call__(self, i):
        if not 0 <= i < self.max_len:
            raise IndexError(f"Index {i} out of range for permutation of {self.max_len}")
        return (self.a * i + self.b) % self.max_len
