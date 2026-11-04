"""Shared numeric and naming helpers."""

import math
import random
import numpy as np

def get_datatype_for_numpy(datatype):
        if isinstance(datatype,int):
            datatype=str(datatype)
        if datatype=='8' or datatype=='uint8':
            return np.uint8
        if datatype=='16' or datatype=='uint16':
            return np.uint16
        if datatype=='32' or datatype=='uint32':
            return np.uint32
        if datatype=='float16':
            return np.float16
        if datatype=='float32':
            return np.float32
        if datatype=='float64':
            return np.float64
        else:
            raise SyntaxError


def sanify_name(_input):
    return ''.join(c if c.isalnum() or c in '_-' else '_' for c in _input)


class RandomPermutator:
    def __init__(self,max_len,seed=27):
        if max_len < 0:
            raise ValueError('max_len must be non-negative')
        self.max_len=max_len
        self.rng = random.Random(seed)
        self.a = self.create_coprime() if max_len > 1 else 1
        self.b = self.rng.randint(0, max_len - 1) if max_len else 0

    def create_coprime(self):
        while True:
            a=self.rng.randint(1,self.max_len-1)
            if math.gcd(a,self.max_len) == 1:
                return a

    def __call__(self,i):
        if 0 <= i < self.max_len:
            return (self.a*i+self.b)%self.max_len
        else:
            raise Exception("Index out of range")
