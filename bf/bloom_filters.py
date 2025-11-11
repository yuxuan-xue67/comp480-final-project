from sklearn.utils import murmurhash3_32
import math, random, sys
import pandas as pd
from bitarray import bitarray
import matplotlib.pyplot as plt
import numpy as np

# Simple hash function factory
def hashfunc(m, seed=0):
    def h(x):
        return murmurhash3_32(str(x), seed=seed, positive=True) % m
    return h

class BloomFilter():
    def __init__(self, n, m, k):
        """
        n: expected number of elements
        m: number of bits
        k: number of hash functions
        """
        self.n = n
        self.m = m
        self.k = k

        # initialize bit array
        self.bit_array = bitarray(self.m)
        self.bit_array.setall(0)

        # hash functions with different seeds
        self.hashes = [hashfunc(self.m, seed=i) for i in range(self.k)]
        
        # num of inserted elements
        self.count = 0

    def insert(self, key):
        for hash in self.hashes:
            idx = hash(key)
            self.bit_array[idx] = 1
        self.count += 1

    def test(self, key):
        return all(self.bit_array[hash(key)] for hash in self.hashes)
    
    @property
    # memory used by the bit buffer + python object overhead
    def mem_bytes(self):
        return sys.getsizeof(self.bit_array)


class CountingBloomFilter(BloomFilter):
    def __init__(self, n, m, k):
        super().__init__(n, m, k)
        self.count_array = np.zeros(self.m, dtype=int)

    def insert(self, key):
        for h in self.hashes:
            self.count_array[h(key)] += 1
        self.count += 1

    def test(self, key):
        return all(self.count_array[h(key)] > 0 for h in self.hashes)

    def remove(self, key):
        for h in self.hashes:
            idx = h(key)
            if self.count_array[idx] > 0:
                self.count_array[idx] -= 1
        self.count = max(0, self.count - 1)

    @property
    def mem_bytes(self):
        return self.count_array.nbytes


class ScalableBloomFilter:
    def __init__(self, n, m, k, growth_factor=2, saturation=0.5):
        """
        growth_factor: expansion ratio for each new filter
        saturation: threshold of filled bits before adding a new filter
        """
        self.n, self.m, self.k = n, m, k
        self.growth_factor = growth_factor
        self.saturation = saturation
        self.filters = [BloomFilter(n, m, k)]

    def _is_saturated(self, bf: BloomFilter):
        ones = bf.bit_array.count(True)
        return ones / bf.m > self.saturation

    def insert(self, key):
        bf = self.filters[-1]
        if self._is_saturated(bf):
            # create a new larger filter
            new_m = int(bf.m * self.growth_factor)
            new_n = int(bf.n * self.growth_factor)
            new_k = bf.k
            self.filters.append(BloomFilter(new_n, new_m, new_k))
            bf = self.filters[-1]
        bf.insert(key)

    def test(self, key):
        return any(bf.test(key) for bf in self.filters)

    @property
    def mem_bytes(self):
        return sum(bf.mem_bytes for bf in self.filters)


class TimeDecayingBloomFilter(CountingBloomFilter):
    def __init__(self, n, m, k, decay_rate=0.9):
        super().__init__(n, m, k)
        self.decay_rate = decay_rate

    def decay(self):
        """Apply decay to counters (simulate fading memory)."""
        self.count_array = (self.count_array * self.decay_rate).astype(int)

    def insert(self, key, weight=1):
        for h in self.hashes:
            self.count_array[h(key)] += weight
        self.count += 1
