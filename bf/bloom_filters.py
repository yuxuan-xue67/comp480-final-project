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
    
    @property
    def fill_ratio(self):
        return self.bit_array.count(True) / self.m


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


# class ScalableBloomFilter:
#     def __init__(self, n, m, k, growth_factor=2, saturation=0.5):
#         """
#         growth_factor: expansion ratio for each new filter
#         saturation: threshold of filled bits before adding a new filter
#         """
#         self.n, self.m, self.k = n, m, k
#         self.growth_factor = growth_factor
#         self.saturation = saturation
#         self.filters = [BloomFilter(n, m, k)]

#     def _is_saturated(self, bf: BloomFilter):
#         ones = bf.bit_array.count(True)
#         return ones / bf.m > self.saturation

#     def insert(self, key):
#         bf = self.filters[-1]
#         if self._is_saturated(bf):
#             # create a new larger filter
#             new_m = int(bf.m * self.growth_factor)
#             new_n = int(bf.n * self.growth_factor)
#             new_k = bf.k
#             self.filters.append(BloomFilter(new_n, new_m, new_k))
#             bf = self.filters[-1]
#         bf.insert(key)

#     def test(self, key):
#         return any(bf.test(key) for bf in self.filters)

#     @property
#     def mem_bytes(self):
#         return sum(bf.mem_bytes for bf in self.filters)

# Optimal ScalableBloomFilter based on Almeida et al.
class ScalableBloomFilter:
    def __init__(self, n, m, k, P0=0.01, r=0.9, s=2, saturation=0.5):
        """
        P0: base false positive rate
        r: tightening ratio for error probability (0<r<1)
        m0: initial bit size
        s: growth factor for size of each new filter (2 or 4 recommended)
        saturation: threshold of filled bits before new filter
        """
        self.P0 = P0
        self.r = r
        self.s = s
        self.saturation = saturation
        
        # Use the provided m as base size m0
        self.m0 = m
        # print(f"[SBF-init] n={n}, m0={m}, k={k}, P0={P0}, r={r}, s={s}")

        self.filters = []
        self.add_filter(0)


    def add_filter(self, i):
        # P_i = P0 * r^i
        Pi = self.P0 * (self.r ** i)
        # optimal number of hashes
        ki = math.ceil(math.log2(1 / Pi))
        # m_i grows geometrically by s^i
        mi = int(self.m0 * (self.s ** i))
        ni = int((mi * (math.log(2) ** 2)) / abs(math.log(Pi)))  # capacity estimate
        self.filters.append(BloomFilter(ni, mi, ki))

    def _is_saturated(self, bf: BloomFilter):
        return bf.fill_ratio > self.saturation

    def insert(self, key):
        bf = self.filters[-1]
        if self._is_saturated(bf):
            self.add_filter(len(self.filters))
            bf = self.filters[-1]
        bf.insert(key)

    def test(self, key):
        return any(bf.test(key) for bf in self.filters)

    @property
    def total_false_positive_bound(self):
        # P_total ≤ P0 / (1 - r)
        return self.P0 / (1 - self.r)

    @property
    def mem_bytes(self):
        return sum(bf.mem_bytes for bf in self.filters)


# class TimeDecayingBloomFilter(CountingBloomFilter):
#     def __init__(self, n, m, k, decay_rate=0.9):
#         super().__init__(n, m, k)
#         self.decay_rate = decay_rate

#     def decay(self):
#         """Apply decay to counters (simulate fading memory)."""
#         self.count_array = (self.count_array * self.decay_rate).astype(int)

#     def insert(self, key, weight=1):
#         for h in self.hashes:
#             self.count_array[h(key)] += weight
#         self.count += 1

class TimeDecayingBloomFilter(CountingBloomFilter):
    # I tuned the decay_factor and epoch a bit to balance the decay speed
    def __init__(self, n, m, k, decay_factor=0.9, epoch=100):
        super().__init__(n, m, k)
        self.decay_factor = decay_factor # λ
        self.epoch = epoch # T
        self._insertions = 0 # count since last decay

    def insert(self, key, weight=1):
        for h in self.hashes:
            self.count_array[h(key)] += weight
        self.count += 1
        self._insertions += 1

        # decay after every epoch
        if self._insertions >= self.epoch:
            self.decay()
            self._insertions = 0

    def decay(self):
        """Apply exponential decay to all counters."""
        self.count_array = np.round(self.count_array * self.decay_factor).astype(int)
        self.count_array[self.count_array < 1] = 0  # reset tiny decayed counts to 0

    def estimate(self, key):
        """Estimate frequency count after decay."""
        return min(self.count_array[h(key)] for h in self.hashes)

    # Ignore remove
    def remove(self, key):
        pass
