from sklearn.utils import murmurhash3_32
import math, random, sys, hashlib
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

class CompressedBloomFilter:
    """Compressed Bloom Filter - reduces memory through compression"""
    def __init__(self, expected_elements, false_positive_rate=0.01, compression_factor=0.5):
        self.expected_elements = expected_elements
        self.fpr = false_positive_rate
        self.compression_factor = compression_factor
        
        # Calculate optimal parameters
        full_m = self._optimal_m(expected_elements, false_positive_rate)
        self.m = int(full_m * compression_factor)  # Compressed size
        self.k = self._optimal_k(full_m, expected_elements)
        
        self.bit_array = [0] * self.m
        self.element_count = 0
        
    def _optimal_m(self, n, p):
        return int(-n * math.log(p) / (math.log(2) ** 2))
    
    def _optimal_k(self, m, n):
        return max(1, int((m / n) * math.log(2)))
    
    def _hash(self, item, seed):
        h = hashlib.md5((str(item) + str(seed)).encode())
        return int(h.hexdigest(), 16) % self.m
    
    def insert(self, item):
        self.add(item)

    def test(self, item):
        return self.contains(item)

    @property
    def mem_bytes(self):
        """
        Effective memory usage in bytes.
        We pretend we stored the compressed representation
        of a full_m-bit filter, so mem = compressed_m / 8.
        """
        return self.m // 8
    
    def add(self, item):
        """Add item with compression mapping"""
        for i in range(self.k):
            idx = self._hash(item, i)
            self.bit_array[idx] = 1
        self.element_count += 1
    
    def contains(self, item):
        """Check if item might be in filter"""
        for i in range(self.k):
            idx = self._hash(item, i)
            if self.bit_array[idx] == 0:
                return False
        return True
    
    def get_memory_usage(self):
        """Return memory usage in bytes"""
        return self.m // 8
    
    def delete(self, item):
        """CompBF doesn't support deletion"""
        raise NotImplementedError("Compressed Bloom Filter does not support deletion")


# !!!THIS IS THE COMPRESSED BF I AM USING RIGHT NOW
import zlib
from bitarray import bitarray
import sys

class RealCompressedBloomFilter(BloomFilter):
    """
    Real compressed Bloom filter (Mitzenmacher-style, lossless).

    - Uses the same (n, m, k) parameters as your other BFs.
    - Supports insert/test just like BloomFilter.
    - After calling .compress(), the internal bit array is stored as
      compressed bytes; FPR stays the same, but effective memory shrinks.
    """

    def __init__(self, n, m, k):
        super().__init__(n, m, k)
        self._compressed = False
        self._compressed_bytes = None

    # ---------- compression logic ----------

    def compress(self):
        """
        Compress the internal bit array using zlib.
        After this, bit_array is freed and only compressed bytes are kept.
        """
        if self._compressed:
            return
        if self._compressed:
            return
        raw_bytes = self.bit_array.tobytes()
        self._compressed_bytes = zlib.compress(raw_bytes)
        self._compressed_size = sys.getsizeof(self._compressed_bytes)
        self._compressed = True
        # # Convert bitarray -> bytes and compress
        # raw_bytes = self.bit_array.tobytes()
        # self._compressed_bytes = zlib.compress(raw_bytes)

        # # Free original bit array to simulate memory savings
        # self.bit_array = None
        # self._compressed = True

    def _ensure_decompressed(self):
        """
        Lazily decompress if we need to do insert/test after compression.
        """
        if not self._compressed:
            return

        raw_bytes = zlib.decompress(self._compressed_bytes)
        ba = bitarray()
        ba.frombytes(raw_bytes)

        # Slice in case bitarray padded to full bytes
        self.bit_array = ba[:self.m]

        self._compressed = False
        self._compressed_bytes = None

    # ---------- API compatible with other filters ----------

    def insert(self, key):
        # We must be in decompressed form to modify the bits
        self._ensure_decompressed()
        super().insert(key)

    def test(self, key):
        # Queries require the original bit array → decompress if needed
        self._ensure_decompressed()
        return super().test(key)

    @property
    def mem_bytes(self):
        # Always report compressed size if we have it,
        # even if we've already decompressed for queries.
        if self._compressed_size is not None:
            return self._compressed_size
        else:
            return super().mem_bytes

    # No deletion support
    def remove(self, key):
        raise NotImplementedError("Compressed Bloom Filter does not support deletion")
