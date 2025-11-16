import pandas as pd
import numpy as np
import hashlib
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import struct
import math

class ClassicBloomFilter:
    """Classic Bloom Filter - baseline implementation"""
    def __init__(self, expected_elements, false_positive_rate=0.01):
        self.expected_elements = expected_elements
        self.fpr = false_positive_rate
        
        # Calculate optimal m and k
        self.m = self._optimal_m(expected_elements, false_positive_rate)
        self.k = self._optimal_k(self.m, expected_elements)
        self.bit_array = [0] * self.m
        self.element_count = 0
        
    def _optimal_m(self, n, p):
        """Calculate optimal bit array size"""
        return int(-n * math.log(p) / (math.log(2) ** 2))
    
    def _optimal_k(self, m, n):
        """Calculate optimal number of hash functions"""
        return max(1, int((m / n) * math.log(2)))
    
    def _hash(self, item, seed):
        """Generate hash using MD5 with seed"""
        h = hashlib.md5((str(item) + str(seed)).encode())
        return int(h.hexdigest(), 16) % self.m
    
    def add(self, item):
        """Add item to filter"""
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
        """CBF doesn't support deletion"""
        raise NotImplementedError("Classic Bloom Filter does not support deletion")


class CountingBloomFilter:
    """Counting Bloom Filter - supports deletion"""
    def __init__(self, expected_elements, false_positive_rate=0.01, counter_size=4):
        self.expected_elements = expected_elements
        self.fpr = false_positive_rate
        self.counter_size = counter_size  # bits per counter
        
        # Calculate optimal m and k
        self.m = self._optimal_m(expected_elements, false_positive_rate)
        self.k = self._optimal_k(self.m, expected_elements)
        self.counters = [0] * self.m
        self.element_count = 0
        
    def _optimal_m(self, n, p):
        return int(-n * math.log(p) / (math.log(2) ** 2))
    
    def _optimal_k(self, m, n):
        return max(1, int((m / n) * math.log(2)))
    
    def _hash(self, item, seed):
        h = hashlib.md5((str(item) + str(seed)).encode())
        return int(h.hexdigest(), 16) % self.m
    
    def add(self, item):
        """Add item to filter"""
        for i in range(self.k):
            idx = self._hash(item, i)
            if self.counters[idx] < (2 ** self.counter_size) - 1:
                self.counters[idx] += 1
        self.element_count += 1
    
    def contains(self, item):
        """Check if item might be in filter"""
        for i in range(self.k):
            idx = self._hash(item, i)
            if self.counters[idx] == 0:
                return False
        return True
    
    def delete(self, item):
        """Remove item from filter"""
        if not self.contains(item):
            return False
        for i in range(self.k):
            idx = self._hash(item, i)
            if self.counters[idx] > 0:
                self.counters[idx] -= 1
        self.element_count -= 1
        return True
    
    def get_memory_usage(self):
        """Return memory usage in bytes"""
        return (self.m * self.counter_size) // 8


class ScalableBloomFilter:
    """Scalable Bloom Filter - grows dynamically"""
    def __init__(self, initial_capacity=1000, false_positive_rate=0.01, 
                 growth_factor=2, tightening_ratio=0.9):
        self.initial_capacity = initial_capacity
        self.base_fpr = false_positive_rate
        self.growth_factor = growth_factor
        self.tightening_ratio = tightening_ratio
        
        self.filters = []
        self.element_count = 0
        self._add_filter()
        
    def _add_filter(self):
        """Add a new filter to the chain"""
        s = len(self.filters)
        fpr = self.base_fpr * (self.tightening_ratio ** s)
        capacity = self.initial_capacity * (self.growth_factor ** s)
        new_filter = ClassicBloomFilter(int(capacity), fpr)
        self.filters.append({
            'filter': new_filter,
            'capacity': int(capacity),
            'count': 0
        })
    
    def add(self, item):
        """Add item to filter"""
        current_filter = self.filters[-1]
        
        # If current filter is at capacity, add new filter
        if current_filter['count'] >= current_filter['capacity']:
            self._add_filter()
            current_filter = self.filters[-1]
        
        current_filter['filter'].add(item)
        current_filter['count'] += 1
        self.element_count += 1
    
    def contains(self, item):
        """Check if item might be in any filter"""
        for filter_dict in self.filters:
            if filter_dict['filter'].contains(item):
                return True
        return False
    
    def get_memory_usage(self):
        """Return total memory usage in bytes"""
        return sum(f['filter'].get_memory_usage() for f in self.filters)
    
    def delete(self, item):
        """SBF doesn't support deletion directly"""
        raise NotImplementedError("Scalable Bloom Filter does not support deletion")


class TimeDecayingBloomFilter:
    """Time-decaying Bloom Filter - ages out old entries"""
    def __init__(self, expected_elements, false_positive_rate=0.01, 
                 num_windows=4, window_size=1000):
        self.expected_elements = expected_elements
        self.fpr = false_positive_rate
        self.num_windows = num_windows
        self.window_size = window_size
        
        # Create multiple time windows
        self.windows = []
        for _ in range(num_windows):
            self.windows.append(ClassicBloomFilter(expected_elements // num_windows, false_positive_rate))
        
        self.current_window = 0
        self.element_count = 0
        self.window_counts = [0] * num_windows
        
    def add(self, item):
        """Add item to current time window"""
        self.windows[self.current_window].add(item)
        self.window_counts[self.current_window] += 1
        self.element_count += 1
        
        # Rotate to next window if current is full
        if self.window_counts[self.current_window] >= self.window_size:
            self.current_window = (self.current_window + 1) % self.num_windows
            # Clear the new current window (decay oldest entries)
            self.element_count -= self.window_counts[self.current_window]
            self.windows[self.current_window] = ClassicBloomFilter(
                self.expected_elements // self.num_windows, self.fpr)
            self.window_counts[self.current_window] = 0
    
    def contains(self, item):
        """Check if item is in any active window"""
        for window in self.windows:
            if window.contains(item):
                return True
        return False
    
    def get_memory_usage(self):
        """Return total memory usage in bytes"""
        return sum(w.get_memory_usage() for w in self.windows)
    
    def delete(self, item):
        """TDBF implicitly deletes through time decay"""
        # Items naturally decay as windows rotate
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


class BloomFilterEvaluator:
    """Evaluate and compare Bloom filter variants"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.results = defaultdict(dict)
        
    def load_ids_data(self, sample_size=None):
        """Load CIC-IDS2017 dataset"""
        print(f"Loading dataset from {self.dataset_path}...")
        df = pd.read_csv(self.dataset_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def create_flow_signature(self, row):
        """Create unique signature for network flow"""
        # Combine key features to create unique flow identifier
        features = [
            str(row.get('Source IP', row.get(' Source IP', ''))),
            str(row.get('Destination IP', row.get(' Destination IP', ''))),
            str(row.get('Source Port', row.get(' Source Port', ''))),
            str(row.get('Destination Port', row.get(' Destination Port', ''))),
            str(row.get('Protocol', row.get(' Protocol', '')))
        ]
        return '|'.join(features)
    
    def run_experiment(self, df, expected_elements, test_ratio=0.2, fpr=0.01):
        """Run comparative experiment on all filters"""
        
        # Split data
        split_idx = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Create flow signatures
        print("Creating flow signatures...")
        train_flows = [self.create_flow_signature(row) for _, row in train_df.iterrows()]
        test_flows = [self.create_flow_signature(row) for _, row in test_df.iterrows()]
        
        # Initialize filters
        filters = {
            'Classic BF': ClassicBloomFilter(expected_elements, fpr),
            'Counting BF': CountingBloomFilter(expected_elements, fpr),
            'Scalable BF': ScalableBloomFilter(expected_elements // 4, fpr),
            'Time-Decaying BF': TimeDecayingBloomFilter(expected_elements, fpr, num_windows=4),
            'Compressed BF': CompressedBloomFilter(expected_elements, fpr, compression_factor=0.6)
        }
        
        print("\n" + "="*80)
        print("Running Bloom Filter Experiments for Network Intrusion Detection")
        print("="*80)
        
        for name, bf in filters.items():
            print(f"\n--- {name} ---")
            
            # Insertion test
            start_time = time.time()
            for flow in train_flows:
                bf.add(flow)
            insertion_time = time.time() - start_time
            
            # Query test
            start_time = time.time()
            true_positives = sum(1 for flow in train_flows[:1000] if bf.contains(flow))
            query_time_positive = time.time() - start_time
            
            start_time = time.time()
            false_positives = sum(1 for flow in test_flows if bf.contains(flow))
            query_time_negative = time.time() - start_time
            
            # Calculate metrics
            memory_usage = bf.get_memory_usage()
            insertion_throughput = len(train_flows) / insertion_time
            query_throughput = (len(train_flows[:1000]) + len(test_flows)) / (query_time_positive + query_time_negative)
            false_positive_rate = false_positives / len(test_flows)
            
            # Test deletion capability
            deletion_supported = "Yes"
            deletion_time = 0
            if name in ['Counting BF', 'Time-Decaying BF']:
                try:
                    start_time = time.time()
                    for flow in train_flows[:100]:
                        bf.delete(flow)
                    deletion_time = time.time() - start_time
                except:
                    deletion_supported = "No"
            else:
                deletion_supported = "No"
            
            # Store results
            self.results[name] = {
                'memory_kb': memory_usage / 1024,
                'insertion_throughput': insertion_throughput,
                'query_throughput': query_throughput,
                'fpr': false_positive_rate,
                'deletion_supported': deletion_supported,
                'deletion_time': deletion_time,
                'insertion_time': insertion_time,
                'query_time': query_time_positive + query_time_negative
            }
            
            # Print results
            print(f"Memory Usage: {memory_usage / 1024:.2f} KB")
            print(f"Insertion Throughput: {insertion_throughput:.2f} flows/sec")
            print(f"Query Throughput: {query_throughput:.2f} flows/sec")
            print(f"False Positive Rate: {false_positive_rate:.4f}")
            print(f"Deletion Support: {deletion_supported}")
            if deletion_supported == "Yes":
                print(f"Deletion Time (100 items): {deletion_time:.4f} sec")
        
        return self.results
    
    def plot_results(self):
        """Visualize comparison results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        filters = list(self.results.keys())
        
        # Memory usage
        memory = [self.results[f]['memory_kb'] for f in filters]
        axes[0, 0].bar(filters, memory, color='skyblue')
        axes[0, 0].set_ylabel('Memory (KB)')
        axes[0, 0].set_title('Memory Usage Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Throughput
        insertion_tp = [self.results[f]['insertion_throughput'] for f in filters]
        query_tp = [self.results[f]['query_throughput'] for f in filters]
        x = np.arange(len(filters))
        width = 0.35
        axes[0, 1].bar(x - width/2, insertion_tp, width, label='Insertion', color='lightgreen')
        axes[0, 1].bar(x + width/2, query_tp, width, label='Query', color='lightcoral')
        axes[0, 1].set_ylabel('Throughput (flows/sec)')
        axes[0, 1].set_title('Throughput Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(filters, rotation=45)
        axes[0, 1].legend()
        
        # False Positive Rate
        fpr = [self.results[f]['fpr'] for f in filters]
        axes[1, 0].bar(filters, fpr, color='salmon')
        axes[1, 0].set_ylabel('False Positive Rate')
        axes[1, 0].set_title('False Positive Rate Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0.01, color='r', linestyle='--', label='Target FPR')
        axes[1, 0].legend()
        
        # Deletion Support
        deletion_support = [1 if self.results[f]['deletion_supported'] == 'Yes' else 0 
                          for f in filters]
        colors = ['green' if x == 1 else 'red' for x in deletion_support]
        axes[1, 1].bar(filters, deletion_support, color=colors)
        axes[1, 1].set_ylabel('Deletion Support')
        axes[1, 1].set_title('Deletion Capability')
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_yticklabels(['No', 'Yes'])
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('bloom_filter_comparison.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved as 'bloom_filter_comparison.png'")
        plt.show()
    
    def generate_summary_table(self):
        """Generate summary table of results"""
        print("\n" + "="*120)
        print("SUMMARY TABLE - Bloom Filter Performance Comparison")
        print("="*120)
        
        header = f"{'Filter Type':<25} {'Memory (KB)':<15} {'Insert TP':<15} {'Query TP':<15} {'FPR':<12} {'Deletion':<12}"
        print(header)
        print("-"*120)
        
        for name, metrics in self.results.items():
            row = f"{name:<25} {metrics['memory_kb']:<15.2f} {metrics['insertion_throughput']:<15.0f} " \
                  f"{metrics['query_throughput']:<15.0f} {metrics['fpr']:<12.4f} {metrics['deletion_supported']:<12}"
            print(row)
        
        print("="*120)


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "../data/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    SAMPLE_SIZE = 50000 
    EXPECTED_ELEMENTS = 40000
    FALSE_POSITIVE_RATE = 0.01
    
    evaluator = BloomFilterEvaluator(DATASET_PATH)
    df = evaluator.load_ids_data(sample_size=SAMPLE_SIZE)
    results = evaluator.run_experiment(df, EXPECTED_ELEMENTS, test_ratio=0.2, fpr=FALSE_POSITIVE_RATE)
    evaluator.plot_results()
    evaluator.generate_summary_table()