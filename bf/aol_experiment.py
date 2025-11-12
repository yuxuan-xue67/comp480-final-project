import time
import random
import pandas as pd
from bloom_filters import BloomFilter, CountingBloomFilter, ScalableBloomFilter, TimeDecayingBloomFilter
import matplotlib.pyplot as plt
import seaborn as sns

# Function for dataset preparation
def prepare_dataset(path: str, test_size: int = 1000, seed: int = 42):
    # load dataset
    data = pd.read_csv(path, sep="\t")
    urls = data.ClickURL.dropna().unique()
    membership = urls.tolist()
    N = len(membership)
    print(f"unique urls (membership size) N = {N}")

    # shuffle for random sampling
    random.seed(seed) 
    random.shuffle(membership)

    # prepare for pos and neg test set
    test_set_pos = membership[:test_size]    
    test_set_neg = [f"fake{i}.com" for i in range(test_size)] 

    return membership, test_set_pos, test_set_neg


# Functions for evaluation metrics measurement
def measure_throughput(func, data):
    start = time.time()
    for x in data:
        func(x)
    duration = time.time() - start
    throughput = len(data) / duration if duration > 0 else 0
    return throughput, duration

def compute_fpr_fnr(bf, positives, negatives):
    fp = fn = 0
    # False positives: negatives wrongly detected as present
    for x in negatives:
        if bf.test(x):
            fp += 1
    # False negatives: positives missing after insertion (relevant for decaying / counting deletions)
    for x in positives:
        if not bf.test(x):
            fn += 1
    fpr = fp / len(negatives)
    fnr = fn / len(positives)
    return fpr, fnr


def evaluate_filter(name, bf, insert_set, test_set_pos, test_set_neg, allow_delete=False):
    print(f"\nEvaluating {name}...")
    
    # 1. Insertion throughput
    insert_tp, insert_time = measure_throughput(bf.insert, insert_set)
    
    # 2. Query throughput
    query_tp, query_time = measure_throughput(bf.test, test_set_pos + test_set_neg)
    
    # 3. FPR and FNR
    fpr, fnr = compute_fpr_fnr(bf, test_set_pos, test_set_neg)
    
    # 4. Deletion test (if supported)
    deletion_success = None
    if allow_delete and hasattr(bf, "remove"):
        random.seed(42) 
        to_delete = random.sample(insert_set, min(500, len(insert_set)))
        for x in to_delete:
            bf.remove(x)
        # Check if deletions introduced false negatives
        _, fnr_after = compute_fpr_fnr(bf, insert_set, test_set_neg)
        deletion_success = fnr_after #< 0.05  # arbitrary threshold for sanity check
    
    # 5. Memory
    mem = bf.mem_bytes
    
    results = {
        "Filter": name,
        "Memory (bytes)": mem,
        "Insert Throughput (ops/s)": round(insert_tp, 2),
        "Query Throughput (ops/s)": round(query_tp, 2),
        "FPR": round(fpr, 6),
        "FNR": round(fnr, 6),
        "Deletion Stable?": deletion_success
    }
    return results

# Get dataset
path = "user-ct-test-collection-01.txt"
membership, test_set_pos, test_set_neg = prepare_dataset(path, test_size=1000)

# Create Bloom filters with similar parameters for comparison
ratios = [1, 3, 5, 10]
n = len(membership)
k = 4
tuning_results = []

for ratio in ratios:
    m = int(n * ratio)
    print(f"\n===== Testing m/n ratio = {ratio} (m={m}) =====")

    filters = [
        ("Classic", BloomFilter(n, m, k)),
        ("Counting", CountingBloomFilter(n, m, k)),
        ("Scalable", ScalableBloomFilter(n, m, k)),
        ("Time-Decaying", TimeDecayingBloomFilter(n, m, k))
    ]

    for name, bf in filters:
        allow_delete = name == "Counting"
        res = evaluate_filter(name, bf, membership, test_set_pos, test_set_neg, allow_delete)
        res["m/n ratio"] = ratio
        tuning_results.append(res)

df_tune = pd.DataFrame(tuning_results)
print(df_tune)

# Visualization
sns.set(style="whitegrid", context="talk")

# FPR vs. m/n ratio
plt.figure(figsize=(9,6))
sns.lineplot(data=df_tune, x="m/n ratio", y="FPR", hue="Filter", marker="o")
plt.title("False Positive Rate vs. m/n Ratio")
plt.ylabel("False Positive Rate")
plt.xlabel("m/n ratio (bits per element)")
plt.tight_layout()
plt.show()

# Throughput (insert) vs. m/n ratio
plt.figure(figsize=(9,6))
sns.lineplot(data=df_tune, x="m/n ratio", y="Insert Throughput (ops/s)", hue="Filter", marker="o")
plt.title("Insertion Throughput vs. m/n Ratio")
plt.ylabel("Insert Throughput (ops/s)")
plt.xlabel("m/n ratio (bits per element)")
plt.tight_layout()
plt.show()

ratio_to_plot = 3
subset = df_tune[df_tune["m/n ratio"] == ratio_to_plot]

plt.figure(figsize=(9,6))
sns.barplot(data=subset, x="Filter", y="FPR", palette="Blues_d")
plt.title(f"FPR Comparison (m/n={ratio_to_plot})")
plt.ylabel("False Positive Rate")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,6))
sns.barplot(data=subset, x="Filter", y="Insert Throughput (ops/s)", palette="Greens_d")
plt.title(f"Insertion Throughput Comparison (m/n={ratio_to_plot})")
plt.ylabel("Insert Throughput (ops/s)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,6))
sns.scatterplot(data=df_tune, x="Memory (bytes)", y="FPR", hue="Filter", style="Filter", s=120)
plt.title("Memory–Accuracy Tradeoff")
plt.xlabel("Memory (bytes)")
plt.ylabel("False Positive Rate")
plt.tight_layout()
plt.show()