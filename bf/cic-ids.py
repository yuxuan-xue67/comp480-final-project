import pandas as pd
import random
import time
from bloom_filters import BloomFilter, CountingBloomFilter, ScalableBloomFilter, TimeDecayingBloomFilter, RealCompressedBloomFilter
import matplotlib.pyplot as plt
import seaborn as sns
import os

if not os.path.exists("graphs/cic-ids"):
    os.makedirs("graphs/cic-ids")

# Function for dataset preparation
def prepare_cicids_dataset(path: str, test_size: int = 1000, seed: int = 42, repetition: bool = False):
    # Load dataset
    data = pd.read_csv(path)
    
    # Clean column names
    data.columns = data.columns.str.strip()
    
    # Create a custom flow identifier 
    data['Flow ID'] = data['Destination Port'].astype(str) + '-' + data['Flow Duration'].astype(str) + '-' + data['Flow Bytes/s'].astype(str)
    
    if repetition:
        urls = data['Flow ID'].dropna()  # For network flow-based test
    else:
        urls = data['Flow ID'].dropna().unique()  # Unique flows
    
    membership = urls.tolist()
    N = len(membership)
    print(f"Unique flows (membership size) N = {N}")

    # Shuffle for random sampling
    random.seed(seed) 
    random.shuffle(membership)

    # Prepare for positive and negative test sets
    test_set_pos = membership[:test_size]    
    test_set_neg = [f"fake_flow{i}" for i in range(test_size)] 

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
    # False negatives: positives missing after insertion
    for x in positives:
        if not bf.test(x):
            fn += 1
    fpr = fp / len(negatives) if len(negatives) > 0 else 0
    fnr = fn / len(positives) if len(positives) > 0 else 0
    return fpr, fnr


def evaluate_filter(name, bf, insert_set, test_set_pos, test_set_neg, allow_delete=False):
    print(f"\nEvaluating {name}...")

    # 1. Insertion throughput
    insert_tp, insert_time = measure_throughput(bf.insert, insert_set)
    
    if isinstance(bf, RealCompressedBloomFilter):
        bf.compress()  
    
    # 2. Query throughput
    query_tp, query_time = measure_throughput(bf.test, test_set_pos + test_set_neg)
    
    # 3. FPR and FNR
    fpr, fnr = compute_fpr_fnr(bf, test_set_pos, test_set_neg)
    
    # 4. Memory
    mem = bf.mem_bytes
    
    results = {
        "Filter": name,
        "Memory (bytes)": mem,
        "Insert Throughput (ops/s)": round(insert_tp, 2),
        "Query Throughput (ops/s)": round(query_tp, 2),
        "FPR": round(fpr, 6),
        "FNR": round(fnr, 6)
    }
    return results


def evaluate_counting_filter_with_deletions(membership, test_set_pos, test_set_neg, deletion_ratios, m, k):
    """Mode 2: Evaluate Counting Bloom Filter with different deletion ratios"""
    results = []
    n = len(membership)
    
    for del_ratio in deletion_ratios:
        print(f"\n--- Counting Filter with {int(del_ratio*100)}% deletion ratio ---")
        bf = CountingBloomFilter(n, m, k)
        
        # Insert all flows
        insert_tp, _ = measure_throughput(bf.insert, membership)
        
        # Delete a portion of flows
        num_to_delete = int(len(membership) * del_ratio)
        to_delete = random.sample(membership, num_to_delete)
        delete_tp, _ = measure_throughput(bf.remove, to_delete)
        
        # Query throughput
        query_tp, _ = measure_throughput(bf.test, test_set_pos + test_set_neg)
        
        # FPR and FNR
        fpr, fnr = compute_fpr_fnr(bf, test_set_pos, test_set_neg)
        
        results.append({
            "Filter": "Counting",
            "Deletion Ratio": f"{int(del_ratio*100)}%",
            "Insert Throughput (ops/s)": round(insert_tp, 2),
            "Delete Throughput (ops/s)": round(delete_tp, 2),
            "Query Throughput (ops/s)": round(query_tp, 2),
            "FPR": round(fpr, 6),
            "FNR": round(fnr, 6),
            "Memory (bytes)": bf.mem_bytes
        })
    
    return results


def evaluate_time_decaying_filter(membership, test_set_pos, test_set_neg, decay_configs, m, k):
    """Mode 2: Evaluate Time-Decaying Bloom Filter with different decay parameters"""
    results = []
    n = len(membership)
    
    for config in decay_configs:
        decay_factor = config['decay_factor']
        epoch_length = config['epoch_length']
        
        print(f"\n--- Time-Decaying Filter (decay={decay_factor}, epoch={epoch_length}) ---")
        bf = TimeDecayingBloomFilter(n, m, k, decay_factor=decay_factor, epoch=epoch_length)
        
        # Simulate real-time insertion with epochs
        insert_times = []
        for i, flow in enumerate(membership):
            start = time.time()
            bf.insert(flow)
            insert_times.append(time.time() - start)
            
            # Trigger epoch if needed (assuming the filter has this mechanism)
            if hasattr(bf, 'check_epoch'):
                bf.check_epoch()
        
        avg_insert_tp = len(membership) / sum(insert_times) if sum(insert_times) > 0 else 0
        
        # Query throughput
        query_tp, _ = measure_throughput(bf.test, test_set_pos + test_set_neg)
        
        # FPR and FNR
        fpr, fnr = compute_fpr_fnr(bf, test_set_pos, test_set_neg)
        
        results.append({
            "Filter": "Time-Decaying",
            "Decay Factor": decay_factor,
            "Epoch Length": epoch_length,
            "Insert Throughput (ops/s)": round(avg_insert_tp, 2),
            "Query Throughput (ops/s)": round(query_tp, 2),
            "FPR": round(fpr, 6),
            "FNR": round(fnr, 6),
            "Memory (bytes)": bf.mem_bytes
        })
    
    return results

path = "path/" # replace this with your local path to CIC-IDS2017 csv
membership, test_set_pos, test_set_neg = prepare_cicids_dataset(path, test_size=1000)

print("\n" + "="*80)
print("MODE 1: MEMBERSHIP TESTING WITH VARYING m/n RATIOS")
print("="*80)

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
        ("Time-Decaying", TimeDecayingBloomFilter(n, m, k)),
        ("Compressed", RealCompressedBloomFilter(n, m, k))
    ]

    for name, bf in filters:
        allow_delete = name == "Counting"
        res = evaluate_filter(name, bf, membership, test_set_pos, test_set_neg, allow_delete)
        res["m/n ratio"] = ratio
        tuning_results.append(res)

df_tune = pd.DataFrame(tuning_results)
print("\n" + "="*80)
print("MODE 1 RESULTS SUMMARY")
print("="*80)
print(df_tune)
df_tune.to_csv("graphs/cic-ids/mode1_results.csv", index=False)

# Plotting Mode 1 Results
sns.set(style="whitegrid", context="talk")

# FPR vs. m/n ratio
plt.figure(figsize=(9,6))
sns.lineplot(data=df_tune, x="m/n ratio", y="FPR", hue="Filter", marker="o")
plt.title("False Positive Rate vs. m/n Ratio")
plt.ylabel("False Positive Rate")
plt.xlabel("m/n ratio (bits per element)")
plt.tight_layout()
plt.savefig("graphs/cic-ids/fpr_vs_m_n_ratio.png")
plt.close()

# Throughput (insert) vs. m/n ratio
plt.figure(figsize=(9,6))
sns.lineplot(data=df_tune, x="m/n ratio", y="Insert Throughput (ops/s)", hue="Filter", marker="o")
plt.title("Insertion Throughput vs. m/n Ratio")
plt.ylabel("Insert Throughput (ops/s)")
plt.xlabel("m/n ratio (bits per element)")
plt.tight_layout()
plt.savefig("graphs/cic-ids/insert_throughput_vs_m_n_ratio.png")
plt.close()

# Query Throughput (test) vs. m/n ratio
plt.figure(figsize=(9,6))
sns.lineplot(data=df_tune, x="m/n ratio", y="Query Throughput (ops/s)", hue="Filter", marker="o")
plt.title("Query Throughput vs. m/n Ratio")
plt.ylabel("Query Throughput (ops/s)")
plt.xlabel("m/n ratio (bits per element)")
plt.tight_layout()
plt.savefig("graphs/cic-ids/query_throughput_vs_m_n_ratio.png")
plt.close()

# Memory vs m/n ratio
plt.figure(figsize=(9,6))
sns.lineplot(data=df_tune, x="m/n ratio", y="Memory (bytes)", hue="Filter", marker="o")
plt.yscale("log")
plt.title("Memory Usage vs. m/n Ratio")
plt.ylabel("Memory (bytes)")
plt.xlabel("m/n ratio (bits per element)")
plt.tight_layout()
plt.savefig("graphs/cic-ids/memory_usage_vs_m_n_ratio.png")
plt.close()

# FPR vs. FNR
plt.figure(figsize=(9,6))
sns.lineplot(data=df_tune, x="FPR", y="FNR", hue="Filter", marker="o")
plt.title("FPR vs FNR Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("False Negative Rate")
plt.tight_layout()
plt.savefig("graphs/cic-ids/fpr_vs_fnr.png")
plt.close()

# FPR vs Insert Throughput
plt.figure(figsize=(9,6))
sns.scatterplot(data=df_tune, x="FPR", y="Insert Throughput (ops/s)", hue="Filter", style="Filter", s=120)
plt.title("FPR vs Insert Throughput")
plt.xlabel("False Positive Rate")
plt.ylabel("Insert Throughput (ops/s)")
plt.tight_layout()
plt.savefig("graphs/cic-ids/fpr_vs_insert_throughput.png")
plt.close()

print("\n" + "="*80)
print("MODE 2: REAL-TIME DATA STREAM SIMULATION")
print("="*80)

m_mode2 = int(n * 5)  # Using ratio of 5 for mode 2

# 2a. Counting Bloom Filter with deletions
print("\n--- MODE 2a: Counting Bloom Filter with Deletions ---")
deletion_ratios = [0.1, 0.3, 0.5]
counting_results = evaluate_counting_filter_with_deletions(
    membership, test_set_pos, test_set_neg, deletion_ratios, m_mode2, k
)
df_counting = pd.DataFrame(counting_results)
print("\n" + "="*80)
print("COUNTING FILTER DELETION RESULTS")
print("="*80)
print(df_counting)
df_counting.to_csv("graphs/cic-ids/mode2_counting_deletions.csv", index=False)

# Plot Counting Filter results
plt.figure(figsize=(8, 5))

# FPR — solid line
sns.lineplot(
    data=df_counting,
    x="Deletion Ratio",
    y="FPR",
    label="FPR",
    marker="o",
    linestyle="-"
)

# FNR — dashed line
sns.lineplot(
    data=df_counting,
    x="Deletion Ratio",
    y="FNR",
    label="FNR",
    marker="o",
    linestyle="--"
)

plt.title("CountingBF Deletion Effects: FPR / FNR vs Delete Ratio")
plt.ylabel("FPR / FNR")
plt.xlabel("Delete Ratio")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/cic-ids/counting_deletion_impact.png")
plt.close()

# 2b. Time-Decaying Bloom Filter with different decay parameters
print("\n--- MODE 2b: Time-Decaying Bloom Filter with Decay Parameters ---")
decay_configs = [
    {'decay_factor': 0.7, 'epoch_length': 100},
    {'decay_factor': 0.7, 'epoch_length': 500},
    {'decay_factor': 0.3, 'epoch_length': 100},
    {'decay_factor': 0.3, 'epoch_length': 500}
]
decay_results = evaluate_time_decaying_filter(
    membership, test_set_pos, test_set_neg, decay_configs, m_mode2, k
)
df_decay = pd.DataFrame(decay_results)
print("\n" + "="*80)
print("TIME-DECAYING FILTER RESULTS")
print("="*80)
print(df_decay)
df_decay.to_csv("graphs/cic-ids/mode2_time_decaying.csv", index=False)

# Plot Time-Decaying Filter results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# FPR comparison
df_decay['Config'] = df_decay.apply(lambda x: f"d={x['Decay Factor']}, e={x['Epoch Length']}", axis=1)
axes[0].bar(range(len(df_decay)), df_decay['FPR'])
axes[0].set_xlabel('Configuration')
axes[0].set_ylabel('False Positive Rate')
axes[0].set_title('Time-Decaying Filter: FPR by Configuration')
axes[0].set_xticks(range(len(df_decay)))
axes[0].set_xticklabels(df_decay['Config'], rotation=45, ha='right')

# FNR comparison
axes[1].bar(range(len(df_decay)), df_decay['FNR'])
axes[1].set_xlabel('Configuration')
axes[1].set_ylabel('False Negative Rate')
axes[1].set_title('Time-Decaying Filter: FNR by Configuration')
axes[1].set_xticks(range(len(df_decay)))
axes[1].set_xticklabels(df_decay['Config'], rotation=45, ha='right')

plt.tight_layout()
plt.savefig("graphs/cic-ids/time_decaying_decay_impact.png")
plt.close()

# Combined comparison plot for Mode 2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(df_counting['Deletion Ratio'], df_counting['FPR'], marker='o', label='FPR', linewidth=2)
plt.plot(df_counting['Deletion Ratio'], df_counting['FNR'], marker='s', label='FNR', linewidth=2)
plt.xlabel('Deletion Ratio')
plt.ylabel('Rate')
plt.title('Counting Filter: Impact of Deletions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
x_labels = [f"d={r['decay_factor']}\ne={r['epoch_length']}" for r in decay_configs]
x_pos = range(len(df_decay))
plt.plot(x_pos, df_decay['FPR'], marker='o', label='FPR', linewidth=2)
plt.plot(x_pos, df_decay['FNR'], marker='s', label='FNR', linewidth=2)
plt.xlabel('Configuration')
plt.ylabel('Rate')
plt.title('Time-Decaying Filter: Impact of Decay Parameters')
plt.xticks(x_pos, x_labels, fontsize=8)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graphs/cic-ids/mode2_combined_comparison.png")
plt.close()

print("EXPERIMENT COMPLETE!")
