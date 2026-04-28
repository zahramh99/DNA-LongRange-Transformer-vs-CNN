import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path

# Settings
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Load data
train_seq = np.load(DATA_DIR / "train_sequences.npy")
train_labels = np.load(DATA_DIR / "train_labels.npy")
val_seq = np.load(DATA_DIR / "val_sequences.npy")
val_labels = np.load(DATA_DIR / "val_labels.npy")
test_seq = np.load(DATA_DIR / "test_sequences.npy")
test_labels = np.load(DATA_DIR / "test_labels.npy")

print(f"Sequence length: {train_seq.shape[1]} bp")
print(f"Number of tracks: {train_labels.shape[1]}")
print(f"Train samples: {len(train_seq):,}")
print(f"Val samples: {len(val_seq):,}")
print(f"Test samples: {len(test_seq):,}")

# Nucleotide composition
comp = {base: (train_seq == i).sum() / train_seq.size for i, base in enumerate("ACGT")}
gc = comp["G"] + comp["C"]
print(f"Nucleotide composition: {comp}")
print(f"GC content: {gc:.3f}")

# Label distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(train_labels.flatten(), bins=100, alpha=0.7, color="steelblue")
axes[0].set_yscale("log")
axes[0].set_title("All labels")
df_box = pd.DataFrame(train_labels[:, :30])
df_box.boxplot(ax=axes[1], rot=90, patch_artist=True)
axes[1].set_title("First 30 tracks")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "label_distributions.png")

# Correlation matrix
corr = np.corrcoef(train_labels[:1000].T)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True)
plt.title("Track correlation matrix")
plt.savefig(RESULTS_DIR / "correlation_matrix.png")

# Save statistics
stats = {
    "seq_len": int(train_seq.shape[1]),
    "n_tracks": int(train_labels.shape[1]),
    "n_train": len(train_seq),
    "n_val": len(val_seq),
    "n_test": len(test_seq),
    "gc_content": float(gc),
    "label_mean": float(train_labels.mean()),
    "label_std": float(train_labels.std())
}
with open(RESULTS_DIR / "data_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("✅ All plots saved to results/")