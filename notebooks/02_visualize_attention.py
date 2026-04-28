import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(".")

from train.finetune_nt import NucleotideTransformerFineTuner
from evaluation.metrics import compute_distance_correlation, compute_attention_snr

# Settings
CHECKPOINT = "checkpoints/nt/best_model.pth"  # if not found, base model will be used
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/attention")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load model
if not Path(CHECKPOINT).exists():
    print("Checkpoint not found. Loading base model (no fine-tuning).")
    model = NucleotideTransformerFineTuner(
        model_name="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        n_tracks=531, freeze_backbone=True
    )
else:
    model = NucleotideTransformerFineTuner(
        model_name="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        n_tracks=531, freeze_backbone=False
    )
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Load one test sequence
test_seq = np.load(DATA_DIR / "test_sequences.npy")[:1]
seq_str = "".join(["ACGT"[b] for b in test_seq[0]])

# Extract attention from last layer
inputs = model.tokenizer([seq_str], return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model.backbone(**inputs, output_attentions=True)
attn = outputs.attentions[-1][0].cpu().numpy()  # (heads, L, L)
print(f"Attention shape: {attn.shape}")

# Average over heads
avg_attn = attn.mean(axis=0)
L = avg_attn.shape[0]

# Attention map (first 200 bp)
plt.figure(figsize=(10, 8))
plt.imshow(avg_attn[:200, :200], cmap="hot", aspect="auto")
plt.colorbar(label="Attention weight")
plt.xlabel("Key position (bp)")
plt.ylabel("Query position (bp)")
plt.title("Attention map (first 200 bp)")
plt.savefig(RESULTS_DIR / "attention_map.png")

# Distance decay
distances, attn_by_dist = compute_distance_correlation(avg_attn, max_distance=500)
plt.figure(figsize=(8, 5))
plt.plot(distances, attn_by_dist, "b-", linewidth=2)
plt.xlabel("Distance (bp)")
plt.ylabel("Mean attention weight")
plt.title("Attention decay with distance")
plt.grid(alpha=0.3)
plt.savefig(RESULTS_DIR / "attention_decay.png")

# Head diversity
head_means = attn.mean(axis=(1, 2))
plt.figure(figsize=(10, 4))
plt.bar(range(len(head_means)), head_means, color="steelblue")
plt.xlabel("Head index")
plt.ylabel("Mean attention weight")
plt.title("Attention head diversity")
plt.savefig(RESULTS_DIR / "head_diversity.png")

snr = compute_attention_snr(attn)
print(f"Attention SNR: {snr:.4f}")

# Save attention weights
np.save(RESULTS_DIR / "attention_weights.npy", attn)
print("✅ All plots saved to results/attention/")