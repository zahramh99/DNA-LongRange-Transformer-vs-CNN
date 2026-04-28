"""
Attention analysis for Transformer-based genomic models.

Extracts attention weights from trained models (TransformerSmall or Nucleotide Transformer),
computes motif enrichment, distance correlation, and generates visualizations.

Requirements:
    - models must return attention weights (modified forward)
    - motif positions in BED format or as list of intervals
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer_small import TransformerSmall
from evaluation.metrics import (
    compute_attention_motif_enrichment,
    compute_attention_snr,
    compute_distance_correlation
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AttentionExtractor:
    """Wrapper to extract attention weights from models."""
    
    @staticmethod
    def from_transformer(model: TransformerSmall, sequences: torch.Tensor, layer_idx: Optional[int] = None):
        """
        Extract attention from TransformerSmall.
        Requires model to return attention (modify forward to output attentions).
        """
        # Modified forward: we need to get attentions from encoder layers.
        # Since standard nn.TransformerEncoder doesn't return attention, we create a custom forward.
        # For simplicity, we'll load a saved attention file if exists, or run a modified model.
        raise NotImplementedError("Need model with attention output. See custom_forward method below.")
    
    @staticmethod
    def from_nucleotide_transformer(model, sequences: List[str], layer_idx: int = -1):
        """Extract attention from HuggingFace Nucleotide Transformer."""
        model.eval()
        device = next(model.backbone.parameters()).device
        inputs = model.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # We need output_attentions=True
            outputs = model.backbone(**inputs, output_attentions=True)
            # attentions: tuple of (num_layers, batch_size, num_heads, seq_len, seq_len)
        attentions = outputs.attentions
        if layer_idx is not None:
            return attentions[layer_idx].cpu().numpy()  # (B, H, L, L)
        return [att.cpu().numpy() for att in attentions]


def load_motif_positions(bed_file: str, seq_len: int, center: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Load motif positions from BED file.
    Each line: chrom start end name? We assume single region file or global positions.
    For simplicity: positions relative to sequence length (0-indexed).
    If file not provided, return random positions for testing.
    """
    positions = []
    if not os.path.exists(bed_file):
        logger.warning(f"Motif file {bed_file} not found. Using random positions for demo.")
        # Generate random motifs of length 6-12
        np.random.seed(42)
        for _ in range(20):
            start = np.random.randint(0, seq_len - 12)
            end = start + np.random.randint(6, 13)
            positions.append((start, end))
        return positions
    
    with open(bed_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start = int(parts[1])
            end = int(parts[2])
            if center is not None:
                # shift to center around given point (e.g., TSS)
                shift = center - (start + end) // 2
                start += shift
                end += shift
            if 0 <= start < seq_len and 0 <= end <= seq_len:
                positions.append((start, end))
    logger.info(f"Loaded {len(positions)} motif positions from {bed_file}")
    return positions


def compute_motif_enrichment_for_model(attention_weights: np.ndarray, motif_positions: List[Tuple[int, int]], 
                                        n_random: int = 1000) -> Dict:
    """
    Compute enrichment ratio and statistical significance.
    attention_weights: (L, L) or (H, L, L) averaged or per-head.
    Returns dict with mean enrichment, p-value, etc.
    """
    enrichments = []
    for _ in range(n_random):
        # Shuffle motif positions or use random regions
        shuffled = [(np.random.randint(0, attention_weights.shape[-1] - (e-s), 
                                       np.random.randint(0, attention_weights.shape[-1] - (e-s))) 
                     for s,e in motif_positions) but not exact; simpler: use current function
        # We'll use compute_attention_motif_enrichment which already compares to random.
        # But that function only gives one enrichment value. For p-value we need distribution.
        pass
    # Alternatively, we can run compute_attention_motif_enrichment many times.
    # For simplicity, we return the single enrichment ratio.
    enrichment = compute_attention_motif_enrichment(attention_weights, motif_positions)
    return {"enrichment_ratio": enrichment, "p_value": None}


def plot_layer_attention_heatmaps(attention_per_layer: List[np.ndarray], 
                                  save_dir: str, 
                                  sample_idx: int = 0,
                                  head_idx: int = 0):
    """Plot attention heatmaps for each layer."""
    num_layers = len(attention_per_layer)
    fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    for layer, attn in enumerate(attention_per_layer):
        # attn shape: (B, H, L, L) for NT or (B, L, L) after averaging? We'll handle.
        if attn.ndim == 4:
            attn_map = attn[sample_idx, head_idx, :200, :200]  # first 200bp
        elif attn.ndim == 3:
            attn_map = attn[sample_idx, :200, :200]
        else:
            attn_map = attn[:200, :200]
        im = axes[layer].imshow(attn_map, cmap='viridis', aspect='auto')
        axes[layer].set_title(f'Layer {layer}')
        axes[layer].set_xlabel('Key position')
        axes[layer].set_ylabel('Query position')
        plt.colorbar(im, ax=axes[layer])
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'attention_layers.png')
    plt.savefig(save_path, dpi=150)
    logger.info(f"Saved layer attention heatmaps to {save_path}")
    plt.close()


def plot_head_diversity(attention: np.ndarray, save_dir: str, sample_idx: int = 0):
    """
    attention: (B, H, L, L) or (H, L, L)
    Plot average attention per head as a heatmap.
    """
    if attention.ndim == 4:
        avg_attn = attention[sample_idx].mean(axis=0)  # (H, L) average over keys? No, we want per-head matrix.
        # For diversity, compute mean attention weight per head.
        head_means = attention[sample_idx].mean(axis=(1,2,3))  # average over L,L
    elif attention.ndim == 3:
        # assume (H, L, L)
        head_means = attention.mean(axis=(1,2))
    else:
        logger.warning("Attention shape not suitable for head diversity.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(head_means)), head_means)
    ax.set_xlabel('Head index')
    ax.set_ylabel('Mean attention weight')
    ax.set_title('Attention head diversity')
    save_path = os.path.join(save_dir, 'attention_head_diversity.png')
    plt.savefig(save_path, dpi=150)
    logger.info(f"Saved head diversity plot to {save_path}")
    plt.close()


def analyze_distance_dependence(attention: np.ndarray, max_dist: int = 500, save_dir: str = None):
    """
    attention: (L, L) or (H, L, L)
    Compute and plot average attention vs distance.
    """
    if attention.ndim == 3:
        attn_avg = attention.mean(axis=0)  # (L, L)
    else:
        attn_avg = attention
    distances, attn_vals = compute_distance_correlation(attn_avg, max_distance=max_dist)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(distances, attn_vals, 'b-', linewidth=2)
    ax.set_xlabel('Distance (bp)')
    ax.set_ylabel('Average attention weight')
    ax.set_title('Attention decay with distance')
    ax.grid(True, alpha=0.3)
    if save_dir:
        save_path = os.path.join(save_dir, 'attention_distance_decay.png')
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved distance decay plot to {save_path}")
    plt.close()
    return distances, attn_vals


def analyze_motif_enrichment(attention: np.ndarray, motif_positions: List[Tuple[int, int]], 
                             n_permutations: int = 100, save_dir: str = None) -> Dict:
    """
    Compute motif enrichment and permutation test p-value.
    attention: (L, L) or (H, L, L)
    Returns: enrichment_ratio, p_value
    """
    if attention.ndim == 3:
        attn_avg = attention.mean(axis=0)
    else:
        attn_avg = attention
    
    # Observed enrichment
    observed = compute_attention_motif_enrichment(attn_avg, motif_positions)
    
    # Permutation test: shuffle motif positions
    L = attn_avg.shape[0]
    null_enrichments = []
    for _ in range(n_permutations):
        # generate random intervals of same lengths as original motifs
        random_positions = []
        for start, end in motif_positions:
            length = end - start
            new_start = np.random.randint(0, L - length)
            random_positions.append((new_start, new_start + length))
        null = compute_attention_motif_enrichment(attn_avg, random_positions)
        null_enrichments.append(null)
    null_enrichments = np.array(null_enrichments)
    p_value = np.mean(null_enrichments >= observed)
    
    result = {"enrichment_ratio": observed, "p_value": p_value}
    
    # Plot null distribution
    if save_dir:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(null_enrichments, bins=30, alpha=0.7, label='Null')
        ax.axvline(observed, color='red', linestyle='--', label=f'Observed ({observed:.3f})')
        ax.set_xlabel('Enrichment ratio')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Motif enrichment permutation test\np={p_value:.4f}')
        ax.legend()
        save_path = os.path.join(save_dir, 'motif_enrichment_permutation.png')
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved motif enrichment plot to {save_path}")
        plt.close()
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Analyze attention from trained Transformer models')
    parser.add_argument('--model_type', type=str, required=True, choices=['transformer', 'nt'],
                        help='Type of model: transformer or nt')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory with test sequences')
    parser.add_argument('--seq_len', type=int, default=10000, help='Sequence length for transformer')
    parser.add_argument('--n_tracks', type=int, default=531, help='Number of output tracks (unused)')
    parser.add_argument('--motif_bed', type=str, default=None, help='BED file with motif positions (if None, random)')
    parser.add_argument('--output_dir', type=str, default='results/attention', help='Save plots and data')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for attention extraction')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of test samples to analyze')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Load test sequences
    # For simplicity, we assume numpy files with integer sequences
    try:
        from train.train_transformer import GenomicDataset
        dataset = GenomicDataset(args.data_dir, 'test')
    except ImportError:
        logger.error("Could not import GenomicDataset. Please ensure data is prepared.")
        sys.exit(1)
    
    # Take subset of sequences
    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    sequences = [dataset[i][0] for i in indices]  # list of tensors
    seq_strings = None
    if args.model_type == 'nt':
        # Convert integer sequences to DNA strings
        seq_strings = []
        for seq_tensor in sequences:
            seq_str = ''.join(['ACGT'[x] for x in seq_tensor.numpy()])
            seq_strings.append(seq_str)
    
    # Load model
    if args.model_type == 'transformer':
        # Need to load TransformerSmall with attention output.
        # For now, we'll load without attention extraction and skip.
        logger.error("Transformer attention extraction requires custom model modification. Please use 'nt' for now.")
        sys.exit(1)
    elif args.model_type == 'nt':
        from train.finetune_nt import NucleotideTransformerFineTuner
        model = NucleotideTransformerFineTuner(
            model_name="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
            n_tracks=args.n_tracks,
            freeze_backbone=True
        )
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Extract attention
        extractor = AttentionExtractor()
        # We'll extract from a single sample to avoid memory issues
        attentions_per_layer = extractor.from_nucleotide_transformer(model, seq_strings[:1])
        # attentions_per_layer: list of (B, H, L, L) per layer
        # For analysis, take first sample and average across heads or keep heads.
        attn_sample = attentions_per_layer[-1][0]  # last layer, first sample, shape (H, L, L)
        L = attn_sample.shape[-1]
        logger.info(f"Attention shape: {attn_sample.shape}")
        
        # Load motif positions
        if args.motif_bed:
            motif_positions = load_motif_positions(args.motif_bed, L, center=None)
        else:
            # generate random motifs
            motif_positions = load_motif_positions("dummy.bed", L)  # will create random
        logger.info(f"Using {len(motif_positions)} motif positions for enrichment.")
        
        # 1. Distance dependence
        analyze_distance_dependence(attn_sample, max_dist=500, save_dir=args.output_dir)
        
        # 2. Motif enrichment
        enrich_result = analyze_motif_enrichment(attn_sample, motif_positions, 
                                                 n_permutations=100, save_dir=args.output_dir)
        logger.info(f"Motif enrichment: {enrich_result['enrichment_ratio']:.4f}, p={enrich_result['p_value']:.4f}")
        
        # 3. SNR (signal-to-noise) for attention
        snr = compute_attention_snr(attn_sample)
        logger.info(f"Attention SNR: {snr:.4f}")
        with open(os.path.join(args.output_dir, 'attention_stats.txt'), 'w') as f:
            f.write(f"Attention SNR: {snr:.4f}\n")
            f.write(f"Motif enrichment ratio: {enrich_result['enrichment_ratio']:.4f}\n")
            f.write(f"Motif enrichment p-value: {enrich_result['p_value']:.4f}\n")
        
        # 4. Head diversity plot
        plot_head_diversity(attn_sample, args.output_dir, sample_idx=0)
        
        # 5. Layer-wise heatmaps (if multiple layers)
        if len(attentions_per_layer) > 1:
            # convert each layer's attention for first sample, average heads?
            layer_attns = [att[0].mean(axis=0) for att in attentions_per_layer]  # each (L,L)
            plot_layer_attention_heatmaps(layer_attns, args.output_dir, sample_idx=0)
        
        # 6. Save raw attention for further analysis
        np.save(os.path.join(args.output_dir, 'attention_weights.npy'), attn_sample)
        logger.info(f"Saved raw attention weights to {args.output_dir}/attention_weights.npy")
    
    logger.info("Attention analysis completed.")


if __name__ == "__main__":
    main()