"""
Comprehensive model comparison script for CNN, Transformer, and Nucleotide Transformer.

Loads trained models, evaluates on test set, computes metrics, and generates:
    - Comparison table (CSV/LaTeX)
    - Attention visualizations (if available)
    - Performance vs sequence length plots
    - Statistical significance tests (paired t-test, McNemar)
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cnn_basenji2 import Basenji2CNN
from models.transformer_small import TransformerSmall
from train.finetune_nt import NucleotideTransformerFineTuner, SequenceTextDataset, collate_fn
from evaluation.metrics import (
    evaluate_all, compute_pearson, compute_spearman, compute_attention_snr,
    compute_distance_correlation
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_cnn_model(checkpoint_path: str, seq_len: int, n_tracks: int, device: torch.device):
    model = Basenji2CNN(seq_len=seq_len, n_tracks=n_tracks, n_channels=64)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info(f"Loaded CNN model from {checkpoint_path}")
    return model


def load_transformer_model(checkpoint_path: str, seq_len: int, n_tracks: int,
                           d_model: int, nhead: int, num_layers: int, device: torch.device):
    model = TransformerSmall(
        vocab_size=4, d_model=d_model, nhead=nhead, num_layers=num_layers,
        max_len=seq_len, n_tracks=n_tracks, dropout=0.0  # dropout 0 for eval
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info(f"Loaded Transformer model from {checkpoint_path}")
    return model


def load_nt_model(checkpoint_path: str, model_name: str, n_tracks: int, device: torch.device):
    model = NucleotideTransformerFineTuner(
        model_name=model_name, n_tracks=n_tracks, dropout=0.0, freeze_backbone=False
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info(f"Loaded NT model from {checkpoint_path}")
    return model


def evaluate_model(model, dataloader, device, task_type='regression', return_predictions=False):
    """Run model on entire dataloader and return metrics and optionally predictions."""
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, tuple):
                if len(batch) == 2:
                    # For CNN/Transformer: (sequences, labels)
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    preds = model(x)
                else:
                    # For NT: sequences are strings, collate_fn returns (seq_list, labels)
                    seq_list, y = batch
                    y = y.to(device)
                    preds = model(seq_list)
            else:
                # Fallback
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    metrics = evaluate_all(all_labels, all_preds, task_type=task_type)
    if return_predictions:
        return metrics, all_preds, all_labels
    return metrics


def plot_attention_comparison(attention_data, save_path=None):
    """
    attention_data: dict {model_name: attention_weights (L,L)}
    """
    fig, axes = plt.subplots(1, len(attention_data), figsize=(5*len(attention_data), 4))
    if len(attention_data) == 1:
        axes = [axes]
    for ax, (name, attn) in zip(axes, attention_data.items()):
        if attn.ndim == 3:
            attn = attn.mean(axis=0)  # average heads
        im = ax.imshow(attn[:200, :200], cmap='hot', aspect='auto')
        ax.set_title(f'{name}\nAttention (first 200bp)')
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved attention plot to {save_path}")
    plt.show()


def plot_distance_correlation(attention_data, max_dist=500, save_path=None):
    """Plot attention weight vs distance for each model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, attn in attention_data.items():
        if attn.ndim == 3:
            attn = attn.mean(axis=0)
        distances, attn_vals = compute_distance_correlation(attn, max_distance=max_dist)
        ax.plot(distances, attn_vals, label=name, linewidth=2)
    ax.set_xlabel('Distance (bp)')
    ax.set_ylabel('Average attention weight')
    ax.set_title('Attention decay with distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved distance correlation plot to {save_path}")
    plt.show()


def statistical_comparison(metrics_dict, metric_name='pearson'):
    """Compare best two models using paired bootstrap or t-test (requires per-sample predictions)."""
    # This is a placeholder; you need per-sample predictions to do proper paired test.
    # We'll implement assuming we have stored per-sample predictions.
    logger.warning("Statistical comparison requires per-sample predictions. Implement if needed.")
    return {}


def generate_latex_table(metrics_dict, save_path=None):
    """Generate LaTeX table from metrics."""
    rows = []
    for model_name, metrics in metrics_dict.items():
        row = [model_name]
        for m in ['mse', 'mae', 'pearson', 'spearman', 'r2']:
            row.append(f"{metrics.get(m, 0.0):.4f}")
        rows.append(row)
    headers = ['Model', 'MSE', 'MAE', 'Pearson', 'Spearman', 'R²']
    latex = tabulate(rows, headers=headers, tablefmt='latex_booktabs')
    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        logger.info(f"Saved LaTeX table to {save_path}")
    print(latex)
    return latex


def main():
    parser = argparse.ArgumentParser(description='Compare trained models on test set')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Test data directory')
    parser.add_argument('--cnn_checkpoint', type=str, default='checkpoints/cnn/best_model.pth')
    parser.add_argument('--transformer_checkpoint', type=str, default='checkpoints/transformer/best_model.pth')
    parser.add_argument('--nt_checkpoint', type=str, default='checkpoints/nt/best_model.pth')
    parser.add_argument('--nt_model_name', type=str, default='InstaDeepAI/nucleotide-transformer-v2-500m-multi-species')
    parser.add_argument('--seq_len', type=int, default=131072, help='Sequence length for CNN')
    parser.add_argument('--transformer_seq_len', type=int, default=10000, help='Max length for Transformer')
    parser.add_argument('--n_tracks', type=int, default=531)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--task_type', type=str, default='regression', choices=['regression', 'classification'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save outputs')
    parser.add_argument('--skip_nt', action='store_true', help='Skip NT (if not trained)')
    parser.add_argument('--skip_cnn', action='store_true')
    parser.add_argument('--skip_transformer', action='store_true')
    parser.add_argument('--attention_from_model', action='store_true', help='Extract attention from model (requires model modification)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load test dataset
    # For regression tasks, we use GenomicDataset (numpy) for CNN/Transformer and SequenceTextDataset for NT
    # But to keep consistent, we'll use GenomicDataset for all, and for NT we convert to strings on the fly.
    # For simplicity, we assume test data exists in both formats; otherwise we'll create one.
    # We'll implement dataloaders for each model separately.

    metrics_dict = {}
    all_predictions = {}

    # ----- CNN evaluation -----
    if not args.skip_cnn:
        logger.info("Evaluating CNN...")
        from train.train_cnn import GenomicDataset as NumpyDataset
        test_dataset_cnn = NumpyDataset(args.data_dir, 'test')
        test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=args.batch_size, shuffle=False, num_workers=2)
        cnn_model = load_cnn_model(args.cnn_checkpoint, args.seq_len, args.n_tracks, device)
        cnn_metrics, cnn_preds, cnn_labels = evaluate_model(
            cnn_model, test_loader_cnn, device, args.task_type, return_predictions=True
        )
        metrics_dict['CNN'] = cnn_metrics
        all_predictions['CNN'] = (cnn_preds, cnn_labels)
        logger.info(f"CNN metrics: {cnn_metrics}")

    # ----- Transformer evaluation -----
    if not args.skip_transformer:
        logger.info("Evaluating Transformer...")
        from train.train_transformer import GenomicDataset as NumpyDataset
        test_dataset_trans = NumpyDataset(args.data_dir, 'test')
        # Limit sequence length if needed
        test_loader_trans = DataLoader(test_dataset_trans, batch_size=args.batch_size, shuffle=False, num_workers=2)
        trans_model = load_transformer_model(
            args.transformer_checkpoint, args.transformer_seq_len, args.n_tracks,
            d_model=128, nhead=8, num_layers=6, device=device
        )
        trans_metrics, trans_preds, trans_labels = evaluate_model(
            trans_model, test_loader_trans, device, args.task_type, return_predictions=True
        )
        metrics_dict['Transformer'] = trans_metrics
        all_predictions['Transformer'] = (trans_preds, trans_labels)
        logger.info(f"Transformer metrics: {trans_metrics}")

    # ----- NT evaluation -----
    if not args.skip_nt and not args.skip_nt:
        logger.info("Evaluating Nucleotide Transformer...")
        test_dataset_nt = SequenceTextDataset(args.data_dir, 'test')
        test_loader_nt = DataLoader(test_dataset_nt, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=collate_fn, num_workers=2)
        nt_model = load_nt_model(args.nt_checkpoint, args.nt_model_name, args.n_tracks, device)
        nt_metrics, nt_preds, nt_labels = evaluate_model(
            nt_model, test_loader_nt, device, args.task_type, return_predictions=True
        )
        metrics_dict['NT'] = nt_metrics
        all_predictions['NT'] = (nt_preds, nt_labels)
        logger.info(f"NT metrics: {nt_metrics}")

    # ----- Generate comparison table -----
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    for name, metrics in metrics_dict.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Save results as JSON
    results_json = {
        model: metrics for model, metrics in metrics_dict.items()
    }
    with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Generate LaTeX table
    generate_latex_table(metrics_dict, save_path=os.path.join(args.output_dir, 'comparison_table.tex'))

    # ----- Attention analysis (if available) -----
    # For real attention extraction, you need to modify models to return attention weights.
    # We'll provide a placeholder: if attention files are provided as .npy, we can plot.
    # Otherwise, we skip.
    attention_files = {
        'CNN': os.path.join(args.output_dir, 'attn_cnn.npy'),
        'Transformer': os.path.join(args.output_dir, 'attn_transformer.npy'),
        'NT': os.path.join(args.output_dir, 'attn_nt.npy')
    }
    attention_data = {}
    for name, path in attention_files.items():
        if os.path.exists(path):
            attn = np.load(path)
            attention_data[name] = attn
            logger.info(f"Loaded attention for {name} from {path}")

    if attention_data:
        # Plot attention maps
        plot_attention_comparison(attention_data, save_path=os.path.join(args.output_dir, 'attention_maps.png'))
        # Plot distance correlation
        plot_distance_correlation(attention_data, max_dist=500, save_path=os.path.join(args.output_dir, 'attention_decay.png'))
    else:
        logger.info("No attention files found. Skipping attention visualization.")
        logger.info("To generate attention, modify model forward to return attention weights and save .npy.")

    # ----- Statistical significance (optional) -----
    # We can compute paired t-test between CNN and Transformer predictions if same test samples.
    if 'CNN' in all_predictions and 'Transformer' in all_predictions:
        cnn_preds, cnn_labels = all_predictions['CNN']
        trans_preds, trans_labels = all_predictions['Transformer']
        # Ensure same number of samples
        min_samples = min(len(cnn_preds), len(trans_preds))
        cnn_preds = cnn_preds[:min_samples]
        trans_preds = trans_preds[:min_samples]
        labels = cnn_labels[:min_samples].ravel()
        # Pearson correlation per sample? Better: compare metric per sample? Use paired t-test on MSE per sample
        cnn_mse_per_sample = np.mean((cnn_preds - labels) ** 2, axis=1)
        trans_mse_per_sample = np.mean((trans_preds - labels) ** 2, axis=1)
        t_stat, p_value = stats.ttest_rel(cnn_mse_per_sample, trans_mse_per_sample)
        logger.info(f"Paired t-test (CNN vs Transformer MSE per sample): t={t_stat:.4f}, p={p_value:.4f}")
        if p_value < 0.05:
            logger.info("Significant difference (p<0.05)")

    logger.info(f"All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()