"""
Evaluation metrics for genomic sequence prediction models.

Includes:
    - Classification metrics: AUROC, AUPRC, F1, MCC
    - Regression metrics: Pearson, Spearman, R², MSE, MAE
    - Attention-based metrics: motif enrichment, attention signal-to-noise ratio
    - Long-range dependency metrics: receptive field analysis, distance correlation

All functions handle numpy arrays or torch tensors.
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple, Optional, Union
import warnings


# ============================================================
# Basic metrics
# ============================================================

def compute_auroc(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """
    Area Under the Receiver Operating Characteristic curve.
    
    Args:
        y_true: (N, T) binary labels (0/1)
        y_pred: (N, T) predicted probabilities
        average: 'macro' (average over tasks) or 'micro' or None (per-task list)
    Returns:
        AUROC score (or list if average is None)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_tasks = y_true.shape[1]
    scores = []
    for i in range(n_tasks):
        if len(np.unique(y_true[:, i])) < 2:
            continue  # skip if only one class
        try:
            scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        except ValueError:
            scores.append(np.nan)
    
    if average == 'macro':
        return float(np.nanmean(scores))
    elif average == 'micro':
        # flatten all tasks
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        return roc_auc_score(y_true_flat, y_pred_flat)
    else:
        return scores


def compute_auprc(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """Area Under Precision-Recall curve."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_tasks = y_true.shape[1]
    scores = []
    for i in range(n_tasks):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        scores.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    
    if average == 'macro':
        return float(np.nanmean(scores))
    elif average == 'micro':
        return average_precision_score(y_true.ravel(), y_pred.ravel())
    else:
        return scores


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray, average: bool = True) -> Union[float, List[float]]:
    """Pearson correlation coefficient per task."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_tasks = y_true.shape[1]
    scores = []
    for i in range(n_tasks):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() > 1:
            r, _ = pearsonr(y_true[mask, i], y_pred[mask, i])
            scores.append(r if not np.isnan(r) else 0.0)
        else:
            scores.append(0.0)
    
    if average:
        return float(np.mean(scores))
    return scores


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray, average: bool = True) -> Union[float, List[float]]:
    """Spearman rank correlation per task."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_tasks = y_true.shape[1]
    scores = []
    for i in range(n_tasks):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() > 1:
            r, _ = spearmanr(y_true[mask, i], y_pred[mask, i])
            scores.append(r if not np.isnan(r) else 0.0)
        else:
            scores.append(0.0)
    
    if average:
        return float(np.mean(scores))
    return scores


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray, average: bool = True) -> Union[float, List[float]]:
    """Coefficient of determination R² per task."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_tasks = y_true.shape[1]
    scores = []
    for i in range(n_tasks):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() > 1:
            r2 = r2_score(y_true[mask, i], y_pred[mask, i])
            scores.append(max(0.0, r2))  # clip negative to 0 for interpretability
        else:
            scores.append(0.0)
    
    if average:
        return float(np.mean(scores))
    return scores


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    return float(mean_squared_error(y_true.ravel(), y_pred.ravel()))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    return float(mean_absolute_error(y_true.ravel(), y_pred.ravel()))


# ============================================================
# Attention-based metrics
# ============================================================

def compute_attention_motif_enrichment(
    attention_weights: np.ndarray,  # (L, L) or (num_heads, L, L)
    motif_positions: List[Tuple[int, int]],  # list of (start, end) positions
    window_size: int = 10
) -> float:
    """
    Measure whether attention weights are enriched at known motif positions.
    
    For each motif position (start, end), we compute average attention weight from all tokens
    to positions within the motif. Then compare to average attention to random positions.
    
    Returns enrichment ratio: (motif_attention / random_attention) - 1.
    """
    if attention_weights.ndim == 3:
        # average over heads
        attention_weights = attention_weights.mean(axis=0)  # (L, L)
    
    L = attention_weights.shape[0]
    motif_attn = []
    random_attn = []
    
    for start, end in motif_positions:
        if start < 0 or end > L:
            continue
        # average attention from all query positions to the motif region
        attn_to_motif = attention_weights[:, start:end].mean()
        motif_attn.append(attn_to_motif)
        
        # random region of same length
        random_start = np.random.randint(0, L - (end - start))
        random_end = random_start + (end - start)
        attn_to_random = attention_weights[:, random_start:random_end].mean()
        random_attn.append(attn_to_random)
    
    if len(motif_attn) == 0:
        return 0.0
    
    mean_motif = np.mean(motif_attn)
    mean_random = np.mean(random_attn)
    if mean_random < 1e-8:
        return 0.0
    return (mean_motif / mean_random) - 1.0


def compute_attention_snr(attention_weights: np.ndarray) -> float:
    """
    Signal-to-noise ratio of attention distribution.
    High SNR means attention is sharply focused (useful for motif detection).
    """
    if attention_weights.ndim == 3:
        attention_weights = attention_weights.mean(axis=0)
    # Compute entropy of attention distribution per query, then average
    # Low entropy = focused. SNR = (1 - normalized entropy)
    L = attention_weights.shape[0]
    entropies = []
    for i in range(L):
        p = attention_weights[i, :] + 1e-8
        p = p / p.sum()
        entropy = -np.sum(p * np.log2(p))
        max_entropy = np.log2(L)
        entropies.append(1 - entropy / max_entropy)
    return float(np.mean(entropies))


# ============================================================
# Long-range dependency metrics
# ============================================================

def compute_distance_correlation(
    attention_weights: np.ndarray,  # (L, L)
    max_distance: Optional[int] = None
) -> np.ndarray:
    """
    Compute average attention weight as a function of distance between query and key.
    
    Returns:
        distances: array of distances (1..max_distance)
        avg_attention: average attention at each distance
    """
    if attention_weights.ndim == 3:
        attention_weights = attention_weights.mean(axis=0)
    
    L = attention_weights.shape[0]
    if max_distance is None:
        max_distance = L // 2
    
    distance_attn = []
    for d in range(1, max_distance + 1):
        # average over all pairs with Manhattan distance = d
        indices = np.array([(i, j) for i in range(L) for j in range(L) if abs(i - j) == d])
        if len(indices) == 0:
            attn = 0.0
        else:
            attn = np.mean([attention_weights[i, j] for i, j in indices])
        distance_attn.append(attn)
    
    return np.arange(1, max_distance + 1), np.array(distance_attn)


def compute_correlation_decay(
    model,
    dataloader,
    device: torch.device,
    max_distance: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measure how well model predictions correlate with true labels as a function of
    distance to a regulatory element. This requires a specific dataset with
    known regulatory element positions.
    
    Returns:
        distances: array of distances
        correlations: Pearson correlation at each distance bin
    """
    # This is a placeholder; implementation depends on dataset having annotations
    # We'll provide a skeleton; actual implementation requires custom dataset.
    raise NotImplementedError("Requires dataset with regulatory element positions.")
    # Example structure:
    # bins = np.linspace(0, max_distance, 20)
    # corrs = []
    # for dist_bin in bins:
    #     mask = (distances >= dist_bin) & (distances < dist_bin+step)
    #     corr = pearsonr(y_true[mask], y_pred[mask])[0]
    #     corrs.append(corr)
    # return bins, corrs


# ============================================================
# Comprehensive evaluation dictionary
# ============================================================

def evaluate_all(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    task_type: str = 'regression'  # 'regression' or 'classification'
) -> dict:
    """
    Compute a comprehensive set of metrics.
    
    Args:
        y_true: ground truth
        y_pred: predictions
        task_type: 'regression' or 'classification'
    
    Returns:
        dictionary of metric names -> values
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    metrics = {}
    
    if task_type == 'regression':
        metrics['mse'] = compute_mse(y_true, y_pred)
        metrics['mae'] = compute_mae(y_true, y_pred)
        metrics['pearson'] = compute_pearson(y_true, y_pred, average=True)
        metrics['spearman'] = compute_spearman(y_true, y_pred, average=True)
        metrics['r2'] = compute_r2(y_true, y_pred, average=True)
    elif task_type == 'classification':
        # Assumes y_true are binary (0/1) and y_pred are probabilities
        metrics['auroc'] = compute_auroc(y_true, y_pred, average='macro')
        metrics['auprc'] = compute_auprc(y_true, y_pred, average='macro')
        # Convert probabilities to binary at 0.5 for F1/MCC
        y_pred_binary = (y_pred > 0.5).astype(int)
        metrics['f1'] = f1_score(y_true.ravel(), y_pred_binary.ravel(), average='macro')
        metrics['mcc'] = matthews_corrcoef(y_true.ravel(), y_pred_binary.ravel())
    else:
        raise ValueError("task_type must be 'regression' or 'classification'")
    
    return metrics


# ============================================================
# Test the module
# ============================================================

if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    y_true_reg = np.random.randn(100, 10)
    y_pred_reg = y_true_reg + 0.1 * np.random.randn(100, 10)
    
    results = evaluate_all(y_true_reg, y_pred_reg, 'regression')
    print("Regression metrics:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    
    y_true_cls = np.random.randint(0, 2, (100, 10))
    y_pred_cls = np.random.rand(100, 10)
    results_cls = evaluate_all(y_true_cls, y_pred_cls, 'classification')
    print("\nClassification metrics:")
    for k, v in results_cls.items():
        print(f"  {k}: {v:.4f}")
    
    # Attention test
    attn = np.random.rand(50, 50)
    attn = attn / attn.sum(axis=1, keepdims=True)
    snr = compute_attention_snr(attn)
    print(f"\nAttention SNR: {snr:.4f}")
    
    distances, attn_by_dist = compute_distance_correlation(attn, max_distance=20)
    print(f"Distance correlation (first few): {attn_by_dist[:5]}")