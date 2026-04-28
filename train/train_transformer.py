"""
Training script for small Transformer model on genomic sequence data.

Usage:
    python train/train_transformer.py --config config.yaml
    python train/train_transformer.py --data_dir data/processed --epochs 50 --batch_size 16 --lr 1e-3 --max_len 10000
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.transformer_small import TransformerSmall

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_transformer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GenomicDataset(Dataset):
    """Same dataset as for CNN: loads sequences and labels from numpy files."""
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        seq_path = self.data_dir / f"{split}_sequences.npy"
        label_path = self.data_dir / f"{split}_labels.npy"
        
        if not seq_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing data files for split {split} in {data_dir}")
        
        self.sequences = np.load(seq_path)
        self.labels = np.load(label_path)
        assert len(self.sequences) == len(self.labels), "Mismatch between sequences and labels"
        logger.info(f"Loaded {split} set: {len(self.sequences)} samples, "
                    f"seq_len={self.sequences.shape[1]}, n_tracks={self.labels.shape[1]}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, accumulation_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    progress = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    
    for i, (seq, label) in enumerate(progress):
        seq, label = seq.to(device), label.to(device)
        
        # Forward pass
        output = model(seq)
        loss = criterion(output, label)
        loss = loss / accumulation_steps  # Normalize for gradient accumulation
        loss.backward()
        
        # Update weights after accumulation_steps batches
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # Recover original loss
        progress.set_postfix(loss=loss.item() * accumulation_steps)
    
    # Handle remaining gradients if not exactly divisible
    if (len(dataloader) % accumulation_steps) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, metric_prefix="Val"):
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for seq, label in tqdm(dataloader, desc=f"{metric_prefix}"):
            seq, label = seq.to(device), label.to(device)
            output = model(seq)
            loss = criterion(output, label)
            total_loss += loss.item()
            all_outputs.append(output.cpu())
            all_labels.append(label.cpu())
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute Pearson correlation per track
    correlations = []
    for i in range(all_labels.shape[1]):
        mask = ~torch.isnan(all_labels[:, i])
        if mask.sum() > 1:
            corr = torch.corrcoef(torch.stack([all_outputs[mask, i], all_labels[mask, i]]))[0, 1]
            correlations.append(corr if not torch.isnan(corr) else 0.0)
        else:
            correlations.append(0.0)
    mean_corr = np.mean(correlations)
    
    return total_loss / len(dataloader), mean_corr


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}, loss={loss:.4f}")
    return start_epoch


def main():
    parser = argparse.ArgumentParser(description='Train small Transformer model for genomic prediction')
    parser.add_argument('--config', type=str, default=None, help='YAML config file path')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory with train/val/test .npy files')
    parser.add_argument('--seq_len', type=int, default=10000, help='Sequence length (max_len for transformer)')
    parser.add_argument('--n_tracks', type=int, default=531, help='Number of output tracks')
    parser.add_argument('--d_model', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (adjust for memory)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation steps (effective batch size = batch_size * accumulation_steps)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume from')
    parser.add_argument('--save_dir', type=str, default='checkpoints/transformer', help='Directory to save checkpoints')
    args = parser.parse_args()
    
    # Override with config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            trans_cfg = config.get('transformer_small', {})
            for key, val in trans_cfg.items():
                if hasattr(args, key):
                    setattr(args, key, val)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    
    # Data
    train_dataset = GenomicDataset(args.data_dir, 'train')
    val_dataset = GenomicDataset(args.data_dir, 'val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = TransformerSmall(
        vocab_size=4,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_len=args.seq_len,
        n_tracks=args.n_tracks,
        dropout=0.1
    )
    model = model.to(device)
    logger.info(f"Model has {model.get_num_params():,} parameters")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.accumulation_steps)
        val_loss, val_corr = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Corr={val_corr:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
        
        # Save periodic checkpoint
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            periodic_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, periodic_path)
    
    logger.info("Training completed.")


if __name__ == "__main__":
    main()