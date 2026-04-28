"""
Training script for Basenji2-like CNN model on genomic sequence data.

Usage:
    python train/train_cnn.py --config config.yaml
    python train/train_cnn.py --data_dir data/processed --epochs 50 --batch_size 32 --lr 1e-3
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
from models.cnn_basenji2 import Basenji2CNN

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_cnn.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GenomicDataset(Dataset):
    """
    PyTorch Dataset for genomic sequences and labels.
    
    Assumes:
        sequences.npy: (N, seq_len) of int8 (0,1,2,3)
        labels.npy: (N, n_tracks) of float32
    """
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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for seq, label in progress:
        seq, label = seq.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
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
    
    # Compute additional metrics (e.g., Pearson correlation per track)
    # For simplicity, we return loss and correlations
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
    parser = argparse.ArgumentParser(description='Train CNN model for genomic prediction')
    parser.add_argument('--config', type=str, default=None, help='YAML config file path')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory with train/val/test .npy files')
    parser.add_argument('--seq_len', type=int, default=131072, help='Sequence length (default: 131072)')
    parser.add_argument('--n_tracks', type=int, default=531, help='Number of output tracks')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume from')
    parser.add_argument('--save_dir', type=str, default='checkpoints/cnn', help='Directory to save checkpoints')
    args = parser.parse_args()
    
    # Override with config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            cnn_cfg = config.get('cnn', {})
            for key, val in cnn_cfg.items():
                if hasattr(args, key):
                    setattr(args, key, val)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Data
    train_dataset = GenomicDataset(args.data_dir, 'train')
    val_dataset = GenomicDataset(args.data_dir, 'val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = Basenji2CNN(seq_len=args.seq_len, n_tracks=args.n_tracks, n_channels=64)
    model = model.to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.MSELoss()  # Regression task for expression values
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
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