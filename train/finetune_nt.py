"""
Fine-tuning script for Nucleotide Transformer (pretrained) on genomic sequence prediction.

Requirements:
    - HuggingFace transformers
    - GPU with at least 16GB memory (or use gradient accumulation + mixed precision)

Usage:
    python train/finetune_nt.py --config config.yaml
    python train/finetune_nt.py --data_dir data/processed --epochs 10 --batch_size 4 --lr 1e-5 --freeze_backbone
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
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_nt.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class NucleotideTransformerFineTuner(nn.Module):
    """
    Wrapper for Nucleotide Transformer with a custom regression head.
    """
    def __init__(
        self,
        model_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        n_tracks: int = 531,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        logger.info(f"Loading pretrained model: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        hidden_size = self.backbone.config.hidden_size  # 1024 for 500M model
        logger.info(f"Hidden size: {hidden_size}")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")
        
        # Prediction head: MLP with LayerNorm
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_tracks)
        )
        
    def forward(self, sequences: list) -> torch.Tensor:
        """
        Args:
            sequences: List of DNA strings (e.g., ["ACGT...", ...])
        Returns:
            predictions: (batch_size, n_tracks)
        """
        # Tokenize: Nucleotide Transformer expects k-mer tokenization (default 6-mer)
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.backbone.config.max_position_embeddings
        )
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.backbone(**inputs)
        # outputs.last_hidden_state: (B, L, hidden_size)
        attention_mask = inputs["attention_mask"]  # (B, L)
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        pooled = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        out = self.head(pooled)
        return out
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")


class SequenceTextDataset(Dataset):
    """
    Dataset for DNA sequences as strings and labels.
    Assumes:
        sequences.txt: one sequence per line (or .npy of strings)
        labels.npy: (N, n_tracks) float32
    Alternatively, can read from numpy arrays.
    """
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Try to load sequences as strings from .txt or .npy
        txt_path = self.data_dir / f"{split}_sequences.txt"
        npy_seq_path = self.data_dir / f"{split}_sequences_str.npy"
        label_path = self.data_dir / f"{split}_labels.npy"
        
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                self.sequences = [line.strip() for line in f.readlines()]
        elif npy_seq_path.exists():
            self.sequences = np.load(npy_seq_path, allow_pickle=True).tolist()
        else:
            raise FileNotFoundError(f"No sequence file for split {split} in {data_dir}")
        
        self.labels = np.load(label_path)
        assert len(self.sequences) == len(self.labels), "Mismatch between sequences and labels"
        logger.info(f"Loaded {split} set: {len(self.sequences)} sequences, labels shape {self.labels.shape}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label


def collate_fn(batch):
    """Custom collate to keep sequences as list of strings."""
    sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return sequences, labels


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, accumulation_steps=4, use_amp=True):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    progress = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for i, (sequences, labels) in enumerate(progress):
        labels = labels.to(device)
        
        # Forward with mixed precision if available
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        progress.set_postfix(loss=loss.item() * accumulation_steps)
    
    # Handle remaining gradients
    if len(dataloader) % accumulation_steps != 0:
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, metric_prefix="Val"):
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc=f"{metric_prefix}"):
            labels = labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Pearson correlation per track
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
    parser = argparse.ArgumentParser(description='Fine-tune Nucleotide Transformer for genomic prediction')
    parser.add_argument('--config', type=str, default=None, help='YAML config file path')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory with train/val sequences.txt and labels.npy')
    parser.add_argument('--model_name', type=str, default='InstaDeepAI/nucleotide-transformer-v2-500m-multi-species', help='Pretrained model name')
    parser.add_argument('--n_tracks', type=int, default=531, help='Number of output tracks')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (small due to large model)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze pretrained backbone initially')
    parser.add_argument('--unfreeze_after_epochs', type=int, default=None, help='Unfreeze backbone after N epochs (if None, stays frozen)')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use automatic mixed precision')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume from')
    parser.add_argument('--save_dir', type=str, default='checkpoints/nt', help='Directory to save checkpoints')
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            nt_cfg = config.get('nucleotide_transformer', {})
            for key, val in nt_cfg.items():
                if hasattr(args, key):
                    setattr(args, key, val)
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    
    # Data
    train_dataset = SequenceTextDataset(args.data_dir, 'train')
    val_dataset = SequenceTextDataset(args.data_dir, 'val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    
    # Model
    model = NucleotideTransformerFineTuner(
        model_name=args.model_name,
        n_tracks=args.n_tracks,
        dropout=0.1,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)
    
    logger.info("Starting fine-tuning...")
    for epoch in range(start_epoch, args.epochs):
        # Optionally unfreeze backbone after certain epoch
        if args.unfreeze_after_epochs is not None and epoch == args.unfreeze_after_epochs:
            model.unfreeze_backbone()
            # Reinitialize optimizer with all parameters
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)  # lower LR when unfreezing
            logger.info(f"Unfroze backbone at epoch {epoch}, reset optimizer with LR={args.lr*0.1}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.accumulation_steps, args.use_amp)
        val_loss, val_corr = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Corr={val_corr:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
        
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            periodic_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, periodic_path)
    
    logger.info("Fine-tuning completed.")


if __name__ == "__main__":
    main()