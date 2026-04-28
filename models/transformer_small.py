"""
A small Transformer encoder for DNA sequences, trained from scratch.
Architecture similar to DNABERT but smaller for fair comparison with CNN.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """Learnable positional encoding for DNA sequences."""
    def __init__(self, d_model: int, max_len: int = 131072):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerSmall(nn.Module):
    """
    Small Transformer encoder for genomic sequence prediction.
    
    Args:
        vocab_size: number of nucleotides (4)
        d_model: embedding dimension
        nhead: number of attention heads
        num_layers: number of transformer encoder layers
        max_len: maximum sequence length
        n_tracks: number of output tasks
        dropout: dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        max_len: int = 10000,
        n_tracks: int = 531,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling and output head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(d_model, n_tracks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) with long indices (0..3)
        Returns:
            predictions: (batch_size, n_tracks)
        """
        # Embedding + positional encoding
        x = self.embedding(x)  # (B, L, d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transformer encoder
        x = self.transformer(x)  # (B, L, d_model)
        
        # Global average pooling over sequence length
        x = x.transpose(1, 2)  # (B, d_model, L)
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        
        # Output layer
        out = self.output(x)  # (B, n_tracks)
        return out
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TransformerSmall(n_tracks=10, max_len=1000)
    dummy = torch.randint(0, 4, (2, 1000))
    out = model(dummy)
    print(f"Transformer output shape: {out.shape}")
    print(f"Number of parameters: {model.get_num_params():,}")