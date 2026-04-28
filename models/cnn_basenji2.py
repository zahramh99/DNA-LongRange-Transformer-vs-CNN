"""
CNN model inspired by Basenji2 architecture for genomic sequence prediction.

Reference:
    Kelley et al. "Cross-species regulatory sequence activity prediction."
    PLoS Computational Biology, 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class DilatedConvBlock(nn.Module):
    """Single dilated convolutional block with residual connection."""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection if dimensions change
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class Basenji2CNN(nn.Module):
    """
    CNN model for genomic sequence prediction with long-range receptive field.
    
    Args:
        seq_len: Input sequence length (number of nucleotides)
        n_tracks: Number of output prediction tracks (e.g., epigenetic marks)
        n_channels: Base number of convolutional channels (scales with depth)
    """
    
    def __init__(self, seq_len: int = 131072, n_tracks: int = 531, n_channels: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.n_tracks = n_tracks
        
        # Initial convolution: one-hot encoding (4 nucleotides) -> n_channels
        self.initial_conv = nn.Sequential(
            nn.Conv1d(4, n_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dilated convolutional blocks with increasing dilations
        # Each block doubles dilations roughly: 1,2,4,8,16,32,64
        dilations = [1, 2, 4, 8, 16, 32, 64]
        self.blocks = nn.ModuleList()
        current_channels = n_channels
        for i, d in enumerate(dilations):
            out_ch = n_channels * (2 if i < 3 else 1)  # Expand early layers
            self.blocks.append(DilatedConvBlock(current_channels, out_ch, dilation=d))
            current_channels = out_ch
        
        # Final tower: average pooling over time and dense output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_fc = nn.Linear(current_channels, n_tracks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) or (batch_size, 4, seq_len) - nucleotide indices or one-hot
        Returns:
            predictions: (batch_size, n_tracks)
        """
        # If input is indices (long), convert to one-hot
        if x.dtype == torch.long:
            x = F.one_hot(x, num_classes=4).float().permute(0, 2, 1)  # (B, 4, L)
        elif x.dim() == 3 and x.shape[1] != 4:
            # Assume (B, L, 4) -> permute
            x = x.permute(0, 2, 1)
        
        out = self.initial_conv(x)
        for block in self.blocks:
            out = block(out)
        out = self.global_pool(out).squeeze(-1)  # (B, C)
        predictions = self.final_fc(out)
        return predictions
    
    def get_receptive_field(self) -> int:
        """Return theoretical receptive field length in bp."""
        # Simplified: each dilated conv of kernel 3 adds 2*dilation
        dilations = [1, 2, 4, 8, 16, 32, 64]
        rf = 15  # initial conv
        for d in dilations:
            rf += 2 * d * (3 - 1)  # kernel=3 => dilation step adds 2*dilation
        return rf


# Small test
if __name__ == "__main__":
    model = Basenji2CNN(seq_len=1000, n_tracks=10, n_channels=32)
    dummy = torch.randint(0, 4, (2, 1000))
    out = model(dummy)
    print(f"CNN output shape: {out.shape}")  # Expected (2, 10)
    print(f"Receptive field: {model.get_receptive_field()} bp")