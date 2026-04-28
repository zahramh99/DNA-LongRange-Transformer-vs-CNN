"""
Fine-tuning Nucleotide Transformer (pretrained) for downstream genomic tasks.
Nucleotide Transformer: https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict, Any

class NucleotideTransformerFineTuner(nn.Module):
    """
    Wrapper for Nucleotide Transformer with a custom head for fine-tuning.
    
    Args:
        model_name: HuggingFace model name (default: 500M multi-species)
        n_tracks: number of output tasks (regression or classification)
        dropout: dropout rate for the head
        freeze_backbone: whether to freeze the pretrained transformer
    """
    
    def __init__(
        self,
        model_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        n_tracks: int = 531,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        hidden_size = self.backbone.config.hidden_size  # usually 1024 for 500M model
        
        # Optional: freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Prediction head: global pooling + MLP
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
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
        # Tokenize: Nucleotide Transformer uses K-mer tokenization (6-mer by default)
        # For simplicity we use the tokenizer directly
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.backbone.config.max_position_embeddings
        )
        # Move to same device as model
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward through backbone
        outputs = self.backbone(**inputs)
        # outputs.last_hidden_state: (B, L, hidden_size)
        # Mean pooling over sequence length (ignore padding)
        attention_mask = inputs["attention_mask"]  # (B, L)
        # Expand mask to hidden dim
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        pooled = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # Head
        pooled = self.dropout(pooled)
        out = self.head(pooled)
        return out
    
    def freeze_backbone(self):
        """Freeze the pretrained transformer, train only head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze the whole model for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# Small test (requires internet to download model)
if __name__ == "__main__":
    # Uncomment to test (downloads ~2GB)
    # model = NucleotideTransformerFineTuner(n_tracks=10, freeze_backbone=True)
    # dummy_seqs = ["ACGT" * 250, "TGCA" * 250]  # 1000 bp each
    # out = model(dummy_seqs)
    # print(f"Fine-tuning output shape: {out.shape}")
    print("Model definition ready. To test, uncomment and run with internet.")