import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, image_size=384, patch_size=16, num_layers=6, num_heads=8, hidden_dim=768, mlp_dim=3072, dropout=0.1, d_model=512):
        super().__init__()
        
        # Use pre-trained ViT for better initialization
        try:
            # Try to load pre-trained ViT
            self.vit = models.vit_b_16(weights='IMAGENET1K_V1')
            # Remove classification head
            self.vit.heads = nn.Identity()
            print("✅ Loaded pre-trained ViT-B/16 weights")
        except:
            # Fallback to custom ViT
            self.vit = VisionTransformer(
                image_size=image_size,
                patch_size=patch_size,
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
                num_classes=0
            )
            print("⚠️ Using random initialized ViT")
        
        self.hidden_dim = hidden_dim
        
        # Project từ ViT hidden_dim xuống d_model của transformer
        self.feature_projection = nn.Linear(hidden_dim, d_model)
        
        # Text-aware attention để focus vào text regions trong ảnh
        self.text_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Positional encoding cho image patches
        self.pos_embedding = nn.Parameter(
            torch.randn(1, (image_size // patch_size) ** 2 + 1, d_model)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, 3, H, W] - Input images
        Returns:
            features: [batch_size, seq_len, d_model] - Encoded features
        """
        batch_size = x.size(0)
        
        # Extract features using ViT
        # Output: [batch_size, seq_len, hidden_dim] where seq_len = num_patches + 1 (cls token)
        vit_features = self.vit(x)
        
        # Project to d_model dimensions
        # [batch_size, seq_len, d_model]
        projected_features = self.feature_projection(vit_features)
        
        # Add positional encoding
        seq_len = projected_features.size(1)
        pos_emb = self.pos_embedding[:, :seq_len, :].expand(batch_size, -1, -1)
        projected_features = projected_features + pos_emb
        
        # Self-attention để model có thể focus vào text regions
        attended_features, _ = self.text_attention(
            projected_features, projected_features, projected_features
        )
        
        # Residual connection + normalization
        output_features = self.norm(projected_features + attended_features)
        
        return output_features