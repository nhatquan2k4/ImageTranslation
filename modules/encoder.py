import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_layers=6, num_heads=8, hidden_dim=768, mlp_dim=3072, dropout=0.1, d_model=768):
        super().__init__()
        
        # Use pre-trained ViT for better initialization
        try:
            # Try to load pre-trained ViT
            self.vit = models.vit_b_16(weights='IMAGENET1K_V1')
            # FIX: Don't remove heads, we need the full feature extraction
            # Keep the encoder part and extract features from the last layer
            print("✅ Loaded pre-trained ViT-B/16 weights")
            vit_output_dim = 768
            self.use_pretrained = True
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
            vit_output_dim = hidden_dim
            self.use_pretrained = False
        
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        
        # Project từ ViT output dimension xuống d_model của transformer
        # FIX: Chỉ project nếu dimensions khác nhau
        if vit_output_dim != d_model:
            self.feature_projection = nn.Linear(vit_output_dim, d_model)
            print(f"🔧 Added projection layer: {vit_output_dim} -> {d_model}")
        else:
            self.feature_projection = nn.Identity()
            print(f"✅ No projection needed: {vit_output_dim} == {d_model}")
        
        # Text-aware attention để focus vào text regions trong ảnh
        self.text_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Positional encoding cho image patches
        # FIX: Tạo đủ lớn để handle various sequence lengths
        max_patches = (image_size // patch_size) ** 2 + 1  # +1 for CLS token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_patches, d_model)
        )
        self.max_patches = max_patches
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, 3, H, W] - Input images
        Returns:
            features: [batch_size, seq_len, d_model] - Encoded features
        """
        batch_size = x.size(0)
        
        # Extract features using ViT
        if self.use_pretrained:
            # For pre-trained ViT, extract features from encoder
            # We need to go through the encoder manually to get sequence output
            x = self.vit._process_input(x)  # Patch embedding + pos encoding
            n = x.shape[0]
            
            # Expand class token and add to sequence
            batch_class_token = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            
            # Pass through encoder layers
            vit_features = self.vit.encoder(x)  # [batch_size, seq_len+1, 768]
        else:
            # Custom ViT
            vit_features = self.vit(x)
        
        # Project to d_model dimensions
        # [batch_size, seq_len, d_model]
        projected_features = self.feature_projection(vit_features)
        
        # Add positional encoding
        seq_len = projected_features.size(1)
        
        # Dynamic positional embedding - handle different sequence lengths
        if seq_len <= self.max_patches:
            pos_emb = self.pos_embedding[:, :seq_len, :].expand(batch_size, -1, -1)
        else:
            # Interpolate if sequence is longer than expected
            pos_emb = torch.nn.functional.interpolate(
                self.pos_embedding.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2).expand(batch_size, -1, -1)
        
        projected_features = projected_features + pos_emb
        
        # Self-attention để model có thể focus vào text regions
        attended_features, _ = self.text_attention(
            projected_features, projected_features, projected_features
        )
        
        # Residual connection + normalization
        output_features = self.norm(projected_features + attended_features)
        
        return output_features