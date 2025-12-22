"""
Model architectures for Image-to-Text Translation
"""
import math
import torch
import torch.nn as nn
import torchvision.models as models


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    Adds positional information to embeddings using sine and cosine functions.
    """
    
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class ResNetEncoder(nn.Module):
    """
    ResNet101-based visual encoder for extracting image features.
    Uses pretrained ResNet101 and removes the final FC layers.
    """
    
    def __init__(self, pretrained=True):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet101(pretrained=pretrained)
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(resnet.children())[:-2])
    
    def forward(self, x):
        """
        Args:
            x: Input images of shape (batch_size, 3, H, W)
        Returns:
            Feature maps of shape (batch_size, 2048, H', W')
        """
        return self.features(x)


class ResNetEncoderCustomDecoder(nn.Module):
    """
    Complete Image-to-Text model combining ResNet encoder with Transformer decoder.
    
    Architecture:
        1. ResNet101 encoder for visual features
        2. Adaptive pooling to reduce height to 1
        3. Linear projection to embedding dimension
        4. Transformer decoder for text generation
    """
    
    def __init__(self, encoder_dim, vocab_size, embed_dim, n_heads, n_layers, 
                 ffn_dim, dropout=0.1, pretrained=True):
        """
        Args:
            encoder_dim: Dimension of encoder features (2048 for ResNet101)
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension for decoder
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            ffn_dim: Dimension of feedforward network
            dropout: Dropout rate
            pretrained: Whether to use pretrained ResNet weights
        """
        super(ResNetEncoderCustomDecoder, self).__init__()
        
        # Visual encoder
        self.encoder = ResNetEncoder(pretrained=pretrained)
        
        # Adaptive pooling to reduce height dimension
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Project encoder features to embedding dimension
        self.enc_project = nn.Linear(encoder_dim, embed_dim)
        
        # Text embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask for autoregressive generation.
        
        Args:
            sz: Size of the mask (sequence length)
        Returns:
            Causal mask of shape (sz, sz)
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, images, captions):
        """
        Forward pass for training.
        
        Args:
            images: Input images of shape (batch_size, 3, H, W)
            captions: Target captions of shape (batch_size, seq_len)
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Encode images
        features = self.encoder(images)  # (B, C, H', W')
        features = self.adaptive_pool(features)  # (B, C, 1, W')
        
        B, C, _, W = features.shape
        memory = features.squeeze(2).permute(0, 2, 1)  # (B, W', C)
        memory = self.enc_project(memory)  # (B, W', embed_dim)
        
        # Generate masks
        tgt_mask = self.generate_square_subsequent_mask(captions.size(1)).to(captions.device)
        tgt_padding_mask = (captions == 0).to(captions.device)  # Assuming pad_token_id is 0
        
        # Embed captions
        tgt = self.embedding(captions) * math.sqrt(self.embed_dim)
        tgt = self.pos_encoder(tgt)
        
        # Decode
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Project to vocabulary
        output = self.fc_out(output)
        return output
    
    def freeze_encoder(self):
        """Freeze encoder parameters for gradual fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_encoder_parameters(self):
        """Get encoder parameters for separate optimization."""
        return self.encoder.parameters()
    
    def get_decoder_parameters(self):
        """Get decoder parameters (everything except encoder)."""
        decoder_params = []
        for name, param in self.named_parameters():
            if 'encoder.features' not in name:
                decoder_params.append(param)
        return decoder_params


def build_model(config, pretrained=True):
    """
    Build model from configuration.
    
    Args:
        config: Configuration object
        pretrained: Whether to use pretrained ResNet weights
    Returns:
        Model instance
    """
    model = ResNetEncoderCustomDecoder(
        encoder_dim=config.encoder_dim,
        vocab_size=None,  # Will be set after tokenizer is loaded
        embed_dim=config.embed_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        pretrained=pretrained
    )
    return model


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model):
    """Get detailed model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'encoder_params': sum(p.numel() for p in model.encoder.parameters()),
        'decoder_params': sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.encoder.parameters())
    }
    return info
