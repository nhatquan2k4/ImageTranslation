import torch
import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, trg_vocab, d_model=512, n=6, heads=8, dropout=0.1, image_size=224, vision_hidden_dim=768, vision_layers=6, vision_heads=8, patch_size=16):
        super().__init__()
        
        # Vision encoder để extract features từ ảnh chứa text
        self.vision_encoder = Encoder(
            image_size=image_size, 
            patch_size=patch_size,
            num_layers=vision_layers,
            num_heads=vision_heads,
            hidden_dim=vision_hidden_dim,
            d_model=d_model,
            dropout=dropout
        )
        
        # Text decoder để generate Vietnamese text
        self.text_decoder = Decoder(trg_vocab, d_model, n, heads, dropout)
        
        # Cross-attention layers để connect vision và text
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
            for _ in range(n)
        ])
        
        # Layer normalization cho cross-attention
        self.cross_norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n)
        ])
        
        # Output projection
        self.out = nn.Linear(d_model, trg_vocab)
        
        # Store dimensions
        self.d_model = d_model
        self.n_layers = n
        
    def forward(self, src_img, trg, trg_mask):
        # Encode image để extract text features
        # src_img: [batch_size, 3, H, W] -> vision_features: [batch_size, seq_len, d_model]
        vision_features = self.vision_encoder(src_img)
        
        # Decode Vietnamese text
        # trg: [batch_size, seq_len] -> decoder_out: [batch_size, seq_len, d_model]  
        decoder_out = self.text_decoder(trg, vision_features, None, trg_mask)
        
        # Apply cross-attention giữa text decoder và vision features
        for i in range(self.n_layers):
            # Cross-attention: decoder output attend to vision features
            attended_out, _ = self.cross_attention_layers[i](
                decoder_out, vision_features, vision_features,
                key_padding_mask=None
            )
            # Residual connection + layer norm
            decoder_out = self.cross_norm_layers[i](decoder_out + attended_out)
        
        # Project to vocabulary
        output = self.out(decoder_out)
        return output