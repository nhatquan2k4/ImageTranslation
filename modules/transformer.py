import torch
import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, trg_vocab, d_model=512, n=6, heads=8, dropout=0.1, image_size=384, vision_hidden_dim=768, vision_layers=6, vision_heads=8, patch_size=16):
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
        
        # FIX: Vision feature projection để đảm bảo cùng dimension
        self.vision_projection = nn.Linear(vision_hidden_dim, d_model)
        
        # Cross-attention layers - sử dụng toàn bộ, không skip
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
            for _ in range(max(1, n//2))
        ])
        
        # Layer normalization cho cross-attention
        self.cross_norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(max(1, n//2))
        ])
        
        # Vision feature fusion
        self.vision_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out = nn.Linear(d_model, trg_vocab)
        
        # Store dimensions
        self.d_model = d_model
        self.n_layers = n
        
    def forward(self, src_img, trg, trg_mask):
        # Encode image để extract text features
        vision_features = self.vision_encoder(src_img)  # [B, V_seq, vision_hidden_dim]
        
        # FIX: Project vision features to match d_model dimension
        vision_features = self.vision_projection(vision_features)  # [B, V_seq, d_model]
        
        # Decode Vietnamese text
        decoder_out = self.text_decoder(trg, vision_features, None, trg_mask)  # [B, T_seq, d_model]
        
        # Apply cross-attention với proper dimension handling
        for i in range(len(self.cross_attention_layers)):
            # FIX: Đảm bảo dimensions match trước khi cross-attention
            assert decoder_out.size(-1) == self.d_model, f"Decoder dimension mismatch: {decoder_out.size(-1)} vs {self.d_model}"
            assert vision_features.size(-1) == self.d_model, f"Vision dimension mismatch: {vision_features.size(-1)} vs {self.d_model}"
            
            # Cross-attention: decoder output attend to vision features
            attended_out, attn_weights = self.cross_attention_layers[i](
                decoder_out, vision_features, vision_features,
                key_padding_mask=None
            )
            
            # Gated fusion of text and vision information
            gate_input = torch.cat([decoder_out, attended_out], dim=-1)
            gate = self.vision_gate(gate_input)
            fused_out = gate * attended_out + (1 - gate) * decoder_out
            
            # FIX: Residual connection + Layer norm
            decoder_out = self.cross_norm_layers[i](fused_out + decoder_out)
        
        # Project to vocabulary - dimension đã đảm bảo match
        output = self.out(decoder_out)  # [B, T_seq, vocab_size]
        return output


# FIX: Tạo class riêng cho ImageToTextTransformer
class ImageToTextTransformer(nn.Module):
    """
    Wrapper class for Image-to-Text Translation
    Handles proper initialization and parameter management
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, 
                 image_size=224, patch_size=16, max_seq_length=256):
        super().__init__()
        
        # Calculate vision parameters
        vision_hidden_dim = 768  # ViT-B/16 standard
        
        self.transformer = Transformer(
            trg_vocab=vocab_size,
            d_model=d_model,
            n=num_decoder_layers,
            heads=num_heads,
            dropout=dropout,
            image_size=image_size,
            vision_hidden_dim=vision_hidden_dim,
            vision_layers=num_encoder_layers,
            vision_heads=num_heads,
            patch_size=patch_size
        )
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
    def forward(self, images, text_input, text_target=None):
        """
        Forward pass for training and inference
        Args:
            images: [B, 3, H, W] 
            text_input: [B, seq_len] - input sequence
            text_target: [B, seq_len] - target sequence (for training)
        """
        batch_size, seq_len = text_input.shape
        
        # Create causal mask for decoder
        trg_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(text_input.device)
        
        # Forward through transformer
        output = self.transformer(images, text_input, trg_mask)
        
        return output