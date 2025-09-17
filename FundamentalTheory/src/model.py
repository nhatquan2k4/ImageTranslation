# model.py
import math
import torch
import torch.nn as nn
import timm
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class ViTEncoder(nn.Module):
    def __init__(self, model_name: str = 'google/vit-base-patch16-224', proj_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.num_features
        self.proj_dim = proj_dim  # Lưu proj_dim làm thuộc tính
        self.proj = nn.Linear(self.embed_dim, proj_dim) if self.embed_dim != proj_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        if hasattr(self.vit, 'forward_features'):
            feats = self.vit.forward_features(x)  # (B, S, D)
        else:
            feats = self.vit(x)
        if feats.dim() == 2:  # Global avg pool case
            feats = feats.unsqueeze(1)  # (B, 1, D)
        return self.proj(feats)  # (B, S, proj_dim)


class TransformerDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_len: int = 128,
        pad_idx: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # tgt_tokens: (B, T)
        B, T = tgt_tokens.shape
        tgt = tgt_tokens.transpose(0, 1)  # (T, B)
        emb = self.embedding(tgt) * math.sqrt(self.d_model)  # (T, B, D)
        emb = self.pos_enc(emb.transpose(0, 1)).transpose(0, 1)  # (T, B, D) with pos_enc on (B, T, D)
        
        # Memory: Ensure (S, B, D)
        if memory.dim() == 3 and memory.size(0) == B:
            memory = memory.transpose(0, 1)  # (S, B, D) if input was (B, S, D)
        
        out = self.decoder(
            tgt=emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # (T, B, D)
        logits = self.output_proj(out)  # (T, B, V)
        return logits


class Image2TextModel(nn.Module):
    def __init__(self, encoder: ViTEncoder, decoder: TransformerDecoderModel):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.proj_dim == decoder.d_model, "proj_dim must match d_model"

    def forward(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        memory = self.encoder(images)  # (B, S, D)
        logits = self.decoder(tgt_tokens, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return logits