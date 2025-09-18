# model.py
import math
import torch
import torch.nn as nn
import timm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))


    def forward(self, x):
        # x: (B, T, D)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class ViTEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', proj_dim=512, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.num_features
        if self.embed_dim != proj_dim:
            self.proj = nn.Linear(self.embed_dim, proj_dim)
        else:
            self.proj = nn.Identity()


    def forward(self, x):
        # attempt to get patch sequence
        if hasattr(self.vit, 'forward_features'):
            feats = self.vit.forward_features(x) # often (B, S, D)
        else:
            feats = self.vit(x)
        if feats.dim() == 2:
            feats = feats.unsqueeze(1)
        return self.proj(feats)


class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_len=128, pad_idx=3):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout=0.1)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)


    def forward(self, tgt_tokens, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt_tokens: (B, T)
        tgt = tgt_tokens.transpose(0,1) # (T, B)
        emb = self.embedding(tgt) * math.sqrt(self.d_model) # (T, B, D)
        emb = emb.transpose(0,1) # (B, T, D)
        emb = self.pos_enc(emb)
        emb = emb.transpose(0,1) # (T,B,D)
        if memory.dim() == 3:
            memory = memory.transpose(0,1) # (S, B, D)
        out = self.decoder(tgt=emb, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.output_proj(out)
        return logits


class Image2TextModel(nn.Module):
    def __init__(self, encoder: ViTEncoder, decoder: TransformerDecoderModel):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, images, tgt_tokens, tgt_mask=None, tgt_key_padding_mask=None):
        memory = self.encoder(images) # (B,S,D)
        logits = self.decoder(tgt_tokens, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return logits