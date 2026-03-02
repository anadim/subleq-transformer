"""
Pre-LN Transformer for learning SUBLEQ execution from data.

Architecture:
    - Encoder-only (bidirectional attention, no causal mask)
    - Default: d_model=256, n_heads=8, n_layers=6, d_ff=1024 (4.9M params)
    - Pre-LayerNorm, GELU activation
    - Token + Position + Type embeddings
    - Output: Linear(d_model -> vocab_size) per position
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import SEQ_LEN, VOCAB_SIZE


class MiniSUBLEQTransformer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=4, d_ff=256,
                 vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.type_emb = nn.Embedding(2, d_model)  # 0=PC position, 1=memory position
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output head
        self.output_head = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

        # Register position indices and type indices
        self.register_buffer('pos_indices',
                             torch.arange(seq_len).unsqueeze(0))
        type_ids = torch.zeros(seq_len, dtype=torch.long)
        type_ids[0] = 0  # PC
        type_ids[1:] = 1  # memory
        self.register_buffer('type_indices', type_ids.unsqueeze(0))

    def _init_weights(self):
        """Initialize with small weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, S = x.shape

        # Embeddings
        tok = self.token_emb(x)
        pos = self.pos_emb(self.pos_indices[:, :S].expand(B, -1))
        typ = self.type_emb(self.type_indices[:, :S].expand(B, -1))
        h = self.emb_dropout(tok + pos + typ)

        # Transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Final norm + output
        h = self.final_norm(h)
        logits = self.output_head(h)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class TransformerBlock(nn.Module):
    """Pre-LN transformer block with multi-head self-attention + FFN."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))
