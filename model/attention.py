import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn = attn.masked_fill(~mask[None, None, :T, :T], float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)
