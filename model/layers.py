import math
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # 标准 LayerNorm：有偏方差 (除以 N) + sqrt(var + eps)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class PositionalEncoding(nn.Module):
    def __init__(self, dim:int, max_len:int =1024):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        dim_term=torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * dim_term)
        pe[:, 1::2] = torch.cos(position * dim_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, start: int = 0):
        """start: 起始位置偏移（KV cache 增量解码时，新 token 的全局位置 = 缓存长度）。"""
        return x + self.pe[:, start:start + x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, dim:int, dropout: float = 0.0):
        super().__init__()
        self.fc1=nn.Linear(dim, 4*dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2=nn.Linear(4*dim, dim)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x=self.fc1(x)
        x=torch.relu(x)
        x=self.dropout(x)
        return self.fc2(x)
