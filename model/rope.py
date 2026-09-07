"""Rotary Position Embedding (RoPE) —— 在 self-attention 中对 Q/K 旋转。

标准 RoPE 不在 embedding 层加位置向量，而是在 Q/K projection 之后、
attention score 计算之前，按 token 绝对位置对 head_dim 维度做旋转：

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

- inv_freq = base^(-2i/dim)，i ∈ [0, dim/2)
- cos/sin cache 按 (max_seq_len, dim) 预生成，形状 (T, 1, 1, dim) 广播到
  (B, H, T, head_dim)
- 支持 start 偏移：KV cache 增量推理时，新 token 的位置 = 缓存历史长度
  （历史 K 已按原位置旋转，新增位置独立旋转 → 因果语义保持）

与 sinusoidal（model/layers.py PositionalEncoding）互斥可选：
GPT(position_encoding='sinusoidal' | 'rope')，默认 sinusoidal（旧行为不变）。
"""
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """对最后一维做 half-split 旋转：(-x2, x1)。x (..., dim)，dim 需为偶数。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE 旋转器：持有 cos/sin cache，对 Q/K 应用旋转。

    用法（attention 内，Q/K projection 之后、score 之前）：
        rope = RotaryEmbedding(head_dim, max_seq_len)
        q, k = rope(q, k)                 # q/k (B, H, T, head_dim)
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f'RoPE 需要 head_dim 为偶数，得到 {dim}'
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # freqs (max_seq_len, dim/2)；重复两半 → (max_seq_len, dim)（配合 rotate_half）
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)          # (T, dim)
        # 形状 (1, 1, T, dim)：与 q/k (B, H, T, head_dim) 按位置维(T)广播
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                start: int = 0):
        """旋转 q/k（各自 (B, H, T, head_dim)）。start = 起始绝对位置（KV cache 偏移）。"""
        T = q.size(2)
        cos = self.cos_cached[:, :, start:start + T]   # 位置维在 dim2
        sin = self.sin_cached[:, :, start:start + T]
        return (q * cos + rotate_half(q) * sin,
                k * cos + rotate_half(k) * sin)
