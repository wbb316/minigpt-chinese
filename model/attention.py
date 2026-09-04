import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x, mask=None, past_kv=None, return_kv=False):
        """多头注意力（支持 KV cache 增量推理）。

        - x: (B, T, C)
        - past_kv: 之前算好的 (k, v)，各 (B, H, T_prev, head_dim)。给定时只算新
          token 的 q/k/v，与缓存拼接后算注意力（生成时不再重算历史）。
        - return_kv: 是否返回 (输出, 新kv)。past_kv 给定时也会返回 kv。

        训练/常规路径（无缓存、无自定义 mask）用 SDPA(flash/mem-efficient)：
        不物化 (B,H,T,T) 注意力矩阵 —— block 256+ 时显存省 ~50%，速度也更快。
        推理缓存路径与自定义 mask 路径保留手写实现。

        返回:
          - 默认: 输出 (B, T, C)
          - past_kv 或 return_kv 给定时: (输出, (k, v))，k/v 形状 (B, H, T_full, head_dim)
        """
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is None and mask is None:
            # ---------- SDPA 路径（训练/常规, 全因果） ----------
            k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True)                     # 自动选 flash / mem-efficient
            out = self.out_proj(y.transpose(1, 2).contiguous().view(B, T, C))
            if return_kv:
                return out, (k, v)
            return out

        # ---------- 手写路径（KV cache 增量 或 自定义 mask） ----------
        if past_kv is not None:
            k_prev, v_prev = past_kv
            k_new = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v_new = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = torch.cat([k_prev, k_new], dim=2)   # (B, H, T_prev+T, head_dim)
            v = torch.cat([v_prev, v_new], dim=2)
            T_full = k.size(2)
            if mask is None:
                # 新 token 可看全部历史 + 自己（含同批内新 token 的因果关系）
                mask = torch.tril(torch.ones(T_full, T_full, device=x.device)).bool()
                mask = mask[T_full - T:, :]         # 只取新 token 对应的查询行
        else:
            k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            T_full = T
            if mask is None:
                mask = torch.tril(torch.ones(T, T, device=x.device)).bool()

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, T, T_full)
        attn = attn.masked_fill(~mask[None, None, :T, :T_full], float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(y)

        if past_kv is not None or return_kv:
            return out, (k, v)
        return out
