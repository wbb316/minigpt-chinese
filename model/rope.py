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

⚠️⚠️ 静默失效警告（本仓库踩过一次，改这里之前务必读完）：
    forward 里的 `cos_cached[:, :, start:start + T]` 是**普通切片**，位置索引
    一旦越界（start + T > 表长）**不会报错**，而是：
      · T == 1（增量解码恰好就是 T=1）→ 切片长度 0，`q * cos` 靠广播把结果变成
        空张量，随后整条链路「位置编码被完全跳过」——
        实测输出与**未旋转的输入逐位相同**，无异常、无 NaN、无形状错误；
      · T > 1 且恰好差几个 → 切片偏短 → 这一步才会抛 RuntimeError。
    也就是说：越界在「常见的 T=1 场景」下是**安静的**，坏得毫无声响。
    因此：
      1) forward 里对 start + T 做了**显式范围检查**，越界直接抛错（带 start/T/表长）；
      2) 需要更长位置范围时，只能通过 extra_len / extend_to() 把表**真的建长**，
         绝不能靠"让索引越界"糊过去。

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

    extra_len: 除 max_seq_len 之外**多建**的位置数，表长 = max_seq_len + extra_len。
        RoPE 的 cos/sin 是无参数的确定性三角函数（由 base/dim 纯计算而来），
        加长表**不引入任何新参数**，只是把"合法的位置索引范围"放大到
        [0, max_seq_len + extra_len)。默认 0 = 与旧实现逐位一致。
        用途：KV cache 允许超窗 max_overrun 步后，位置索引会到
        block_size + max_overrun - 1，表必须覆盖到 block_size + max_overrun。
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0,
                 extra_len: int = 0):
        super().__init__()
        assert dim % 2 == 0, f'RoPE 需要 head_dim 为偶数，得到 {dim}'
        assert extra_len >= 0, f'extra_len 不能为负，得到 {extra_len}'
        self.dim = dim
        self.base = base                      # 供 extend_to() 重算 inv_freq 用
        self.max_seq_len = max_seq_len
        self.extra_len = extra_len
        t = torch.arange(max_seq_len + extra_len, dtype=torch.float32)
        # freqs (T, dim/2)；重复两半 → (T, dim)（配合 rotate_half）
        freqs = torch.outer(t, self._inv_freq())
        emb = torch.cat((freqs, freqs), dim=-1)          # (T, dim)
        # 形状 (1, 1, T, dim)：与 q/k (B, H, T, head_dim) 按位置维(T)广播
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))

    def _inv_freq(self) -> torch.Tensor:
        """inv_freq = base^(-2i/dim)，i ∈ [0, dim/2)（float32，与原实现同序同 dtype）。"""
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

    def extend_to(self, total_len: int) -> int:
        """把 cos/sin 表**就地**增长到至少 total_len（幂等；不改任何参数）。

        为什么需要"就地"：GPT.__init__ 把**同一个** RotaryEmbedding 对象共享给所有
        block（每个 attn.rope 引用它），所以增长 gpt.rope 这一张表，所有层同时可见。
        （若每个 block 各持一份副本，这里就必须逐层处理 → 见
        generation.ensure_pos_capacity 里对"共享 = 同一对象"的显式断言。）

        返回增长到的表长。total_len <= 当前表长时不做任何事（保证与旧行为逐位一致）。
        新行沿用与原表**完全相同**的公式/dtype/设备，旧行原样保留（不重算 → 不引入差异）。
        """
        cur = self.cos_cached.size(2)
        if total_len <= cur:
            return cur
        t = torch.arange(cur, total_len, dtype=torch.float32,
                         device=self.cos_cached.device)
        freqs = torch.outer(t, self._inv_freq().to(self.cos_cached.device))
        emb = torch.cat((freqs, freqs), dim=-1)          # (新增行, dim)
        new_cos = emb.cos().unsqueeze(0).unsqueeze(0)
        new_sin = emb.sin().unsqueeze(0).unsqueeze(0)
        # 赋回同名 buffer：nn.Module.__setattr__ 会写回 _buffers（仍是 buffer，不是普通属性）
        self.cos_cached = torch.cat(
            [self.cos_cached, new_cos.to(self.cos_cached.dtype)], dim=2)
        self.sin_cached = torch.cat(
            [self.sin_cached, new_sin.to(self.sin_cached.dtype)], dim=2)
        self.extra_len = self.cos_cached.size(2) - self.max_seq_len
        return self.cos_cached.size(2)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                start: int = 0):
        """旋转 q/k（各自 (B, H, T, head_dim)）。start = 起始绝对位置（KV cache 偏移）。

        ★ 显式范围检查（见模块顶部"静默失效警告"）：越界切片在 T=1 时不会报错，
          而是安静地跳过位置编码。所以这里宁可抛错，也不让它悄悄算错。
        """
        T = q.size(2)
        table_len = self.cos_cached.size(2)
        if start + T > table_len:
            raise IndexError(
                f'RoPE 位置越界：start={start}, T={T} → 需要表长 >= {start + T}，'
                f'但 cos/sin 表长只有 {table_len}（max_seq_len={self.max_seq_len}, '
                f'extra_len={self.extra_len}）。'
                f'若不扩表而硬跑，T=1 时切片会静默变空、位置编码被整个跳过（算出垃圾）。'
                f'请扩表：RotaryEmbedding(..., extra_len=N) / GPT(rope_extra_len=N) / '
                f'generation.ensure_pos_capacity(gpt, N)。')
        cos = self.cos_cached[:, :, start:start + T]   # 位置维在 dim2
        sin = self.sin_cached[:, :, start:start + T]
        return (q * cos + rotate_half(q) * sin,
                k * cos + rotate_half(k) * sin)
