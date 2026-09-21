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
    """经典 sinusoidal 位置编码（可切换，见 GPT(position_encoding=...)）。

    ⚠️ 与 model/rope.py 同款坑：forward 里 `pe[:, start:start+T]` 是普通切片，
       位置越界**不报错** —— T=1（增量解码）时切片变空，(1,0,C) 与 (B,1,C) 广播后
       位置向量被整个跳过，输出"看起来正常"但位置信息丢光。故这里做显式范围检查。
    """

    def __init__(self, dim:int, max_len:int =1024, extra_len: int = 0):
        super().__init__()
        assert extra_len >= 0, f'extra_len 不能为负，得到 {extra_len}'
        self.dim = dim
        self.max_len = max_len
        self.extra_len = extra_len
        # dim_term 存成属性：extend_to() 需要用它按**同一公式**续算新位置
        self.dim_term = torch.exp(torch.arange(0, dim, 2).float()
                                  * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len + extra_len, dim)
        position = torch.arange(0, max_len + extra_len, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * self.dim_term)
        pe[:, 1::2] = torch.cos(position * self.dim_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def extend_to(self, total_len: int) -> int:
        """把 pe 表**就地**增长到至少 total_len（幂等；无参数，只是多算几行 sin/cos）。

        返回增长到的表长；total_len <= 当前表长时不做任何事。
        续算行沿用与原表完全相同的公式/dtype/设备，旧行原样保留。
        """
        cur = self.pe.size(1)
        if total_len <= cur:
            return cur
        position = torch.arange(cur, total_len, dtype=torch.float,
                                device=self.pe.device).unsqueeze(1)
        term = self.dim_term.to(self.pe.device)
        new_pe = torch.zeros(total_len - cur, self.dim, device=self.pe.device,
                             dtype=self.pe.dtype)
        new_pe[:, 0::2] = torch.sin(position * term)
        new_pe[:, 1::2] = torch.cos(position * term)
        self.pe = torch.cat([self.pe, new_pe.unsqueeze(0)], dim=1)
        self.extra_len = self.pe.size(1) - self.max_len
        return self.pe.size(1)

    def forward(self, x, start: int = 0):
        """start: 起始位置偏移（KV cache 增量解码时，新 token 的全局位置 = 缓存长度）。"""
        T = x.size(1)
        pe_len = self.pe.size(1)
        if start + T > pe_len:
            raise IndexError(
                f'sinusoidal 位置越界：start={start}, T={T} → 需要表长 >= {start + T}，'
                f'但 pe 表长只有 {pe_len}（max_len={self.max_len}, '
                f'extra_len={self.extra_len}）。越界时 T=1 会静默丢掉位置向量。'
                f'请扩表：PositionalEncoding(..., extra_len=N) / '
                f'GPT(rope_extra_len=N) / generation.ensure_pos_capacity(gpt, N)。')
        return x + self.pe[:, start:start + T]


# ---------------------------------------------------------------------------
# FFN 变体（v3 FFN 优化实验，2026-09-08）
#
#   relu   : Linear(dim→4d) → ReLU   → Linear(4d→dim)      ← v3_alpha 原实现（保留可回退）
#   gelu   : Linear(dim→4d) → GELU   → Linear(4d→dim)      ← 第一阶段：只换激活
#   swiglu : gate_proj/up_proj(dim→h) → silu(gate)*up → down_proj(h→dim)
#            ← 第二阶段：结构变化，h 重新设计使参数量对齐 4d（见 swiglu_hidden）
#
# 默认值 = DEFAULT_FF_TYPE（2026-09-08 起为 'swiglu'：5000 步短程对照胜出，见
# docs/EXPERIMENT_LOG.md「v3 FFN 变体」；relu/gelu 保留可回退）。
#
# 约束（对照实验公平性）：
#   - relu/gelu 的 module 名与参数名与 v3_alpha 完全一致（fc1/fc2）→ 旧 checkpoint
#     可直接 load，单变量只有激活函数。
#   - swiglu 的参数量与 relu 版对齐（dim=576 时 h=1536：FFN 层 +0.036%，
#     全模型 +0.018%）→ 不因参数增加而获得不公平优势。
# ---------------------------------------------------------------------------

FF_TYPES = ('relu', 'gelu', 'swiglu')
DEFAULT_FF_TYPE = 'swiglu'


def swiglu_hidden(dim: int, round_to: int = 0) -> int:
    """解 SwiGLU 的中间维 h，使参数量 ≈ 原 FFN 的 4d。

    原 FFN 参数量 = 2·4d² + 4d（fc1/fc2 权重 + 两个 bias）
    SwiGLU 参数量 = 3·h·d + h + 3d（gate/up/down 权重 + 三个 bias）

    解 3hd ≈ 8d² → h ≈ 8d/3 = 2.667d。

    对齐（round_to）：
      - 0（默认）= 自适应：8d/3 ≥ 512 时对齐到 32，更小的 dim 对齐到 8
        （对齐粒度只为 kernel/访存友好；granularity 越小参数量越贴，实测
        dim=576 → h=1536，FFN 层参数差仅 +0.036%）。
      - 指定值：按该粒度对齐；round_to=1 取最近整数（参数差最小）。
    """
    h = 8.0 * dim / 3.0
    r = int(round_to)
    if r <= 0:
        r = 32 if h >= 512 else 8
    return max(r, int(round(h / r) * r))


class FeedForward(nn.Module):
    """可切换 FFN：ff_type ∈ {'relu', 'gelu', 'swiglu'}，默认 DEFAULT_FF_TYPE（swiglu）。

    - relu/gelu：保持原 4d 中间维与 fc1/fc2 参数名 → 与 v3_alpha checkpoint 兼容。
    - swiglu：SwiGLU 门控结构；ff_hidden 缺省按参数量对齐自动求解。
    """

    def __init__(self, dim: int, dropout: float = 0.0,
                 ff_type: str = DEFAULT_FF_TYPE, ff_hidden: int | None = None):
        super().__init__()
        if ff_type not in FF_TYPES:
            raise ValueError(f'ff_type 仅支持 {FF_TYPES}，得到 {ff_type!r}')
        self.dim = dim
        self.ff_type = ff_type
        self.dropout = nn.Dropout(dropout)
        if ff_type == 'swiglu':
            self.hidden = int(ff_hidden) if ff_hidden else swiglu_hidden(dim)
            # 原实现：fc1 = Linear(dim, 4*dim)。SwiGLU 的两路上投影参数形状一致。
            self.gate_proj = nn.Linear(dim, self.hidden)
            self.up_proj = nn.Linear(dim, self.hidden)
            self.down_proj = nn.Linear(self.hidden, dim)
        else:
            self.hidden = 4 * dim
            # ★ v3_alpha 原 FFN 实现（参数名/形状不变，勿改）
            self.fc1 = nn.Linear(dim, 4 * dim)
            self.fc2 = nn.Linear(4 * dim, dim)

    # ---- 供 GPT-2 残差缩放 init 使用：返回残差分支末端的输出投影权重 ----
    def out_proj_weight(self) -> torch.Tensor:
        return self.fc2.weight if self.ff_type != 'swiglu' else self.down_proj.weight

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ff_type == 'swiglu':
            # silu(gate) * up → down（门控：gate 决定放行多少，up 提供内容）
            return self.down_proj(
                self.dropout(torch.nn.functional.silu(self.gate_proj(x))
                             * self.up_proj(x)))
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x) if self.ff_type == 'gelu' else torch.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
