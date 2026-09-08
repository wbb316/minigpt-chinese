import math

import torch
import torch.nn as nn
from model.layers import (LayerNorm, PositionalEncoding, FeedForward, FF_TYPES,
                          DEFAULT_FF_TYPE)
from model.attention import MultiHeadAttention

class Block(nn.Module):
    def __init__(self, dim, n_head, dropout=0.0, rope=None,
                 ff_type=DEFAULT_FF_TYPE, ff_hidden=None):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_head, dropout=dropout,
                                       rope=rope)   # ★ 传 dropout / rope
        self.ln2 = LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, ff_type=ff_type,
                              ff_hidden=ff_hidden)                   # ★ FFN 变体可切换

    def forward(self, x :torch.Tensor, past_kv=None, return_kv=False):
        """past_kv: 该层缓存的 (k, v)；return_kv=True 或给了 past_kv 时返回 (x, 新kv)。"""
        h = self.ln1(x)
        a, kv = self.attn(h, past_kv=past_kv, return_kv=True)   # 总是同时拿到输出和新kv
        x = x + a
        x = x + self.ff(self.ln2(x))
        if past_kv is not None or return_kv:
            return x, kv
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int, n_head: int,
                 n_embd: int, block_size: int, dropout: float = 0.0,
                 tie_embeddings: bool = False,
                 position_encoding: str = 'rope',
                 ff_type: str = DEFAULT_FF_TYPE, ff_hidden: int | None = None):
        """position_encoding: 'rope'(默认——2026-09-06 短训对比胜出) |
        'sinusoidal'（旧实现，复现旧模型用）。

        ff_type: 'swiglu'（默认，2026-09-08 短程对照胜出）| 'relu'（v3_alpha 原 FFN，
        可回退）| 'gelu'；ff_hidden: 仅 swiglu 用，缺省按参数量对齐 4d 自动求解。

        注意：加载旧 checkpoint（v3_alpha / v2 系列，FFN=ReLU）时，推理侧
        （generate.py / app/server.py / visualize_attention.py）会从 state_dict
        自动检测 ff_type，不受本默认值影响。
        """
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings
        self.n_layer = n_layer
        assert position_encoding in ('sinusoidal', 'rope'), \
            f'position_encoding 仅支持 sinusoidal|rope，得到 {position_encoding}'
        self.position_encoding = position_encoding
        if ff_type not in FF_TYPES:
            raise ValueError(f'ff_type 仅支持 {FF_TYPES}，得到 {ff_type!r}')
        self.ff_type = ff_type
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = PositionalEncoding(n_embd, max_len=block_size)
        self.drop = nn.Dropout(dropout)                            # ★ embedding dropout
        # RoPE：sinusoidal 模式 rope=None（旧行为不变）；rope 模式建旋转器（共享各层）
        rope = None
        if position_encoding == 'rope':
            from model.rope import RotaryEmbedding  # noqa: E402
            rope = RotaryEmbedding(dim=n_embd // n_head,
                                   max_seq_len=block_size)
            self.rope = rope
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout=dropout, rope=rope,
                   ff_type=ff_type, ff_hidden=ff_hidden)
             for _ in range(n_layer)])   # ★ 传 dropout / FFN 变体
        self.ln = LayerNorm(n_embd)   # 用我们自己写的 LayerNorm
        self.head = nn.Linear(n_embd, vocab_size)
        if tie_embeddings:
            # 输入/输出嵌入共享权重（省掉 head 的 833K 参数）。
            # 保留 head.bias 不改动，checkpoint 结构（token_emb.weight/head.weight/head.bias）
            # 与不 tie 时一致，generate/visualize 等加载代码无需任何改动。
            self.head.weight = self.token_emb.weight
        # ---- GPT-2 风格初始化（2026-09-06，FUTURE item 转正）----
        # 修复：nn.Embedding 默认 N(0,1) + tie → head 共享 std=1 权重 →
        # logits 尺度爆炸 → initial CE 300+（正确基线 ≈ ln(vocab)）。
        # 标准做法：所有权重 N(0, 0.02)；残差分支按 1/sqrt(2*n_layer) 缩小。
        # 已有 checkpoint 加载路径用 load_state_dict 覆盖本初始化 → 零影响。
        self.apply(self._init_weights)
        for blk in self.blocks:      # GPT-2 residual scaled init
            nn.init.normal_(blk.attn.out_proj.weight, mean=0.0,
                            std=0.02 / math.sqrt(2.0 * n_layer))
            # 残差分支输出投影：relu/gelu = fc2，swiglu = down_proj
            # （SwiGLU 论文同样只对输出投影做 1/sqrt(2N) 缩放）
            nn.init.normal_(blk.ff.out_proj_weight(), mean=0.0,
                            std=0.02 / math.sqrt(2.0 * n_layer))

    def _init_weights(self, module):
        """GPT-2 风格权重初始化：Linear/Embedding → N(0, 0.02)，bias 置零。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x :torch.Tensor, past_kvs=None, return_kv=False):
        """GPT 前向（支持 KV cache 增量推理）。

        - past_kvs: 各层的缓存列表 [(k, v), ...]，每项 (B, H, T_prev, head_dim)。
          给定时输入 x 只需是新 token（T 很小），位置编码从缓存长度偏移。
        - return_kv=True 或给了 past_kvs 时返回 (logits, new_kvs)，否则只返回 logits。

        past_kvs=None 且 return_kv=False 时与原来行为完全一致。
        """
        B,T=x.shape
        assert T<=self.block_size
        x = self.token_emb(x)
        if self.position_encoding == 'sinusoidal':
            if past_kvs is not None:
                # 新 token 的真实全局位置 = 缓存的历史长度
                start = past_kvs[0][0].size(2)
                x = self.pos_emb(x, start=start)
            else:
                x = self.pos_emb(x)
        # rope 模式：不在 embedding 加位置向量（RoPE 在 attention 内旋转 Q/K）
        x = self.drop(x)   # ★ embedding dropout（token+pos 求和后）

        new_kvs = []
        for i, block in enumerate(self.blocks):
            if past_kvs is not None:
                x, kv = block(x, past_kvs[i], return_kv=True)
            else:
                x, kv = block(x, return_kv=True)
            new_kvs.append(kv)
        x = self.ln(x)
        x=self.head(x)

        if past_kvs is not None or return_kv:
            return x, new_kvs
        return x

    def get_num_params(self):
        # 共享参数（tie_embeddings）只计一次
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())
