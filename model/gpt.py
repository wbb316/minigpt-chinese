import torch
import torch.nn as nn
from model.layers import LayerNorm, PositionalEncoding, FeedForward
from model.attention import MultiHeadAttention

class Block(nn.Module):
    def __init__(self,dim,n_head,dropout=0.0):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn= MultiHeadAttention(dim,n_head,dropout=dropout)   # ★ 传 dropout
        self.ln2 = LayerNorm(dim)
        self.ff= FeedForward(dim, dropout=dropout)                  # ★ FFN 也加 dropout

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
                   n_embd: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = PositionalEncoding(n_embd, max_len=block_size)
        self.drop = nn.Dropout(dropout)                            # ★ embedding dropout
        self.blocks=nn.ModuleList([Block(n_embd,n_head,dropout=dropout) for _ in range(n_layer)])   # ★ 传 dropout
        self.ln=LayerNorm(n_embd)   # 用我们自己写的 LayerNorm
        self.head=nn.Linear(n_embd,vocab_size)

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
        if past_kvs is not None:
            # 新 token 的真实全局位置 = 缓存的历史长度
            start = past_kvs[0][0].size(2)
            x = self.pos_emb(x, start=start)
        else:
            x = self.pos_emb(x)
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
        return sum(p.numel() for p in self.parameters())
