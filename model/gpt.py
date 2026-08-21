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
        self.ff= FeedForward(dim)

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        x=x+self.attn(self.ln1(x))
        x=x+self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int, n_head: int,
                   n_embd: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = PositionalEncoding(n_embd, max_len=block_size)
        self.blocks=nn.ModuleList([Block(n_embd,n_head,dropout=dropout) for _ in range(n_layer)])   # ★ 传 dropout
        self.ln=LayerNorm(n_embd)   # 用我们自己写的 LayerNorm
        self.head=nn.Linear(n_embd,vocab_size)

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        B,T=x.shape
        assert T<=self.block_size
        x = self.token_emb(x)
        x= self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        x=self.head(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

