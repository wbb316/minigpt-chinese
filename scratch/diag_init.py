# -*- coding: utf-8 -*-
"""诊断：initial CE>300 的根因验证 + 底层模型尺度审查。

假设：nn.Embedding 默认 N(0,1) init + tie_embeddings → head 共享 std=1 权重
      → logits 尺度爆炸 → initial CE 巨大（应 ≈ ln(vocab)）。

对照：小初始化 embedding (std=0.02，标准 GPT 做法)。
输出：logits std / initial CE / LayerNorm 输出统计。
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT  # noqa: E402

VOCAB, N_LAYER, N_HEAD, N_EMBD, BLOCK = 1024, 4, 4, 128, 64
B, T = 2, 64


def gpt_stats(gpt, x, y):
    with torch.no_grad():
        logits = gpt(x)
        ce = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        print(f'  logits: mean={logits.mean():.3f} std={logits.std():.3f} '
              f'max={logits.max():.3f} min={logits.min():.3f}')
        print(f'  initial CE = {ce.item():.3f}   (随机基线应 ≈ ln({VOCAB}) = '
              f'{math.log(VOCAB):.2f})')
        emb_std = gpt.token_emb.weight.std().item()
        print(f'  token_emb weight std = {emb_std:.4f}')
        if gpt.tie_embeddings:
            print(f'  head.weight is token_emb.weight (tie): std = {emb_std:.4f}')
        return ce.item()


print('=' * 60)
print('模型 A: 当前实现（Embedding 默认 N(0,1) + tie）')
print('=' * 60)
torch.manual_seed(0)
gpt_a = GPT(vocab_size=VOCAB, block_size=BLOCK, n_layer=N_LAYER,
            n_head=N_HEAD, n_embd=N_EMBD, dropout=0.0, tie_embeddings=True)
x = torch.randint(0, VOCAB, (B, T))
y = torch.randint(0, VOCAB, (B, T))
ce_a = gpt_stats(gpt_a, x, y)

print()
print('=' * 60)
print('模型 B: 小初始化 embedding (std=0.02，标准 GPT) + tie')
print('=' * 60)
torch.manual_seed(0)
gpt_b = GPT(vocab_size=VOCAB, block_size=BLOCK, n_layer=N_LAYER,
            n_head=N_HEAD, n_embd=N_EMBD, dropout=0.0, tie_embeddings=True)
with torch.no_grad():
    gpt_b.token_emb.weight.normal_(mean=0.0, std=0.02)
ce_b = gpt_stats(gpt_b, x, y)

print()
print('=' * 60)
print('模型 C: 小初始化 + 非 tie（隔离变量）')
print('=' * 60)
torch.manual_seed(0)
gpt_c = GPT(vocab_size=VOCAB, block_size=BLOCK, n_layer=N_LAYER,
            n_head=N_HEAD, n_embd=N_EMBD, dropout=0.0, tie_embeddings=False)
with torch.no_grad():
    gpt_c.token_emb.weight.normal_(mean=0.0, std=0.02)
ce_c = gpt_stats(gpt_c, x, y)

print()
print(f'结论: A(默认)={ce_a:.1f}  B(小init+tie)={ce_b:.1f}  '
      f'C(小init+非tie)={ce_c:.1f}')
