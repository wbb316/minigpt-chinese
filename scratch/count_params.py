# -*- coding: utf-8 -*-
"""实测若干候选模型形状的参数量（不靠估算，直接实例化 + get_num_params）。

50M 基线 = 12L/576d/9H/swiglu/tie → 实测 51,421,056（vocab=6144）
目标：100M 量级，且 head_dim 保持 64（与 8H/512d、9H/576d 一致，RoPE 已验证的宽度）。

用法: python scratch/count_params.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPT                                  # noqa: E402
from model.layers import swiglu_hidden                     # noqa: E402

# (n_layer, n_embd, n_head)  —— 全部满足 n_embd / n_head == 64
CANDS = [
    (12, 576, 9),    # 50M 基线（v3 线）
    (16, 704, 11),   # 推荐：等比例放大
    (20, 640, 10),
    (24, 576, 9),    # 只加深度（最干净地隔离深度变量）
    (14, 768, 12),
    (12, 768, 12),   # 贴近 GPT-2 small 比例
    (18, 704, 11),
]

VOCABS = [6144, 8192]

print(f'{"L":>3} {"d":>4} {"H":>3} {"hd":>3} {"h_ffn":>6} '
      f'{"V=6144":>12} {"V=8192":>12} {"d/L":>6}')
print('-' * 66)
rows = []
for L, d, H in CANDS:
    assert d % H == 0, (L, d, H)
    h = swiglu_hidden(d, 0)
    counts = {}
    for V in VOCABS:
        m = GPT(vocab_size=V, block_size=512, n_layer=L, n_head=H, n_embd=d,
                dropout=0.1, tie_embeddings=True, position_encoding='rope',
                ff_type='swiglu', ff_hidden=h or None)
        counts[V] = m.get_num_params()
        del m
    rows.append((L, d, H, d // H, h, counts[6144], counts[8192], d / L))
    print(f'{L:>3} {d:>4} {H:>3} {d//H:>3} {h:>6} '
          f'{counts[6144]:>12,} {counts[8192]:>12,} {d/L:>6.1f}')

print()
print('目标 ~100M 的候选（|params-100M| 排序，按 V=8192）：')
for r in sorted(rows, key=lambda x: abs(x[6] - 100e6)):
    print(f'  {r[0]:>2}L/{r[1]:>3}d/{r[2]:>2}H  V=6144 {r[5]:>11,}  '
          f'V=8192 {r[6]:>11,}  (差 {r[6]-100e6:+,})')
