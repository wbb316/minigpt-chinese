# -*- coding: utf-8 -*-
"""检查已训练 tokenizer 的结构，验证「大 vocab 截取小 vocab」可行性。

原理：BPE 贪心合并顺序由 counts 决定，与目标 vocab_size 无关（不提前停止时）。
      merges[(a,b)] = nid 且 nid 递增 → 截断 id < N 即得 vocab=N 的等价 tokenizer。
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = pickle.load(open(os.path.join(ROOT, 'tok_exp', 'tok', 'tok_4M.pkl'), 'rb'))

print('vocab_size(目标):', tok.vocab_size)
print('vocab 实际条目:', len(tok.vocab))
print('merges 条目:', len(tok.merges))
print()
# id 分布
ids = sorted(tok.vocab.keys())
print('vocab id 范围:', min(ids), '~', max(ids))
print('前 10 个 id:', ids[:10])
print('后 5 个 id:', ids[-5:])
# merges 的 nid 范围
nids = sorted(tok.merges.values())
print('merges nid 范围:', min(nids), '~', max(nids))
# 检查 nid 是否连续递增（BPE 顺序分配的标志）
expected_start = 258   # 256 bytes + 2 special
seq_ok = nids == list(range(expected_start, expected_start + len(nids)))
print(f'nid 是否连续递增（{expected_start}..{max(nids)}）:', seq_ok)
print()
# 检查是否存在 nid 冲突（截断安全性的前提）
dup = len(nids) != len(set(nids))
print('nid 有重复（不安全）:', dup)
# 特殊 token
print('special tokens:', {k: v for k, v in list(tok.vocab.items())[:4]})
