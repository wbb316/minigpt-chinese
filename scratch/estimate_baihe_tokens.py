# -*- coding: utf-8 -*-
"""估算百合语料用 vocab6144 tokenizer（6M 归档自带）编码后的 token 数。
取 train.txt 前 2MB 字符测比率，外推总量。
"""
import pickle
import sys
import time

sys.path.insert(0, r'D:\WBB_Python\pytorch')  # 供 tokenizer pkl 反序列化找 data.tokenizer

tok_path = r'D:\WBB_Python\pytorch\result\6M参数+LN修复\tokenizer_best.pkl'
train_txt = r'D:\WBB_Python\pytorch\data\train.txt'
val_txt = r'D:\WBB_Python\pytorch\data\val.txt'

with open(tok_path, 'rb') as f:
    tok = pickle.load(f)
print('tokenizer vocab:', len(tok.vocab))

import os
tr_size = os.path.getsize(train_txt)
va_size = os.path.getsize(val_txt)
print(f'train.txt bytes: {tr_size} ({tr_size/1e6:.1f}M), val.txt bytes: {va_size} ({va_size/1e6:.1f}M)')

# 采样编码 2MB 字符测比率（中文字符 3B → 采样字节数要 3 的倍数对齐边界，取首 2MB）
sample_bytes = 2_000_000
with open(train_txt, 'r', encoding='utf-8', errors='ignore') as f:
    sample = f.read(sample_bytes)
t0 = time.time()
ids = tok.encode(sample)
dt = time.time() - t0
chars = len(sample)
ratio = len(ids) / chars
print(f'采样 {chars} 字符 → {len(ids)} token，比率 {ratio:.3f} token/字符，编码耗时 {dt:.1f}s')
print(f'外推 train ≈ {int(tr_size/3 * ratio):,} token（按纯中文 3B/char 估）')
print(f'外推 val   ≈ {int(va_size/3 * ratio):,} token')
