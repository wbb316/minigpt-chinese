# -*- coding: utf-8 -*-
"""把 shard1+2 语料 gzip 压缩，供上传云端（上行仅 ~0.5 MB/s，压缩能省一半时间）。

用法: python scratch/gzip_corpus.py
输出: data/train_webnovel_shard12.txt.gz / data/val_webnovel_shard12.txt.gz
"""
import gzip
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = [
    (os.path.join(ROOT, 'data', 'train_webnovel_shard12.txt'),
     os.path.join(ROOT, 'data', 'train_webnovel_shard12.txt.gz')),
    (os.path.join(ROOT, 'data', 'val_webnovel_shard12.txt'),
     os.path.join(ROOT, 'data', 'val_webnovel_shard12.txt.gz')),
]

for src, dst in PAIRS:
    if not os.path.exists(src):
        print(f'❌ 缺文件: {src}')
        continue
    raw = os.path.getsize(src)
    t0 = time.perf_counter()
    with open(src, 'rb') as fi, gzip.open(dst, 'wb', compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, 8 << 20)
    dt = time.perf_counter() - t0
    comp = os.path.getsize(dst)
    print(f'{os.path.basename(src)}: {raw/1e9:.3f}GB → {comp/1e9:.3f}GB '
          f'({comp/raw:.1%})  {dt:.0f}s  ({raw/dt/1e6:.0f} MB/s)')
