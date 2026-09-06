# -*- coding: utf-8 -*-
"""encode pipeline benchmark：单进程 encode vs 多进程分片管线。

规模: 1M / 10M / 100M 字符（真实 webnovel 语料）。
指标: chars/s, tokens/s, wall time, peak RAM, output size。

用法:
  python benchmark/benchmark_encode.py            # 全部（10M/100M 较久）
  python benchmark/benchmark_encode.py --chars 1000000
  python benchmark/benchmark_encode.py --fast-only
"""
import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from data.tokenizer import BPETokenizer  # noqa: E402
from data.token_cache import encode_to_cache, ShardMemmap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, 'data', 'train_webnovel_v2.txt')
TOK = os.path.join(ROOT, 'result', '20M参数+416Mtokens', 'tokenizer_best.pkl')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'bench_encode_tmp')

try:
    import psutil
except ImportError:
    psutil = None


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6 if psutil else -1


def load_text(chars):
    """从真实语料均匀取 8 段拼接（保持多样性）。"""
    import pickle
    if not os.path.exists(REAL):
        raise FileNotFoundError(REAL)
    size = os.path.getsize(REAL)
    seg = max(1, chars // 8)
    step = size // 8
    parts = []
    with open(REAL, 'rb') as f:
        for i in range(8):
            f.seek(i * step)
            raw = f.read(seg * 4)
            parts.append(raw.decode('utf-8', errors='replace')[:seg])
    text = ''.join(parts)[:chars]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chars', type=int, default=0)
    ap.add_argument('--fast-only', action='store_true',
                    help='跳过单进程对照（100M 时省时）')
    args = ap.parse_args()

    sizes = [args.chars] if args.chars else [1_000_000, 10_000_000, 100_000_000]

    with open(TOK, 'rb') as f:
        import pickle
        tok = pickle.load(f)
    print(f'tokenizer: vocab={len(tok.vocab)} merges={len(tok.merges)}')

    import csv
    rows = []
    for chars in sizes:
        text = load_text(chars)
        print(f'\n===== {chars/1e6:.0f}M 字符 =====', flush=True)
        n_tokens = None

        # --- 1. 单进程 legacy encode（对照） ---
        if not args.fast_only:
            gc.collect()
            r0 = rss_mb()
            t0 = time.time()
            ids = tok.encode(text)
            dt = time.time() - t0
            r1 = rss_mb()
            n_tokens = len(ids)
            print(f'单进程 encode: {dt:.1f}s | {chars/dt/1e3:.0f}K chars/s | '
                  f'{n_tokens/dt/1e3:.0f}K tok/s | RSS {max(0,r1-r0):.0f}MB',
                  flush=True)
            rows.append(['single', chars, round(dt, 2),
                         round(max(0, r1 - r0), 0), n_tokens])

        # --- 2. 多进程分片管线（worker 直写 uint16 bin） ---
        gc.collect()
        r0 = rss_mb()
        t0 = time.time()
        cache_dir = os.path.join(OUT_DIR, f'bench_{chars}')
        import shutil
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        cp = encode_to_cache(text, tok, os.path.dirname(cache_dir),
                             f'bench_{chars}', n_procs=8)
        mm = ShardMemmap(cp)
        dt = time.time() - t0
        r1 = rss_mb()
        n2 = mm.token_count
        total_bytes = sum(os.path.getsize(os.path.join(cp, s['file']))
                          for s in mm.index['shards'])
        print(f'多进程管线: {dt:.1f}s | {chars/dt/1e3:.0f}K chars/s | '
              f'{n2/dt/1e3:.0f}K tok/s | RSS {max(0,r1-r0):.0f}MB | '
              f'uint16 输出 {total_bytes/1e6:.0f}MB', flush=True)
        rows.append(['multiproc', chars, round(dt, 2),
                     round(max(0, r1 - r0), 0), n2])
        mm.close()

        # 一致性（若跑了单进程）
        if n_tokens is not None and n_tokens != n2:
            print(f'!!! token 数不一致: {n_tokens} vs {n2} !!!')
            sys.exit(2)

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'benchmark_encode_results.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'chars', 'time_sec', 'peak_rss_mb', 'tokens'])
        w.writerows(rows)
    print(f'\n✅ {out_csv}')


if __name__ == '__main__':
    main()
