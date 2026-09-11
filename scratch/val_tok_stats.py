# -*- coding: utf-8 -*-
"""统计各 tokenizer 在完整 val 语料上的 token 数 / chars-per-token。

用途：
  1) 交叉校验——云端 train.py 日志会打印「验证集 N token」，本脚本在本地用同一份
     val 文件复现该数字，确认两边语料完全一致。
  2) 为 bits/char 换算提供精确分母（跨 vocab 比较必须用它，不能用 benchmark 的小样本）。

用法:
  python scratch/val_tok_stats.py --val-txt data/val_webnovel_v2.txt \
      --toks path/a.pkl:labelA path/b.pkl:labelB --out tok_exp/val_tok_stats.json
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ENC_TOK = None


def _enc_init(tok):
    global _ENC_TOK
    _ENC_TOK = tok


def _enc_slice(s):
    return len(_ENC_TOK.encode(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-txt', required=True)
    ap.add_argument('--toks', nargs='+', required=True, help='格式 path:label')
    ap.add_argument('--out', default='')
    ap.add_argument('--workers', type=int, default=14)
    args = ap.parse_args()

    with open(args.val_txt, encoding='utf-8') as f:
        text = f.read()
    n_chars = len(text)
    n_bytes = len(text.encode('utf-8'))
    print(f'val: {args.val_txt}')
    print(f'  {n_chars:,} 字符 / {n_bytes:,} 字节 (bytes/char={n_bytes/n_chars:.4f})')

    import multiprocessing as mp
    step = len(text) // args.workers
    slices = []
    for i in range(args.workers):
        a = i * step
        b = len(text) if i == args.workers - 1 else (i + 1) * step
        if i > 0:
            nl = text.find('\n', a)
            if 0 <= nl < b:
                a = nl + 1
        if i < args.workers - 1:
            nl = text.rfind('\n', a, b)
            if nl > a:
                b = nl + 1
        slices.append(text[a:b])

    results = {}
    for spec in args.toks:
        path, label = spec.rsplit(':', 1)
        with open(path, 'rb') as f:
            tok = pickle.load(f)
        t0 = time.perf_counter()
        with mp.Pool(args.workers, initializer=_enc_init, initargs=(tok,)) as pool:
            parts = pool.map(_enc_slice, slices)
        n_tok = sum(parts)
        dt = time.perf_counter() - t0
        r = {
            'vocab_size': len(tok.vocab),
            'n_val_tokens': n_tok,
            'n_val_chars': n_chars,
            'n_val_bytes': n_bytes,
            'chars_per_token': n_chars / n_tok,
            'bytes_per_token': n_bytes / n_tok,
            'encode_seconds': dt,
        }
        results[label] = r
        print(f'[{label}] vocab={r["vocab_size"]} val_tokens={n_tok:,} '
              f'chars/token={r["chars_per_token"]:.4f} '
              f'bytes/token={r["bytes_per_token"]:.4f} ({dt:.0f}s)')

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'结果已保存: {args.out}')


if __name__ == '__main__':
    main()
