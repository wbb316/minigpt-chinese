# -*- coding: utf-8 -*-
"""tokenizer 统一评测：对多个 tokenizer pkl 跑同一套 held-out benchmark 指标。

指标（用户指定的全部）：
  tokens/char, characters/token, bytes/token, 总 token 数
  vocab 利用率, token 频率分布(top100 覆盖 / 熵)
  不同小说压缩率(B1 每本 chars/token 均值±std)
  encode 速度(chars/s), decode 速度(tokens/s)
  100% round-trip 正确性

用法:
  python scratch/eval_tokenizers.py --toks tok_4M.pkl:4M tok_8M.pkl:8M ... --bench tok_bench --out result.json
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_tok(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def metrics_for(tok, text, vocab_size=6144):
    """在给定文本上算全套指标。"""
    n_chars = len(text)
    n_bytes = len(text.encode('utf-8'))

    t0 = time.perf_counter()
    ids = tok.encode(text)
    enc_dt = time.perf_counter() - t0
    n_tok = len(ids)

    t0 = time.perf_counter()
    back = tok.decode(ids)
    dec_dt = time.perf_counter() - t0

    rt_ok = (back == text)

    cnt = Counter(ids)
    used = len(cnt)
    top100 = sum(c for _, c in cnt.most_common(100)) / max(1, n_tok)
    # 频率分布熵（bits）
    ent = -sum((c / n_tok) * math.log2(c / n_tok) for c in cnt.values()) if n_tok else 0.0

    return {
        'chars': n_chars,
        'bytes': n_bytes,
        'tokens': n_tok,
        'tokens_per_char': n_tok / max(1, n_chars),
        'chars_per_token': n_chars / max(1, n_tok),
        'bytes_per_token': n_bytes / max(1, n_tok),
        'vocab_used': used,
        'vocab_util': used / vocab_size,
        'top100_coverage': top100,
        'freq_entropy_bits': ent,
        'encode_chars_per_s': n_chars / max(1e-9, enc_dt),
        'decode_tokens_per_s': n_tok / max(1e-9, dec_dt),
        'roundtrip_ok': rt_ok,
        'unique_token_ids': sorted(cnt.keys())[:5],   # 调试用
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--toks', nargs='+', required=True, help='格式 path:label')
    ap.add_argument('--bench', default='tok_bench')
    ap.add_argument('--out', default='tok_eval_result.json')
    ap.add_argument('--vocab-size', type=int, default=0,
                    help='vocab 利用率分母；0 = 自动取各 tokenizer 自身 vocab 大小')
    args = ap.parse_args()

    bench = args.bench
    # B1: 20 本 jsonl
    b1_books = []
    with open(os.path.join(bench, 'bench_B1.jsonl'), encoding='utf-8') as f:
        for line in f:
            b1_books.append(json.loads(line))
    b2 = open(os.path.join(bench, 'bench_B2.txt'), encoding='utf-8').read()
    b3 = open(os.path.join(bench, 'bench_B3.txt'), encoding='utf-8').read()
    b1_all = ''.join(r['text'] for r in b1_books)
    print(f'benchmark: B1 {len(b1_books)} 本/{len(b1_all):,}字符  B2 {len(b2):,}  B3 {len(b3):,}')

    results = {}
    for spec in args.toks:
        path, label = spec.rsplit(':', 1)
        if not os.path.exists(path):
            print(f'⚠️ 跳过（不存在）: {path}')
            continue
        tok = load_tok(path)
        r = {'label': label, 'path': path, 'vocab_size': len(tok.vocab)}
        vs = args.vocab_size if args.vocab_size > 0 else len(tok.vocab)
        r['B1'] = metrics_for(tok, b1_all, vs)
        r['B2'] = metrics_for(tok, b2, vs)
        r['B3'] = metrics_for(tok, b3, vs)
        # 分书压缩率
        per_book = []
        for bk in b1_books:
            ids = tok.encode(bk['text'])
            per_book.append(len(bk['text']) / max(1, len(ids)))
        r['B1_per_book_chars_per_token'] = per_book
        mean = sum(per_book) / len(per_book)
        var = sum((x - mean) ** 2 for x in per_book) / len(per_book)
        r['B1_per_book_mean'] = mean
        r['B1_per_book_std'] = math.sqrt(var)
        results[label] = r
        print(f'  [{label}] B1 chars/tok={r["B1"]["chars_per_token"]:.4f} '
              f'vocab_util={r["B1"]["vocab_util"]:.4f} roundtrip={r["B1"]["roundtrip_ok"]} '
              f'B2 chars/tok={r["B2"]["chars_per_token"]:.4f} B3={r["B3"]["chars_per_token"]:.4f}')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {args.out}')

    # 打印汇总表
    print('\n' + '=' * 130)
    hdr = (f'{"tokenizer":>10} | {"tok/char":>8} | {"char/tok":>8} | {"bytes/tok":>9} | '
           f'{"B2 tok":>9} | {"vocab用":>8} | {"top100":>7} | {"熵bit":>6} | '
           f'{"enc kc/s":>8} | {"dec kt/s":>8} | {"RT":>4}')
    print(hdr)
    print('-' * 130)
    for label, r in results.items():
        m = r['B2']
        print(f'{label:>10} | {m["tokens_per_char"]:>8.4f} | {m["chars_per_token"]:>8.4f} | '
              f'{m["bytes_per_token"]:>9.4f} | {m["tokens"]:>9,} | {m["vocab_used"]:>8} | '
              f'{m["top100_coverage"]:>7.4f} | {m["freq_entropy_bits"]:>6.2f} | '
              f'{m["encode_chars_per_s"]/1000:>8.1f} | {m["decode_tokens_per_s"]/1000:>8.1f} | '
              f'{"OK" if m["roundtrip_ok"] else "FAIL":>4}')
    print('=' * 130)


if __name__ == '__main__':
    main()
