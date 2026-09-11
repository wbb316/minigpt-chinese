# -*- coding: utf-8 -*-
"""提取 eval_all.json 的明细，生成结论表。"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'tok_exp', 'eval_all.json'), encoding='utf-8'))

print('=== 分书压缩率（B1 20 本 chars/token）===')
for k in ['4M', '8M', '16M', '32M']:
    r = d[k]
    pb = r['B1_per_book_chars_per_token']
    print(f'{k:>4}: mean={r["B1_per_book_mean"]:.4f} std={r["B1_per_book_std"]:.4f} '
          f'min={min(pb):.4f} max={max(pb):.4f}')

print('\n=== B1/B2/B3 三档 chars/token + vocab 使用 ===')
for k in ['4M', '8M', '16M', '32M']:
    r = d[k]
    print(f'{k:>4}: B1={r["B1"]["chars_per_token"]:.4f}(vocab {r["B1"]["vocab_used"]}) '
          f'B2={r["B2"]["chars_per_token"]:.4f}(vocab {r["B2"]["vocab_used"]}) '
          f'B3={r["B3"]["chars_per_token"]:.4f}(vocab {r["B3"]["vocab_used"]})')

print('\n=== 相对 4M 的 token 节省（B2 400K 字符）===')
base = d['4M']['B2']['tokens']
prev = None
for k in ['4M', '8M', '16M', '32M']:
    t = d[k]['B2']['tokens']
    step = '' if prev is None else f' | 相对上一档 {(prev - t) / prev * 100:+.3f}%'
    print(f'{k:>4}: {t:,} token | 相对 4M {(base - t) / base * 100:+.3f}%{step}')
    prev = t

print('\n=== vocab 未使用数量（6144 - used）===')
for k in ['4M', '8M', '16M', '32M']:
    u = d[k]['B2']['vocab_used']
    print(f'{k:>4}: 用到 {u} / 6144，未用 {6144 - u} ({u / 6144 * 100:.2f}% 利用)')
