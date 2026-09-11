# -*- coding: utf-8 -*-
"""生成 tokenizer 评测用 held-out benchmark（一次生成后冻结，4 个 tokenizer 共用）。

来源：data/val_webnovel_v2.txt（405MB，**不参与任何 tokenizer 训练**，天然 held-out）
格式：每本小说以 【书名】 开头。

构成：
  B1 book-level : 固定 seed 随机 20 本，每本取中部 20K 字符（避开开头格式差异）
  B2 uniform    : 全 val 均匀撒 200 段 × 2K 字符 = 400K 字符
  B3 boundary   : 含特殊/边界内容的片段（emoji、英文、数字、罕见标点、极长无标点串）

输出：tok_bench/ 下 bench_B1.jsonl（20 行）+ bench_B2.txt + bench_B3.txt + meta.json
"""
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(ROOT, 'data', 'val_webnovel_v2.txt')
OUT_DIR = os.path.join(ROOT, 'tok_bench')
SEED = 20260910


def split_books(text):
    """按 【书名】 切分成书列表。"""
    idx = [m.start() for m in re.finditer(r'【[^】]{1,60}】', text)]
    books = []
    for i, s in enumerate(idx):
        e = idx[i + 1] if i + 1 < len(idx) else len(text)
        books.append(text[s:e])
    return books


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'读取 {VAL} ...', flush=True)
    with open(VAL, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    print(f'val 总字符: {len(text):,}', flush=True)

    books = split_books(text)
    print(f'切分出 {len(books)} 本小说', flush=True)
    books = [b for b in books if len(b) >= 40_000]      # 只要够长的
    print(f'其中长度 ≥40K 的: {len(books)} 本', flush=True)

    rng = random.Random(SEED)

    # ---- B1: 20 本 × 20K 字符（取中部，避开开头）----
    picked = rng.sample(range(len(books)), min(20, len(books)))
    b1 = []
    for i in picked:
        b = books[i]
        start = (len(b) - 20_000) // 2
        seg = b[start:start + 20_000]
        title = b[:b.index('】') + 1] if '】' in b[:80] else f'book_{i}'
        b1.append({'book_id': i, 'title': title, 'text': seg})
    with open(os.path.join(OUT_DIR, 'bench_B1.jsonl'), 'w', encoding='utf-8') as f:
        for r in b1:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'B1: {len(b1)} 本 × 20K = {sum(len(r["text"]) for r in b1):,} 字符', flush=True)

    # ---- B2: 全 val 均匀撒 200 段 × 2K 字符 ----
    seg_chars, n_seg = 2_000, 200
    stride = len(text) // n_seg
    b2 = ''.join(text[i * stride:i * stride + seg_chars] for i in range(n_seg))
    with open(os.path.join(OUT_DIR, 'bench_B2.txt'), 'w', encoding='utf-8') as f:
        f.write(b2)
    print(f'B2: {len(b2):,} 字符（{n_seg} 段 × {seg_chars}）', flush=True)

    # ---- B3: 边界内容（从 val 中找含特殊字符的片段 + 人工补充）----
    special_pat = re.compile(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]|[\x00-\x1f]|[０-９Ａ-Ｚ]')
    hits, scan = [], 0
    while scan < len(text) and len(hits) < 6:
        chunk = text[scan:scan + 5_000]
        if special_pat.search(chunk) or 'http' in chunk or re.search(r'[A-Za-z]{20,}', chunk):
            hits.append(chunk[:2_000])
        scan += 50_000
    manual = (
        '😀🎉💯 emoji 测试 ★☆♠♥→←↑↓ 全角ＡＢＣ１２３ '
        'This is an English sentence with numbers 12345 and symbols @#$%^&*(). '
        '<html>&amp;&lt;tag&gt;</html> URL: https://example.com/path?q=1 '
        '没有任何标点的超长连续中文字符串用来测试预分词粒度' * 3 +
        '\t\n\r 制表与换行\u200b零宽空格 '
        '①②③㊀㊁ⅢⅣ 罗马数字与序号 '
    )
    b3 = manual + ''.join(hits)
    with open(os.path.join(OUT_DIR, 'bench_B3.txt'), 'w', encoding='utf-8') as f:
        f.write(b3)
    print(f'B3: {len(b3):,} 字符（人工边界串 + {len(hits)} 段特殊内容）', flush=True)

    meta = {
        'seed': SEED,
        'source': 'data/val_webnovel_v2.txt',
        'note': 'held-out: val 语料不参与 tokenizer 训练（训练样本取自 train_webnovel_v2.txt）',
        'B1': {'books': len(b1), 'chars_per_book': 20_000,
               'total_chars': sum(len(r['text']) for r in b1),
               'book_ids': [r['book_id'] for r in b1]},
        'B2': {'segments': n_seg, 'chars_per_seg': seg_chars, 'total_chars': len(b2)},
        'B3': {'total_chars': len(b3), 'special_hits': len(hits)},
    }
    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'\nbenchmark 已生成 → {OUT_DIR}')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
