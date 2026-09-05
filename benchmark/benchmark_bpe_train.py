# -*- coding: utf-8 -*-
"""BPE tokenizer trainer 性能 benchmark（legacy vs fast）。

分级规模（任务书第七节）:
  Stage 1: 200K 字符
  Stage 2: 1M 字符
  Stage 3: 4M 字符
vocab: 1024 / 3256 / 6144

记录: trainer, sample_chars, vocab, time_sec, peak_rss_mb, merges_sec,
      final_vocab, merge_count, equivalent → benchmark_results.csv

语料: 优先用真实语料 data/train_webnovel_v2.txt（存在时流式取，
      否则用内置混合中文语料生成）。

用法:
  python benchmark/benchmark_bpe_train.py            # 全跑（可能较久）
  python benchmark/benchmark_bpe_train.py --stage 1 # 只跑 200K
  python benchmark/benchmark_bpe_train.py --vocab 6144 --chars 200000
"""
import argparse
import csv
import gc
import os
import sys
import time

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import BPETokenizer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_CORPUS = os.path.join(ROOT, 'data', 'train_webnovel_v2.txt')
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'benchmark_results.csv')

# 内置语料兜底（真实语料不存在时）
_FALLBACK_CN = ('她抬起头望向窗外的天空心里想着那个人的模样，夜色渐渐深了。'
                '他走进教室看见黑板上的字迹模糊不清。'
                '不知道为什么总觉得今天会有什么事情发生。'
                '窗外的风轻轻吹过树梢，带起一阵沙沙的声响。') * 200
_FALLBACK_EN = ('The quick brown fox jumps over the lazy dog while counting '
                '1234567890 numbers and symbols!? ' * 100)


def load_text(chars: int) -> str:
    """流式取真实语料前 chars 字符（多样：均匀跳段采样）；失败用内置语料。"""
    if os.path.exists(REAL_CORPUS):
        try:
            size = os.path.getsize(REAL_CORPUS)
            # 从文件中均匀取 8 段拼接，保证多样性（避免开头单本书偏差）
            seg_target = max(1, chars // 8)
            step = max(1, size // 8)
            parts = []
            with open(REAL_CORPUS, 'rb') as f:
                for i in range(8):
                    f.seek(i * step)
                    raw = f.read(seg_target * 4)
                    parts.append(raw.decode('utf-8', errors='replace')[:seg_target])
            text = ''.join(parts)
            if len(text) >= chars:
                return text[:chars]
        except Exception as e:
            print(f'  真实语料读取失败({e})，用内置语料')
    return (_FALLBACK_CN + _FALLBACK_EN)[:chars]


def rss_mb() -> float:
    """当前进程 RSS（MB）。无 psutil 时返回 -1。"""
    if _HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1e6
    return -1.0


def train_and_measure(text, vocab, backend):
    """训练一次，返回 (time_sec, peak_rss_mb, tokenizer)。

    注意：不要用 tracemalloc 包裹训练——它对百万级 dict 分配的 hook
    会让训练慢 ~8 倍，严重失真（实测 6.7s → 51s）。用 psutil RSS 前后差。
    """
    tok = BPETokenizer(vocab_size=vocab)
    gc.collect()
    rss0 = rss_mb()
    t0 = time.time()
    tok.train(text, backend)
    elapsed = time.time() - t0
    rss1 = rss_mb()
    peak = max(0.0, rss1 - rss0)
    return elapsed, peak, tok


def main():
    ap = argparse.ArgumentParser(description='BPE trainer benchmark')
    ap.add_argument('--stage', type=int, default=0,
                    help='1=200K only, 2=1M only, 3=4M only; 0=all')
    ap.add_argument('--vocab', type=int, default=0,
                    help='只跑指定 vocab (1024/3256/6144); 0=全部')
    ap.add_argument('--chars', type=int, default=0, help='自定义字符数')
    args = ap.parse_args()

    stages = {1: 200_000, 2: 1_000_000, 3: 4_000_000}
    if args.chars:
        runs = [(f'{args.chars/1e3:.0f}K', args.chars)]
    elif args.stage:
        runs = [(f'{stages[args.stage]/1e3:.0f}K', stages[args.stage])]
    else:
        runs = [(f'{v/1e3:.0f}K', v) for v in stages.values()]
    vocabs = [args.vocab] if args.vocab else [1024, 3256, 6144]

    results = []
    for label, chars in runs:
        text = load_text(chars)
        print(f'\n=== 语料 {label} 字符 (实际 {len(text):,}) ===', flush=True)
        for vocab in vocabs:
            # legacy first（拿 reference + 计时）
            print(f'--- vocab {vocab} / legacy ---', flush=True)
            t_legacy, peak_legacy, tok_legacy = train_and_measure(
                text, vocab, 'legacy')

            print(f'--- vocab {vocab} / fast ---', flush=True)
            t_fast, peak_fast, tok_fast = train_and_measure(
                text, vocab, 'fast')

            equiv = (tok_legacy.merges == tok_fast.merges
                     and tok_legacy.vocab == tok_fast.vocab)
            n_merges = len(tok_fast.merges)
            speedup = t_legacy / t_fast if t_fast > 0 else float('inf')
            print(f'  legacy {t_legacy:.1f}s | fast {t_fast:.1f}s | '
                  f'speedup {speedup:.1f}x | merges {n_merges} | '
                  f'等价 {"✓" if equiv else "❌ FAIL"} | '
                  f'RSS fast {peak_fast:.0f}MB')
            results.append({
                'trainer': 'legacy', 'sample_chars': len(text), 'vocab': vocab,
                'time_sec': round(t_legacy, 2),
                'peak_rss_mb': round(peak_legacy, 1),
                'merges_sec': round(n_merges / t_legacy, 1),
                'final_vocab': len(tok_legacy.vocab),
                'merge_count': n_merges, 'equivalent': 'n/a',
            })
            results.append({
                'trainer': 'fast', 'sample_chars': len(text), 'vocab': vocab,
                'time_sec': round(t_fast, 2),
                'peak_rss_mb': round(peak_fast, 1),
                'merges_sec': round(n_merges / t_fast, 1),
                'final_vocab': len(tok_fast.vocab),
                'merge_count': n_merges, 'equivalent': 'PASS' if equiv else 'FAIL',
            })
            if not equiv:
                print('!!! 等价性失败，终止后续 benchmark !!!')
                sys.exit(2)
            del tok_legacy, tok_fast
            gc.collect()

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f'\n✅ 结果已写: {OUT_CSV}')
    # 摘要
    for r in results:
        if r['trainer'] == 'fast':
            print(f"  fast v{r['vocab']} {r['sample_chars']:,}: "
                  f"{r['time_sec']}s | {r['merges_sec']}/s | {r['equivalent']}")


if __name__ == '__main__':
    main()
