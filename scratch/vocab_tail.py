# -*- coding: utf-8 -*-
"""vocab_size 消融 —— token 频率长尾分析（第二阶段决定性问题）。

问题：vocab 变大时新增的 token 都是低频 token，它们在固定 1B token 训练预算下
      能拿到多少次梯度更新？若太少，embedding 训不好 → 压缩收益被抵消。

做法：在 held-out 的 val_webnovel_v2.txt 上**分层均匀采样** 40M 字符（与训练侧同样的
      均匀分段采样，不是取前 N 字符），对每个 tokenizer 统计词频，再按
      1B token 训练预算线性外推每个 token 的出现次数。

用法:
  python scratch/vocab_tail.py --toks tok_exp/tok/tok_v4096_from8192.pkl:v4096 ... \
      --corpus data/val_webnovel_v2.txt --chars 40000000 --budget 1e9 --out tok_exp/tail_vocab.json
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_SPECIAL = 2   # <unk> <eos>


def load_tok(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def sample_uniform(path, target_chars, n_seg=100):
    """按 100 个均匀分布的段落采样，累计到 target_chars。全程二进制读写，避免字符/字节混淆。"""
    size = os.path.getsize(path)
    parts = []
    got = 0
    with open(path, 'rb') as f:
        for i in range(n_seg):
            if got >= target_chars:
                break
            pos = size * i // n_seg
            f.seek(pos)
            f.read(1)  # 丢弃可能的半个字符
            want_chars = (target_chars - got) // max(1, n_seg - i) or (target_chars - got)
            want_bytes = min(int(want_chars * 3.1) + 64, size - f.tell())
            raw = f.read(max(0, want_bytes))
            txt = raw.decode('utf-8', errors='ignore')
            # 截到完整行边界，避免半句
            cut = txt.rfind('\n')
            if cut > 0:
                txt = txt[:cut]
            need = target_chars - got
            if len(txt) > need:
                txt = txt[:need]
            parts.append(txt)
            got += len(txt)
    text = ''.join(parts)
    return text[:target_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--toks', nargs='+', required=True, help='格式 path:label')
    ap.add_argument('--corpus', default='data/val_webnovel_v2.txt')
    ap.add_argument('--chars', type=int, default=40_000_000)
    ap.add_argument('--budget', type=float, default=1e9, help='总训练 token 预算')
    ap.add_argument('--out', default='tok_exp/tail_vocab.json')
    args = ap.parse_args()

    print(f'采样 {args.chars:,} 字符 ← {args.corpus} ...')
    text = sample_uniform(args.corpus, args.chars)
    n_chars = len(text)
    print(f'实得 {n_chars:,} 字符')

    results = {}
    for spec in args.toks:
        path, label = spec.rsplit(':', 1)
        if not os.path.exists(path):
            print(f'⚠️ 跳过: {path}')
            continue
        tok = load_tok(path)
        V = len(tok.vocab)
        ids = tok.encode(text)
        n_tok = len(ids)
        # 只统计真实词表 token（排除特殊 token）
        cnt = Counter(i for i in ids if i >= NUM_SPECIAL)
        used = len(cnt)
        scale = args.budget / n_tok          # 外推到 1B token 预算
        # 每 token 在 1B 训练中的预计出现次数
        proj = {k: c * scale for k, c in cnt.items()}
        thr = {}
        for t in (10, 100, 1000, 10000):
            below = [k for k, v in proj.items() if v < t]
            # 这些 token 占用的训练 token 槽位比例
            slot_share = sum(cnt[k] for k in below) / max(1, n_tok)
            thr[f'lt_{t}'] = {
                'num_tokens': len(below),
                'pct_of_vocab': len(below) / V,
                'pct_of_token_slots': slot_share,
            }
        results[label] = {
            'vocab_size': V,
            'eval_chars': n_chars,
            'eval_tokens': n_tok,
            'vocab_used': used,
            'vocab_util': used / V,
            'num_unused': V - used,
            'extrapolate_scale': scale,
            'thresholds': thr,
            'median_proj_occ': sorted(proj.values())[len(proj) // 2] if proj else 0,
        }
        r = results[label]
        print(f'\n[{label}] V={V} 用到={used} ({used/V:.1%}) 未用={V-used}')
        print(f'  评测 token 数={n_tok:,}  外推系数={scale:.1f}x (→{args.budget:.0e} tokens)')
        for t in (10, 100, 1000, 10000):
            d = thr[f'lt_{t}']
            print(f'  预计 <{t:>5} 次的 token: {d["num_tokens"]:>5} 个 ({d["pct_of_vocab"]:>6.1%} vocab) '
                  f'占总槽位 {d["pct_of_token_slots"]:>6.2%}')
        print(f'  中位数出现次数: {r["median_proj_occ"]:,.0f}')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {args.out}')


if __name__ == '__main__':
    main()
