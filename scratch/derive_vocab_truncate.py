# -*- coding: utf-8 -*-
"""从大 vocab tokenizer 截取派生小 vocab tokenizer（BPE 前缀性质）。

原理：byte-level BPE 的合并是**顺序贪心**——第 k 次合并由当前 pair counts 决定，
与目标 vocab_size 无关（只要不提前停止）。id 按 `len(vocab)` 递增分配，
merges[(a,b)] = nid 且 nid 连续。

因此 vocab=M 的 tokenizer ≡ vocab=N（N>M）的「前 M 个 token + nid<M 的 merge 规则」。

用法:
  python scratch/derive_vocab_truncate.py --src tok_v8192.pkl --targets 4096 6144 \
      --outdir tok_exp/tok --baseline tok_4M.pkl:6144
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer import BPETokenizer  # noqa: E402


def truncate(tok, target):
    """按 id 截断 vocab 与 merges，返回新的 BPETokenizer。"""
    new = BPETokenizer(vocab_size=target)
    new.vocab = {k: v for k, v in tok.vocab.items() if k < target}
    new.merges = {k: v for k, v in tok.merges.items() if v < target}
    return new


def compare(a, b):
    """比较两个 tokenizer 是否逐位一致，返回 (是否一致, 差异描述列表)。"""
    diffs = []
    if set(a.vocab.keys()) != set(b.vocab.keys()):
        diffs.append(f'vocab key 集合不同: |a|={len(a.vocab)} |b|={len(b.vocab)}')
    else:
        bad = [k for k in a.vocab if a.vocab[k] != b.vocab[k]]
        if bad:
            diffs.append(f'vocab 内容不同: {len(bad)} 个 id（例 {bad[:5]}）')
    if set(a.merges.keys()) != set(b.merges.keys()):
        only_a = len(set(a.merges) - set(b.merges))
        only_b = len(set(b.merges) - set(a.merges))
        diffs.append(f'merges 规则集不同: 仅a有={only_a} 仅b有={only_b}')
    else:
        bad = [k for k in a.merges if a.merges[k] != b.merges[k]]
        if bad:
            diffs.append(f'merges 映射不同: {len(bad)} 条（例 {bad[:3]}）')
    return len(diffs) == 0, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='源 tokenizer pkl（大 vocab）')
    ap.add_argument('--targets', nargs='+', type=int, required=True)
    ap.add_argument('--outdir', default='tok_exp/tok')
    ap.add_argument('--baseline', default='', help='独立训练的同 vocab tokenizer，格式 path:vocab，用于一致性验证')
    args = ap.parse_args()

    src = pickle.load(open(args.src, 'rb'))
    print(f'源 tokenizer: vocab_size={src.vocab_size} 实际 vocab={len(src.vocab)} merges={len(src.merges)}')

    for t in args.targets:
        if t > src.vocab_size:
            print(f'⚠️ 跳过 {t}（大于源 vocab {src.vocab_size}）')
            continue
        new = truncate(src, t)
        out = os.path.join(args.outdir, f'tok_v{t}_from{src.vocab_size}.pkl')
        os.makedirs(args.outdir, exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump(new, f)
        print(f'派生 vocab={t}: vocab {len(new.vocab)} 条, merges {len(new.merges)} 条 → {out}')

    # 与独立训练结果对比
    if args.baseline:
        bpath, bvocab = args.baseline.rsplit(':', 1)
        bvocab = int(bvocab)
        if os.path.exists(bpath):
            base = pickle.load(open(bpath, 'rb'))
            derived = truncate(src, bvocab)
            same, diffs = compare(derived, base)
            print()
            print(f'=== 一致性验证: 截取 vocab={bvocab} vs 独立训练 {os.path.basename(bpath)} ===')
            if same:
                print('✅ 完全一致 —— 截取方案成立（BPE 前缀性质验证通过）')
            else:
                print('❌ 存在差异：')
                for d in diffs:
                    print('   -', d)
        else:
            print(f'⚠️ baseline 不存在，跳过验证: {bpath}')


if __name__ == '__main__':
    main()
