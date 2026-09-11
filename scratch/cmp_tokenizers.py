# -*- coding: utf-8 -*-
"""比较两个 tokenizer pkl 是否逐位等价（vocab + merges 全量）。

用途：确认实验用的 tokenizer 与生产用的确实是同一个（跨机器/跨流程对比前的自检）。

用法:
  python scratch/cmp_tokenizers.py a.pkl b.pkl [c.pkl ...]
  # 以第一个为基准，逐个比较并打印差异
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    base = load(sys.argv[1])
    print(f'基准: {sys.argv[1]}  vocab={len(base.vocab)} merges={len(base.merges)}')

    ok = True
    for path in sys.argv[2:]:
        t = load(path)
        same_v = base.vocab == t.vocab
        same_m = base.merges == t.merges
        flag = '✅ 完全一致' if (same_v and same_m) else '❌ 不一致'
        print(f'{flag}: {path}  vocab={len(t.vocab)} merges={len(t.merges)}')
        if not (same_v and same_m):
            ok = False
            if not same_v:
                diff = [k for k in set(base.vocab) | set(t.vocab)
                        if base.vocab.get(k) != t.vocab.get(k)]
                print(f'   vocab 差异 {len(diff)} 项，前 5: {sorted(diff)[:5]}')
            if not same_m:
                diff = [k for k in set(base.merges) | set(t.merges)
                        if base.merges.get(k) != t.merges.get(k)]
                print(f'   merges 差异 {len(diff)} 项，前 5: {diff[:5]}')
    print('\n总结:', '✅ 全部一致' if ok else '❌ 存在差异')
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
