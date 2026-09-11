# -*- coding: utf-8 -*-
"""tokenizer sample-size 实验：BPE 训练速度实测 + 抽样逻辑。

抽样与 train.py 完全一致（100 段均匀撒，覆盖全语料，不截取前 N 字符）：
    seg_len = tokens_sample // 100
    step    = len(text) // 100
    parts   = [text[s:s+seg_len] for s in range(0, len(text), step)][:100]

用法: python scratch/tok_sample_experiment.py --sample 4000000 --backend fast
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tokenizer import BPETokenizer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, 'data', 'train_webnovel_v2.txt')


def load_and_sample(path, tokens_sample):
    """按 train.py 的 100 段均匀撒逻辑抽样（流式 seek，避免整读 3.5GB）。

    ★ 关键：按【字符数】精确控制样本量。做法是先读一小段估算 bytes/char，
    再据此换算每段应读的字节数，读完 decode 后按字符数截断，保证实得字符数
    精确等于 tokens_sample（误差 <1%）。
    """
    size = os.path.getsize(path)
    with open(path, 'rb') as fb:                        # ★ 全程二进制：seek/read 都是字节语义
        probe = fb.read(3_000_000)
        bpc = len(probe) / max(1, len(probe.decode('utf-8', errors='ignore')))
        seg_chars = max(1, tokens_sample // 100)
        seg_bytes = int(seg_chars * bpc) + 32           # 多读一点，decode 后按字符截断
        step_bytes = max(1, size // 100)
        parts = []
        for i in range(100):
            fb.seek(i * step_bytes)
            raw = fb.read(seg_bytes)
            chunk = raw.decode('utf-8', errors='ignore')
            parts.append(chunk[:seg_chars])             # ★ 精确控量
        text = ''.join(parts)
    print(f'  [抽样] bytes/char={bpc:.3f} 目标={tokens_sample:,} 实得={len(text):,} 字符')
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=4_000_000)
    ap.add_argument('--backend', default='fast')
    ap.add_argument('--vocab-size', type=int, default=6144)
    ap.add_argument('--out', default='')
    ap.add_argument('--corpus', default=CORPUS, help='训练语料路径（云端默认用 --corpus 指定）')
    args = ap.parse_args()

    t0 = time.time()
    text = load_and_sample(args.corpus, args.sample)
    t_load = time.time() - t0
    print(f'抽样: 目标 {args.sample:,} 字符 → 实得 {len(text):,} 字符, 耗时 {t_load:.1f}s', flush=True)

    tok = BPETokenizer(vocab_size=args.vocab_size)
    t1 = time.time()
    tok.train(text, backend=args.backend)
    t_train = time.time() - t1
    print(f'BPE 训练完成: vocab={len(tok.vocab)} backend={args.backend} 耗时 {t_train:.1f}s '
          f'({t_train/60:.2f} min)', flush=True)

    if args.out:
        import pickle
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'wb') as f:
            pickle.dump(tok, f)
        print(f'已保存: {args.out}', flush=True)

    print(f'RESULT sample={args.sample} load={t_load:.1f}s train={t_train:.1f}s '
          f'total={t_load+t_train:.1f}s vocab={len(tok.vocab)}', flush=True)


if __name__ == '__main__':
    main()
