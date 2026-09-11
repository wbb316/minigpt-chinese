# -*- coding: utf-8 -*-
"""vocab 对照（6144 vs 8192，各 5000 步）结果汇总图与对比表。

关键点：per-token val_loss 跨 vocab **不可比**（词表越大，单 token 信息越少，
loss 天然更低）。可比量是 bits/char 与 bits/byte：

    bits/char = (val_loss_nats / ln2) × (n_val_tokens / n_val_chars)
              = (val_loss_nats / ln2) / chars_per_token

其中 chars_per_token 用**完整 val 语料**上的实测值（不是小 benchmark）。

用法:
  python scratch/plot_vocab_cmp.py \
    --csv log/vocab对照6144vs8192/val_history_v6144.csv \
    --csv log/vocab对照6144vs8192/val_history_v8192.csv \
    --eval log/vocab对照6144vs8192/eval_v6144.json log/vocab对照6144vs8192/eval_v8192.json \
    --out log/vocab对照6144vs8192/vocab_cmp_summary.png
"""
import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

LN2 = math.log(2.0)


def read_val_csv(path):
    """train.py 的 val_history CSV → [(step, val)]，只取步级验证行。

    用 utf-8-sig 读：文件可能被 Windows 工具（PowerShell Set-Content 等）加上 BOM，
    否则首列名会变成 '\\ufeffstep' 导致所有行静默解析失败。
    """
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                step = int(r['step'])
                val = float(r['val'])
            except (KeyError, ValueError, TypeError):
                continue
            rows.append((step, val))
    # 同一步可能因 resume 重复出现 → 保留最后一个
    dedup = {}
    for s, v in rows:
        dedup[s] = v
    return sorted(dedup.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs=2, required=True, metavar=('CSV6144', 'CSV8192'))
    ap.add_argument('--eval', nargs=2, required=True, metavar=('JSON6144', 'JSON8192'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    curves = {}
    for tag, p in zip(('vocab 6144', 'vocab 8192'), args.csv):
        curves[tag] = read_val_csv(p)
        print(f'{tag}: {len(curves[tag])} 个验证点  {p}')

    evals = {}
    for tag, p in zip(('vocab 6144', 'vocab 8192'), args.eval):
        with open(p, encoding='utf-8-sig') as f:
            evals[tag] = json.load(f)
        e = evals[tag]
        print(f'{tag}: vocab={e["vocab_size"]} val_loss={e["val_loss_nats_per_token"]:.4f} '
              f'bits/char={e["bits_per_char"]:.4f} bits/byte={e["bits_per_byte"]:.4f} '
              f'chars/token={e["chars_per_token"]:.4f}')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = {'vocab 6144': '#2f5f8f', 'vocab 8192': '#c0562c'}

    # ① per-token val_loss（不可跨 vocab 比较）
    ax = axes[0][0]
    for tag, pts in curves.items():
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', lw=2, ms=5,
                    color=colors[tag], label=tag)
    ax.set_xlabel('step'); ax.set_ylabel('val_loss (nats per TOKEN)')
    ax.set_title('(1) per-token val loss  [NOT comparable across vocab]')
    ax.grid(alpha=.3); ax.legend()

    # ② bits/char（可比）
    ax = axes[0][1]
    for tag, pts in curves.items():
        if not pts:
            continue
        cpt = evals[tag]['chars_per_token']
        ax.plot([p[0] for p in pts], [p[1] / LN2 / cpt for p in pts], 'o-', lw=2, ms=5,
                color=colors[tag], label=f'{tag} (chars/tok {cpt:.4f})')
    ax.set_xlabel('step'); ax.set_ylabel('bits per CHARACTER  (lower = better)')
    ax.set_title('(2) bits/char  [comparable - the decisive metric]')
    ax.grid(alpha=.3); ax.legend()

    # ③ Δ(bits/char) = 8192 − 6144
    ax = axes[1][0]
    a = dict(curves['vocab 6144'])
    b = dict(curves['vocab 8192'])
    common = sorted(set(a) & set(b))
    if common:
        cpt_a = evals['vocab 6144']['chars_per_token']
        cpt_b = evals['vocab 8192']['chars_per_token']
        d = [(s, b[s] / LN2 / cpt_b - a[s] / LN2 / cpt_a) for s in common]
        ax.axhline(0, color='k', lw=1)
        ax.plot([x[0] for x in d], [x[1] for x in d], 'o-', lw=2, ms=6, color='#7a4fa0')
        for s, v in d:
            ax.annotate(f'{v:+.4f}', (s, v), textcoords='offset points',
                        xytext=(0, 8 if v >= 0 else -14), ha='center', fontsize=8)
        win = '8192 better' if d[-1][1] < 0 else '6144 better'
        ax.set_title(f'(3) Delta bits/char (8192 - 6144) -> negative = {win}')
    else:
        ax.set_title('(3) Delta bits/char (no common step)')
    ax.set_xlabel('step'); ax.set_ylabel('Δ bits/char')
    ax.grid(alpha=.3)

    # ④ 全量 val 终值对比
    ax = axes[1][1]
    tags = ['vocab 6144', 'vocab 8192']
    bc = [evals[t]['bits_per_char'] for t in tags]
    bb = [evals[t]['bits_per_byte'] for t in tags]
    x = [0, 1]
    w = 0.35
    ax.bar([i - w / 2 for i in x], bc, w, label='bits/char', color='#2f5f8f')
    ax.bar([i + w / 2 for i in x], bb, w, label='bits/byte', color='#c0562c')
    for i, (v1, v2) in enumerate(zip(bc, bb)):
        ax.annotate(f'{v1:.4f}', (i - w / 2, v1), textcoords='offset points',
                    xytext=(0, 3), ha='center', fontsize=9)
        ax.annotate(f'{v2:.4f}', (i + w / 2, v2), textcoords='offset points',
                    xytext=(0, 3), ha='center', fontsize=9)
    lo = min(bc + bb) * 0.97
    ax.set_ylim(lo, max(bc + bb) * 1.06)
    ax.set_xticks(x); ax.set_xticklabels([f'{t}\n(V={evals[t]["vocab_size"]})' for t in tags])
    ax.set_ylabel('bits (lower = better)')
    ax.set_title('(4) Full-val final: normalised metrics (step 5000)')
    ax.grid(alpha=.3, axis='y'); ax.legend()

    dchar = bc[1] - bc[0]
    dbyte = bb[1] - bb[0]
    fig.suptitle(f'vocab ablation LM check - 50M/12L/576d/9H, 5000 steps of the production '
                 f'recipe\ndelta bits/char {dchar:+.4f} ({dchar/bc[0]*100:+.2f}%)   '
                 f'delta bits/byte {dbyte:+.4f} ({dbyte/bb[0]*100:+.2f}%)',
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print('\n图已保存:', args.out)

    print('\n' + '=' * 92)
    print(f'{"arm":<12} {"vocab":>6} {"val_loss":>10} {"bits/tok":>9} {"bits/char":>10} '
          f'{"bits/byte":>10} {"chars/tok":>10} {"train_tok":>12}')
    print('-' * 92)
    for t in tags:
        e = evals[t]
        print(f'{t:<12} {e["vocab_size"]:>6} {e["val_loss_nats_per_token"]:>10.4f} '
              f'{e["val_loss_bits_per_token"]:>9.4f} {e["bits_per_char"]:>10.4f} '
              f'{e["bits_per_byte"]:>10.4f} {e["chars_per_token"]:>10.4f} '
              f'{e["n_val_tokens"]:>12,}')
    print('=' * 92)
    print(f'Δ bits/char = {dchar:+.4f} ({dchar/bc[0]*100:+.2f}%)  '
          f'Δ bits/byte = {dbyte:+.4f} ({dbyte/bb[0]*100:+.2f}%)')
    print('负值 = 8192 更好。')


if __name__ == '__main__':
    main()
