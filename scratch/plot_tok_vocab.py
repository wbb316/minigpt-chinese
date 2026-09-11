# -*- coding: utf-8 -*-
"""tokenizer vocab_size 消融（第二阶段）汇总图。

读 eval_vocab.json（held-out benchmark 指标）+ tail_vocab.json（40M 字符频率长尾），画 6 面板：
  ① 压缩率 chars/token（B1/B2/B3）
  ② 固定 400K 字符所需 token 数
  ③ vocab 利用率：小样本 benchmark vs 40M 大样本
  ④ 未被使用的 token 数
  ⑤ 边际收益：每 +2048 vocab 省下的 token（递减）
  ⑥ 长尾：1B token 预算下预计出现次数偏少的 token 占比
输出: tok_exp/tok_vocab_summary.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, 'tok_exp', 'eval_vocab.json')
TAIL = os.path.join(ROOT, 'tok_exp', 'tail_vocab.json')
OUT = os.path.join(ROOT, 'tok_exp', 'tok_vocab_summary.png')

with open(EVAL, encoding='utf-8') as f:
    ev = json.load(f)
with open(TAIL, encoding='utf-8') as f:
    tl = json.load(f)

LABELS = ['v4096', 'v6144', 'v8192']
VOCABS = [4096, 6144, 8192]
X = range(len(VOCABS))
xlab = [str(v) for v in VOCABS]

fig, axes = plt.subplots(2, 3, figsize=(19, 10))

# ① 压缩率
ax = axes[0][0]
for key, name, mk in [('B1', 'B1 20 books', 'o-'), ('B2', 'B2 uniform', 's-'), ('B3', 'B3 boundary', '^-')]:
    ys = [ev[l][key]['chars_per_token'] for l in LABELS]
    ax.plot(X, ys, mk, lw=2, ms=8, label=name)
    for x, y in zip(X, ys):
        ax.annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=8)
ax.set_xticks(list(X)); ax.set_xticklabels(xlab)
ax.set_xlabel('vocab size'); ax.set_ylabel('chars / token  (higher = better)')
ax.set_title('① Compression vs vocab size')
ax.grid(alpha=.3); ax.legend(fontsize=9)

# ② 固定 400K 字符所需 token 数
ax = axes[0][1]
ys = [ev[l]['B2']['tokens'] for l in LABELS]
bars = ax.bar([str(v) for v in VOCABS], ys, color=['#8fbcd4', '#5b8fb9', '#2f5f8f'])
base = ys[0]
for b, y in zip(bars, ys):
    ax.annotate(f'{y:,}\n({(y/base-1)*100:+.2f}%)', (b.get_x() + b.get_width() / 2, y),
                textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9)
ax.set_ylim(0, max(ys) * 1.22)
ax.set_xlabel('vocab size'); ax.set_ylabel('tokens for 400K chars (B2)')
ax.set_title('② Fewer tokens = more text per token budget')
ax.grid(alpha=.3, axis='y')

# ③ vocab 利用率（小 benchmark vs 40M 大样本）
ax = axes[0][2]
small = [ev[l]['B2']['vocab_util'] * 100 for l in LABELS]
large = [tl[l]['vocab_util'] * 100 for l in LABELS]
w = 0.38
ax.bar([i - w / 2 for i in X], small, w, label='B2 (400K chars)', color='#e0a06a')
ax.bar([i + w / 2 for i in X], large, w, label='40M chars eval', color='#7a9e6b')
for i, (s, l) in enumerate(zip(small, large)):
    ax.annotate(f'{s:.1f}%', (i - w / 2, s), textcoords='offset points', xytext=(0, 3),
                ha='center', fontsize=8)
    ax.annotate(f'{l:.1f}%', (i + w / 2, l), textcoords='offset points', xytext=(0, 3),
                ha='center', fontsize=8)
ax.set_xticks(list(X)); ax.set_xticklabels(xlab)
ax.set_ylim(0, 108)
ax.set_xlabel('vocab size'); ax.set_ylabel('vocab utilization (%)')
ax.set_title('③ Vocabulary utilization')
ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=9)

# ④ 未使用 token 数
ax = axes[1][0]
uns_small = [ev[l]['vocab_size'] - ev[l]['B2']['vocab_used'] for l in LABELS]
uns_large = [tl[l]['num_unused'] for l in LABELS]
ax.bar([i - w / 2 for i in X], uns_small, w, label='B2 (400K chars)', color='#c98b8b')
ax.bar([i + w / 2 for i in X], uns_large, w, label='40M chars eval', color='#9a6b9e')
for i, (s, l) in enumerate(zip(uns_small, uns_large)):
    ax.annotate(f'{s}', (i - w / 2, s), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=8)
    ax.annotate(f'{l}', (i + w / 2, l), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=8)
ax.set_xticks(list(X)); ax.set_xticklabels(xlab)
ax.set_xlabel('vocab size'); ax.set_ylabel('unused tokens (count)')
ax.set_title('④ Unused tokens (small sample overestimates)')
ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=9)

# ⑤ 边际收益
ax = axes[1][1]
d1 = ev['v4096']['B2']['tokens'] - ev['v6144']['B2']['tokens']
d2 = ev['v6144']['B2']['tokens'] - ev['v8192']['B2']['tokens']
bars = ax.bar(['4096→6144\n(+2048)', '6144→8192\n(+2048)'], [d1, d2], color=['#5b8fb9', '#a8c4dc'])
for b, y in zip(bars, [d1, d2]):
    ax.annotate(f'{y:,} tokens\n({y/d1:.1%} of 1st block)', (b.get_x() + b.get_width() / 2, y),
                textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9)
ax.set_ylim(0, d1 * 1.3)
ax.set_ylabel('tokens saved on B2 (400K chars)')
ax.set_title('⑤ Diminishing marginal gain per +2048 vocab')
ax.grid(alpha=.3, axis='y')

# ⑥ 长尾
ax = axes[1][2]
for key, name, mk in [('lt_1000', '<1000 occurrences', 'o-'), ('lt_10000', '<10000 occurrences', 's-')]:
    ys = [tl[l]['thresholds'][key]['pct_of_vocab'] * 100 for l in LABELS]
    ax.plot(X, ys, mk, lw=2, ms=8, label=name)
    for x, y in zip(X, ys):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
ax.set_xticks(list(X)); ax.set_xticklabels(xlab)
ax.set_xlabel('vocab size')
ax.set_ylabel('% of vocab (projected, 1B-token budget)')
ax.set_title('⑥ Tail: tokens with few projected updates')
ax.grid(alpha=.3); ax.legend(fontsize=9)
ax.set_ylim(0, max(tl[l]['thresholds']['lt_10000']['pct_of_vocab'] * 100 for l in LABELS) * 1.35)

fig.suptitle('Tokenizer vocabulary-size ablation (sample=4M chars fixed, held-out benchmark)',
             fontsize=14, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT, dpi=130)
print('图已保存:', OUT)
