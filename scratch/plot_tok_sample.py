# -*- coding: utf-8 -*-
"""tokenizer sample-size 实验汇总图：样本量 vs 各项指标。

读 eval_tokenizers.py 输出的 JSON（含 4M/8M/16M/32M），画：
  ① 压缩率 chars/token（B1/B2/B3 三档）
  ② vocab 利用率 + top100 覆盖
  ③ 分书压缩率标准差（跨书泛化）
  ④ encode/decode 速度
输出: tok_exp/tok_sample_summary.png
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(ROOT, 'tok_exp', 'eval_all.json')
OUT = os.path.join(ROOT, 'tok_exp', 'tok_sample_summary.png')

with open(RESULT, encoding='utf-8') as f:
    data = json.load(f)

# 按样本量排序
def size_of(label):
    m = re.match(r'(\d+)M', label.replace('tok_', '').replace('_', ''))
    return int(m.group(1)) if m else 0

labels = sorted(data.keys(), key=size_of)
sizes = [size_of(l) for l in labels]
print('标签:', labels, '样本量(M):', sizes)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# ① 压缩率 chars/token
ax = axes[0][0]
for key, name in [('B1', 'B1 20 books'), ('B2', 'B2 uniform'), ('B3', 'B3 boundary')]:
    ys = [data[l][key]['chars_per_token'] for l in labels]
    ax.plot(sizes, ys, 'o-', lw=2, ms=7, label=name)
ax.set_xlabel('tokenizer training sample size (M chars)')
ax.set_ylabel('chars / token  (higher = better compression)')
ax.set_title('① Compression vs sample size')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(sizes)

# ② vocab 利用率 + top100 覆盖
ax = axes[0][1]
ys1 = [data[l]['B2']['vocab_util'] * 100 for l in labels]
ys2 = [data[l]['B2']['top100_coverage'] * 100 for l in labels]
ax.plot(sizes, ys1, 'o-', lw=2, ms=7, color='#d9534f', label='vocab utilization %')
ax.plot(sizes, ys2, 's-', lw=2, ms=7, color='#5b9bd5', label='top-100 coverage %')
ax.set_xlabel('sample size (M chars)')
ax.set_ylabel('%')
ax.set_title('② Vocabulary usage')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(sizes)

# ③ 分书压缩率标准差
ax = axes[1][0]
means = [data[l]['B1_per_book_mean'] for l in labels]
stds = [data[l]['B1_per_book_std'] for l in labels]
ax.errorbar(sizes, means, yerr=stds, fmt='o-', lw=2, ms=7, capsize=5, color='#70ad47')
ax.set_xlabel('sample size (M chars)')
ax.set_ylabel('chars/token (mean ± std over 20 books)')
ax.set_title('③ Cross-book generalization')
ax.grid(alpha=0.3)
ax.set_xticks(sizes)

# ④ 速度
ax = axes[1][1]
enc = [data[l]['B2']['encode_chars_per_s'] / 1000 for l in labels]
dec = [data[l]['B2']['decode_tokens_per_s'] / 1000 for l in labels]
ax.plot(sizes, enc, 'o-', lw=2, ms=7, color='#c55a11', label='encode (k chars/s)')
ax.plot(sizes, dec, 's-', lw=2, ms=7, color='#888', label='decode (k tokens/s)')
ax.set_xlabel('sample size (M chars)')
ax.set_ylabel('speed (k/s)')
ax.set_title('④ Encode / decode speed')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(sizes)

plt.suptitle('Tokenizer training sample size — 4M/8M/16M/32M (vocab6144, byte-level BPE, held-out val benchmark)',
             fontsize=12)
plt.tight_layout()
plt.savefig(OUT, dpi=130)
print('saved:', OUT)
