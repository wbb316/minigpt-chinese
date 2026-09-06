# -*- coding: utf-8 -*-
"""画 v2 35M 完整训练曲线（E1+E2，60,915 步）：EMA train loss + val loss。

输入: log/35M参数+998Mtokens/step_history_train_webnovel_v2.csv + val_history
输出: result/35M参数+998Mtokens/training_curve_v2.png
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGD = os.path.join(ROOT, 'log', '35M参数+998Mtokens')
OUT = os.path.join(ROOT, 'result', '35M参数+998Mtokens', 'training_curve_v2.png')

step = pd.read_csv(os.path.join(LOGD, 'step_history_train_webnovel_v2.csv'))
val = pd.read_csv(os.path.join(LOGD, 'val_history_train_webnovel_v2.csv'))

# val 行也以 step 行形式存在（train_loss 列 NaN）→ 合并画点
train_rows = step[step['train_loss'].notna()]

fig, ax = plt.subplots(figsize=(11, 6))
# EMA train loss（每 200 步抽一个点，避免 6 万点糊成一团）
tr = train_rows.iloc[::200]
ax.plot(tr['step'], tr['train_loss'], lw=1.0, alpha=0.85,
        label='train loss (EMA, sampled)', color='#8888cc')
# val loss 点
ax.plot(val['step'], val['val'], 'o-', lw=1.6, ms=6,
        label='val loss', color='#d9534f')
# E1/E2 分界
ax.axvline(30463, color='gray', ls='--', lw=1, alpha=0.7)
ax.text(30463, ax.get_ylim()[1] if False else 0.99, 'E1/E2',
        transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9,
        color='gray')
# best val 标注
b = val.loc[val['val'].idxmin()]
ax.annotate(f"best val {b['val']:.4f}\nstep {int(b['step'])}",
            xy=(b['step'], b['val']), xytext=(b['step'] - 8000, b['val'] + 0.25),
            arrowprops=dict(arrowstyle='->', color='#333'), fontsize=10,
            color='#333')
for _, v in val.iterrows():
    if v['is_best'] == 1:
        ax.plot(v['step'], v['val'], 'r*', ms=11)

ax.set_xlabel('global step (E1: 1-30463, E2: 30464-60915)')
ax.set_ylabel('loss (nats)')
ax.set_title('v2 35M training curve — webnovel_v2 shard0 x2 epochs (2B tokens)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=130)
print('saved:', OUT)
print(f'train rows: {len(train_rows)}, val points: {len(val)}, best val {b["val"]:.4f} @ step {int(b["step"])}')
