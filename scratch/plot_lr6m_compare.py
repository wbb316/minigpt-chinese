# -*- coding: utf-8 -*-
"""6M LR schedule 筛选实验对比图（5 组 from-scratch，百合 20M 单遍 5144 步）

val CSV 是 5 组共享追加文件，按 wall_time 升序每 21 点切一组（每组 step 250..5144）。
输出：
  - val 曲线对比（5 组同图）
  - lr 曲线对比（5 组同图）
  到 log/6M参数+20Mtokens_LR实验/
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'log', '6M参数+20Mtokens_LR实验')
val = pd.read_csv(os.path.join(D, 'val_history_train.csv'))

# 5 组，每组 21 个点（step 250..5144, 每 250 一步）
GROUPS = ['S1 cosine-fast', 'S2 cosine-slow', 'S3 WSD',
          'S4 hold-decay', 'S5 const-tail']
COLORS = ['#888888', '#5b9bd5', '#d9534f', '#70ad47', '#c55a11']
N = 21

groups = []
for i, name in enumerate(GROUPS):
    g = val.iloc[i * N:(i + 1) * N].copy().reset_index(drop=True)
    groups.append(g)
    print(f'{name}: {len(g)} 点, best {g["val"].min():.4f} @ step {g.loc[g["val"].idxmin(), "step"]}')

# ---- 图 1: val 曲线 ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for g, name, c in zip(groups, GROUPS, COLORS):
    ax1.plot(g['step'], g['val'], 'o-', lw=1.8, ms=3.5, color=c, label=f'{name} (best {g["val"].min():.4f})')
ax1.set_ylabel('val loss (nats)')
ax1.set_title('6M LR schedule from-scratch — baihe 20M single pass (6L/256d/vocab6144, seed42)')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(alpha=0.3)

for g, name, c in zip(groups, GROUPS, COLORS):
    ax2.plot(g['step'], g['lr'], '-', lw=1.6, color=c, label=name)
ax2.set_xlabel('step (4096 tok/step, ~21M token total)')
ax2.set_ylabel('learning rate')
ax2.set_ylim(0, 0.0009)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(alpha=0.3)
plt.tight_layout()
out1 = os.path.join(D, 'lr6m_val_compare.png')
plt.savefig(out1, dpi=130)
print('saved:', out1)

# ---- 图 2: 每百万 token 改善速度（分段斜率） ----
# 用 val 每 250 步(≈1.024M token) 的下降量，中段(1000-4000步)平均斜率
fig2, ax = plt.subplots(figsize=(12, 5))
for g, name, c in zip(groups, GROUPS, COLORS):
    mid = g[(g['step'] >= 1000) & (g['step'] <= 4000)]
    slope = (mid['val'].iloc[0] - mid['val'].iloc[-1]) / ((mid['step'].iloc[-1] - mid['step'].iloc[0]) / 1000)
    ax.bar(name, slope, color=c, alpha=0.8)
    ax.text(name, slope + 0.002, f'{slope:.4f}', ha='center', fontsize=9)
ax.set_ylabel('val improvement per 1000 steps (≈4.1M token)')
ax.set_title('Improvement rate in mid-training (step 1000→4000) — higher = faster learning')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out2 = os.path.join(D, 'lr6m_slope_compare.png')
plt.savefig(out2, dpi=130)
print('saved:', out2)
