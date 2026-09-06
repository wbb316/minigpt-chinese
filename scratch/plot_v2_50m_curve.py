# -*- coding: utf-8 -*-
"""画 v2 50M 训练曲线（单轮 30,454 步）：EMA train loss + val loss → result/50M参数+998Mtokens/
仿 35M 的 result/training_curve_v2.png 风格。
⚠️ step_history CSV 混有 35M(E1/E2) 行 —— 按 wall_time >= 2026-09-06 20:00 过滤出 50M 部分。
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGD = os.path.join(ROOT, 'log', '50M参数+998Mtokens')
OUT = os.path.join(ROOT, 'result', '50M参数+998Mtokens', 'training_curve_v2.png')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

T0 = '2026-09-06 20:00'   # 50M run 起点（35M E2 结束于 13:49，互不重叠）

step = pd.read_csv(os.path.join(LOGD, 'step_history_train_webnovel_v2.csv'))
val = pd.read_csv(os.path.join(LOGD, 'val_history_train_webnovel_v2.csv'))

# 只保留 50M 的行（时间过滤）
step = step[step['time'] >= T0]
val = val[val['wall_time'] >= T0]
# 去掉 step 0 的初始随机验证点(368, 假信号)与 train_loss NaN 的 val 行
val = val[val['step'] > 0]
train_rows = step[step['train_loss'].notna()]

fig, ax = plt.subplots(figsize=(11, 6))
tr = train_rows.iloc[::200]
ax.plot(tr['step'], tr['train_loss'], lw=1.0, alpha=0.85,
        label='train loss (EMA, sampled)', color='#8888cc')
ax.plot(val['step'], val['val'], 'o-', lw=1.6, ms=6,
        label='val loss', color='#d9534f')
b = val.loc[val['val'].idxmin()]
ax.annotate(f"best val {b['val']:.4f}\nstep {int(b['step'])}",
            xy=(b['step'], b['val']), xytext=(b['step'] - 8000, b['val'] + 0.25),
            arrowprops=dict(arrowstyle='->', color='#333'), fontsize=10, color='#333')
for _, v in val.iterrows():
    if v['is_best'] == 1:
        ax.plot(v['step'], v['val'], 'r*', ms=11)

ax.set_xlabel('global step (single epoch, 1B tokens)')
ax.set_ylabel('loss (nats)')
ax.set_title('v2 50M training curve — webnovel_v2 shard0 (12L/576d/9H, 51.4M)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=130)
print('saved:', OUT)
print(f'50M steps: {len(train_rows)}, val points: {len(val)}, best val {b["val"]:.4f} @ step {int(b["step"])}')
