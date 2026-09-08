# -*- coding: utf-8 -*-
"""v3 vs v2 50M 同点对比曲线：val loss 按 step 对齐（同结构同数据，底层 rope+init vs sinusoidal+旧init）
→ log/50M参数_v3_alpha+998Mtokens/v3_vs_v2_val_compare.png
数据源：v2 50M E1 (2026-09-06 20:00-22:23) 与 v3 (2026-09-08 16:55-18:08)，同一共享 val CSV。
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGD = os.path.join(ROOT, 'log', '50M参数_v3_alpha+998Mtokens')
OUT = os.path.join(LOGD, 'v3_alpha_vs_v2_val_compare.png')

# v2 50M 归档里的 val CSV 只有截至当时的共享历史（含 v2 50M 段）
V2VAL = os.path.join(ROOT, 'log', '50M参数_v2+998Mtokens', 'val_history_train_webnovel_v2.csv')
val = pd.read_csv(os.path.join(LOGD, 'val_history_train_webnovel_v3.csv'))

T_V2 = '2026-09-06 20:00'   # v2 50M run 起点
T_V3 = '2026-09-08 16:55'   # v3 run 起点

v2 = val[(val['wall_time'] >= T_V2) & (val['wall_time'] < '2026-09-06 23:00')]
v3 = val[(val['wall_time'] >= T_V3) & (val['wall_time'] < '2026-09-08 19:00')]
v2 = v2[v2['step'] > 0].reset_index(drop=True)
v3 = v3[v3['step'] > 0].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(v2['step'], v2['val'], 's-', lw=1.5, ms=6,
        label='v2 50M (sinusoidal + old init): 3.6154', color='#b0b0b0')
ax.plot(v3['step'], v3['val'], 'o-', lw=1.8, ms=6,
        label='v3 50M (RoPE + GPT-2 init): 3.2097', color='#d9534f')

b2 = v2.loc[v2['val'].idxmin()]
b3 = v3.loc[v3['val'].idxmin()]
ax.annotate(f"v2 best {b2['val']:.4f}", xy=(b2['step'], b2['val']),
            xytext=(b2['step'] - 9000, b2['val'] + 0.28),
            arrowprops=dict(arrowstyle='->', color='#888'), fontsize=9, color='#666')
ax.annotate(f"v3 best {b3['val']:.4f}  (−0.406 nats)", xy=(b3['step'], b3['val']),
            xytext=(b3['step'] - 9000, b3['val'] + 0.18),
            arrowprops=dict(arrowstyle='->', color='#d9534f'), fontsize=10, color='#d9534f')

# 同点差值标注（终点）
last = v3.iloc[-1]
gap_v = b2['val'] - last['val']
ax.set_xlabel('step (same data, single epoch 1B tokens)')
ax.set_ylabel('val loss (nats)')
ax.set_title('v3 vs v2 50M — same arch & data, bottom-layer upgrade (RoPE + GPT-2 init)')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=130)
print('saved:', OUT)
print(f'v2 points: {len(v2)} (best {b2["val"]:.4f}), v3 points: {len(v3)} '
      f'(best {b3["val"]:.4f}), 终差 {gap_v:.4f} nats')
