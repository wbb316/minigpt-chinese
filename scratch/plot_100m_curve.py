# -*- coding: utf-8 -*-
"""100M + shard1+2 (2B) 训练曲线图。

读归档到 log/100M参数_v3_2Btokens/ 的两个 CSV，画：
  ① train loss（按 log 采样的 EMA 走势）
  ② val loss（12 个验证点）
  ③ 两者同图 + gap
输出: log/100M参数_v3_2Btokens/training_curve_100m.png
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAND = os.path.join(ROOT, 'log', '100M参数_v3_2Btokens')
STEP = os.path.join(LAND, 'step_history_train_webnovel_shard12.csv')
VAL = os.path.join(LAND, 'val_history_train_webnovel_shard12.csv')
OUT = os.path.join(LAND, 'training_curve_100m.png')

steps, tloss, lrs = [], [], []
with open(STEP, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        try:
            s = int(r['step'])
            tl = r.get('train_loss', '').strip()
            lr = r.get('lr', '').strip()
            if tl:
                steps.append(s)
                tloss.append(float(tl))
                lrs.append(float(lr) if lr else float('nan'))
        except (KeyError, ValueError):
            continue

vsteps, vval, vtrain = [], [], []
with open(VAL, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        try:
            vsteps.append(int(r['step']))
            vval.append(float(r['val']))
            te = r.get('train_eval', '').strip()
            vtrain.append(float(te) if te else float('nan'))
        except (KeyError, ValueError):
            continue

print(f'step 点 {len(steps)}  val 点 {len(vsteps)}')
print('val 曲线:', ', '.join(f'{s}:{v:.4f}' for s, v in zip(vsteps, vval)))

fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))

# ① train loss
ax = axes[0]
ax.plot(steps, tloss, lw=1.2, color='#2f5f8f')
ax.set_xlabel('step'); ax.set_ylabel('train loss (logged EMA)')
ax.set_title('(1) train loss')
ax.grid(alpha=.3)
ax.set_ylim(min(tloss) * 0.98, max(tloss) * 1.02)

# ② val loss
ax = axes[1]
ax.plot(vsteps, vval, 'o-', lw=2, ms=7, color='#c0562c')
for s, v in zip(vsteps, vval):
    ax.annotate(f'{v:.4f}', (s, v), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=8)
ax.set_xlabel('step'); ax.set_ylabel('val loss (nats/token)')
ax.set_title('(2) val loss  [12 checkpoints]')
ax.grid(alpha=.3)

# ③ val + train_eval (gap)
ax = axes[2]
ax.plot(vsteps, vval, 'o-', lw=2, ms=6, color='#c0562c', label='val')
ax.plot(vsteps, vtrain, 's--', lw=1.5, ms=5, color='#7a9e6b',
        label='train_eval (no dropout)')
ax.set_xlabel('step'); ax.set_ylabel('loss')
ax.set_title('(3) val vs train_eval  [gap = overfit indicator]')
ax.grid(alpha=.3); ax.legend()

fig.suptitle('MiniGPT-Chinese  100M (16L/704d/11H, vocab 8192) + shard1+2 (1.96B tokens, 1 epoch)'
             f'  —  best val {min(vval):.4f}', fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=130)
print('图已保存:', OUT)
