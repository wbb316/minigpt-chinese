# -*- coding: utf-8 -*-
"""画 benchmark/performance_results.csv → performance_summary.png（tokens/s by variant）。"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'performance_results.csv')
OUT = os.path.join(HERE, 'performance_summary.png')

rows = []
with open(CSV, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        try:
            rows.append((r['variant'], float(r['tokens_per_sec'])))
        except (ValueError, KeyError, TypeError):
            rows.append((r['variant'], None))
rows = [(v, t) for v, t in rows if t is not None]
if not rows:
    print('no data in', CSV)
    raise SystemExit(1)

names = [v for v, _ in rows]
tps = [t for _, t in rows]
colors = ['#C44E52' if v == 'B0_current_baseline' else '#4C72B0' for v in names]

fig, ax = plt.subplots(figsize=(12, 5.5), dpi=140)
bars = ax.bar(range(len(names)), tps, color=colors, alpha=0.9, width=0.65)
for b, t, v in zip(bars, tps, names):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
            f'{t:.0f}', ha='center', va='bottom', fontsize=8.5, rotation=0)
base = tps[0] if names[0].startswith('B0') else None
if base:
    ax.axhline(base, color='#C44E52', ls='--', lw=1, alpha=0.6)
    ax.text(len(names) - 0.4, base + 1, f'baseline {base:.0f} tok/s',
            ha='right', va='bottom', fontsize=9, color='#C44E52')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('tokens/s')
ax.set_title('MiniGPT 50M — training throughput by benchmark variant (消融)')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, bbox_inches='tight')
print('saved:', OUT)
for v, t in rows:
    print(f'{v:24s} {t:8.0f} tok/s')
