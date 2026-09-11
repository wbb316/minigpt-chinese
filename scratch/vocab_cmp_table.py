# -*- coding: utf-8 -*-
"""vacab 对照：逐验证点换算 bits/char 并列出差值。

只读 val_history CSV（+ 全量 val 上的 chars/token 做换算）。
"""
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAND = os.path.join(ROOT, 'log', 'vocab对照6144vs8192')
LN2 = math.log(2.0)
# 全量 val（143,893,657 字符）上实测的 chars/token —— 换算分母必须用它
CPT = {'v6144': 1.2429, 'v8192': 1.2861}

curves = {}
for tag in ('v6144', 'v8192'):
    rows = {}
    with open(os.path.join(LAND, f'val_history_{tag}.csv'), encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                rows[int(r['step'])] = float(r['val'])
            except (KeyError, ValueError):
                continue
    curves[tag] = rows

hdr = (f'{"step":>6} {"6144 val":>9} {"6144 b/c":>9} '
       f'{"8192 val":>9} {"8192 b/c":>9} {"delta b/c":>10} {"rel%":>7}')
print(hdr)
print('-' * len(hdr))
common = sorted(set(curves['v6144']) & set(curves['v8192']))
for s in common:
    a = curves['v6144'][s]
    b = curves['v8192'][s]
    ac = a / LN2 / CPT['v6144']
    bc = b / LN2 / CPT['v8192']
    print(f'{s:>6} {a:>9.4f} {ac:>9.4f} {b:>9.4f} {bc:>9.4f} '
          f'{bc-ac:>+10.4f} {(bc-ac)/ac*100:>+6.2f}%')
print('-' * len(hdr))
print('b/c = bits per character（跨词表可比，越小越好）')
print('delta 为负 = 8192 更好')
print()
print('※ 注意：per-token val（val 列）跨词表**不可比**，8192 那列天然更大，')
print('  这正是必须换算到 bits/char 的原因。')
