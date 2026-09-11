# -*- coding: utf-8 -*-
"""验证 train.py 新增 3 种 LR schedule 的曲线形状（纯 lambda 逻辑复刻 + 冒烟）。

不实际训练：直接复刻 lr_lambda 的数学，画出 5 种曲线确认形态，
再跑一个 6M 极小 CPU 冒烟确认代码路径可执行。
"""
import math

total_steps = 5000
warmup = 250          # 5%
min_ratio = 0.0625    # v3 实际用的底部比例

def _cos(frac, mr=min_ratio):
    return mr + 0.5 * (1.0 - mr) * (1.0 + math.cos(math.pi * frac))

def lr_lambda(step, scheme):
    if step < warmup:
        return (step + 1) / warmup
    if scheme == 'const':
        return min_ratio
    t = (step - warmup) / max(1, total_steps - warmup)
    t = min(t, 1.0)
    if scheme == 'cosine':
        return _cos(t)
    if scheme == 'wsd':
        if t < 0.7:
            return 1.0
        return _cos((t - 0.7) / 0.3)
    if scheme == 'hold-decay':
        if t < 0.55:
            return 1.0
        return _cos((t - 0.55) / 0.45)
    if scheme == 'const-tail':
        if t < 0.5:
            return _cos(t / 0.5)
        return min_ratio
    raise ValueError(scheme)

schemes = ['cosine', 'wsd', 'hold-decay', 'const-tail']
steps = [0, 100, 250, 500, 1000, 2000, 3000, 3750, 4000, 4500, 4999]
print('scheme       | ' + ' | '.join(f'{s:>5}' for s in ['0','100','250','500','1k','2k','3k','3750','4k','4500','4999']))
for sc in schemes:
    vals = [lr_lambda(s, sc) for s in steps]
    print(f'{sc:12s} | ' + ' | '.join(f'{v:5.3f}' for v in vals))

# 形态断言
def check(cond, msg):
    print(('✅' if cond else '❌'), msg)

w_hold = [lr_lambda(s, 'wsd') for s in (300, 1000, 3000)]
check(w_hold[0] == 1.0 and w_hold[1] == 1.0 and w_hold[2] == 1.0, 'WSD: stable 段恒 peak(1.0) 直到 70%')
wsd_tail = [lr_lambda(s, 'wsd') for s in (4000, 4999)]
check(wsd_tail[0] < 1.0 and wsd_tail[1] <= wsd_tail[0] and wsd_tail[1] > min_ratio - 1e-6, 'WSD: 尾部衰减且不破 min')
hd = [lr_lambda(s, 'hold-decay') for s in (300, 2500, 4000, 4999)]
check(hd[0] == 1.0 and hd[1] == 1.0 and hd[2] < hd[1] and hd[3] <= hd[2], 'hold-decay: hold 到 ~55% 剩余步后衰减')
ct = [lr_lambda(s, 'const-tail') for s in (1000, 2500, 4000, 4999)]
check(ct[1] < ct[0] and abs(ct[2] - min_ratio) < 1e-9 and abs(ct[3] - min_ratio) < 1e-9, 'const-tail: 前50%衰减后恒 min')
cos = [lr_lambda(s, 'cosine') for s in (300, 2500, 4999)]
check(cos[0] > cos[1] > cos[2] and cos[2] >= min_ratio - 1e-9, 'cosine: 单调衰减到 min')
print('\n曲线形状验证完成')
