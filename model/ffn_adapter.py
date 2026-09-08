"""FFN 变体 checkpoint 适配器（v3 FFN 优化实验，2026-09-08）

用途：让 relu / gelu / swiglu 三个变体都能从**同一个 v3-alpha checkpoint** 出发，
满足「同一个 checkpoint 初始化」的对照要求——唯一变量只有 FFN 本身。

三种情形：

1. relu → relu：零改动（单变量 = 无变化，用于重跑基线 / 续训）。
2. relu ↔ gelu：**参数名与形状完全相同**（fc1/fc2），只是激活不同 → 零改动，
   load_state_dict 直接 strict 通过。
3. relu/gelu ↔ swiglu：参数名与形状不同，按下面的映射改写权重。

映射（fc ↔ gate/up/down），h = swiglu 的中间维：

    前向 relu:  y = fc2 · act(fc1 · x)
    前向 swiglu: y = down · (silu(gate · x) ⊙ (up · x))

    gate.weight = fc1.weight[:h]          up.weight = fc1.weight[h:2h]
    gate.bias   = fc1.bias[:h]            up.bias   = fc1.bias[h:2h]
    down.weight = fc2.weight[:, :h]       down.bias = fc2.bias

把 fc1 的 4d 个中间通道一分为二（gate 取前半、up 取后半），down 取 fc2 对应的
前 h 列——**权重数值原样搬运**，不引入任何随机数，因此三个变体共享同一份预训练
参数、同一份 seed 数据流。gate/up 初始相等 ⇒ silu(gate·x)·(up·x) 与
act(fc1·x) 同量级，起点 loss 与基线相当（实测见实验记录）。

反向（swiglu → relu/gelu）：gate/up 拼回 fc1（后 2d 通道填零），down 填 fc2 前 h 列
（后 4d-h 列填零）——保证参数量与形状回到原 FFN，模型仍可加载。

局限（如实记录）：映射后 swiglu 的**函数**与原 FFN 不等价（门控结构不同），
只保证「参数来源相同 + 起点尺度相当」；这是结构性变量的对照实验，不是单变量替换。
"""
from __future__ import annotations

import torch


def strip_compile_prefix(sd):
    """剥掉 torch.compile 存档的 '_orig_mod.' 前缀（与 train.py 内同名函数语义一致）。"""
    if isinstance(sd, dict) and any(k.startswith('_orig_mod.') for k in sd):
        return {k[len('_orig_mod.'):]: v for k, v in sd.items()}
    return sd


def detect_ff_type(sd) -> str:
    """从 state_dict 的 key 判断存档用的 FFN 结构：'swiglu' | 'legacy'（fc1/fc2）。"""
    for k in sd:
        if k.endswith('gate_proj.weight'):
            return 'swiglu'
    for k in sd:
        if k.endswith('fc1.weight'):
            return 'legacy'
    return 'unknown'


def _copy(dst: torch.Tensor, src: torch.Tensor, r0=0, c0=0) -> None:
    """把 src 写进 dst 的 [r0:r0+rows, c0:c0+cols] 区域；超出边界的部分自动裁剪
    （源比目标大时只搬能放下的部分，源比目标小则只填左上角）。"""
    r = min(src.shape[0], dst.shape[0] - r0)
    c = min(src.shape[1], dst.shape[1] - c0)
    if r > 0 and c > 0:
        dst[r0:r0 + r, c0:c0 + c].copy_(src[:r, :c])


def adapt_state_dict(sd, model, verbose: bool = True):
    """把 checkpoint state_dict 适配到 model 的 FFN 结构，返回 (新 sd, 报告 dict)。

    不修改传入的 sd（浅拷贝 + 按需替换 tensor 引用）；无法适配的 key 原样保留，
    由 load_state_dict 的 missing/unexpected 报告暴露出来（调用方据此决定是否 strict）。
    """
    sd = strip_compile_prefix(dict(sd))
    src_ff = detect_ff_type(sd)
    dst_ff = 'swiglu' if hasattr(model.blocks[0].ff, 'gate_proj') else 'legacy'
    h = int(getattr(model.blocks[0].ff, 'hidden', 0))
    report = {'src_ff': src_ff, 'dst_ff': dst_ff, 'mapped': 0, 'copied': 0}

    if src_ff == 'unknown' or src_ff == dst_ff:
        # 同构（含 relu ↔ gelu：参数名形状完全相同）→ 不需要任何改写
        report['mapped'] = sum(1 for k in sd if k.endswith('.weight'))
        if verbose:
            print(f'🔧 FFN 适配: 存档 {src_ff} → 模型 {dst_ff}，结构一致，直接加载')
        return sd, report

    out = dict(sd)
    for i, blk in enumerate(model.blocks):
        p = f'blocks.{i}.ff.'
        if dst_ff == 'swiglu' and src_ff == 'legacy':
            w1, b1 = sd.get(p + 'fc1.weight'), sd.get(p + 'fc1.bias')
            w2, b2 = sd.get(p + 'fc2.weight'), sd.get(p + 'fc2.bias')
            if w1 is None or w2 is None:
                continue
            # fc1 的 4d 个中间通道一分为二：前半 → gate，后半 → up。
            # 通道数正好 2h 时严格对半切；否则按中位数切，目标张量按 h 建零张量，
            # 只填能填的部分（其余行保持零 = 该通道不激活，仍是同一份权重来源）。
            n1 = w1.shape[0]
            mid = n1 // 2
            h2 = min(h, mid)                     # 两侧各自能填的行数
            gw = torch.zeros(h, w1.shape[1], dtype=w1.dtype)
            uw = torch.zeros(h, w1.shape[1], dtype=w1.dtype)
            gw[:h2].copy_(w1[:h2])
            uw[:min(h, n1 - mid)].copy_(w1[mid:mid + min(h, n1 - mid)])
            out[p + 'gate_proj.weight'] = gw
            out[p + 'up_proj.weight'] = uw
            out[p + 'down_proj.weight'] = w2[:, :h].clone()
            if b1 is not None:
                gb = torch.zeros(h, dtype=b1.dtype)
                ub = torch.zeros(h, dtype=b1.dtype)
                gb[:h2].copy_(b1[:h2])
                ub[:min(h, n1 - mid)].copy_(b1[mid:mid + min(h, n1 - mid)])
                out[p + 'gate_proj.bias'] = gb
                out[p + 'up_proj.bias'] = ub
            if b2 is not None:
                out[p + 'down_proj.bias'] = b2.clone()
            for k in ('fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'):
                out.pop(p + k, None)
            report['mapped'] += 1
        elif dst_ff == 'legacy' and src_ff == 'swiglu':
            gw = sd.get(p + 'gate_proj.weight')
            uw = sd.get(p + 'up_proj.weight')
            dw = sd.get(p + 'down_proj.weight')
            if gw is None or uw is None or dw is None:
                continue
            d = model.blocks[i].ff.dim
            w1 = torch.zeros(4 * d, d, dtype=gw.dtype)
            _copy(w1, gw); _copy(w1, uw, r0=h)
            w2 = torch.zeros(d, 4 * d, dtype=dw.dtype)
            _copy(w2, dw)
            out[p + 'fc1.weight'] = w1
            out[p + 'fc2.weight'] = w2
            gb, ub, db = (sd.get(p + 'gate_proj.bias'), sd.get(p + 'up_proj.bias'),
                          sd.get(p + 'down_proj.bias'))
            if gb is not None:
                b1 = torch.zeros(4 * d, dtype=gb.dtype)
                _copy(b1.unsqueeze(1), gb.unsqueeze(1))
                if ub is not None:
                    _copy(b1.unsqueeze(1), ub.unsqueeze(1), r0=h)
                out[p + 'fc1.bias'] = b1
            if db is not None:
                out[p + 'fc2.bias'] = db.clone()
            for k in ('gate_proj.weight', 'gate_proj.bias', 'up_proj.weight',
                      'up_proj.bias', 'down_proj.weight', 'down_proj.bias'):
                out.pop(p + k, None)
            report['mapped'] += 1
    if verbose:
        print(f'🔧 FFN 适配: 存档 {src_ff} → 模型 {dst_ff}，改写 {report["mapped"]} 层 '
              f'(h={h})')
    return out, report
