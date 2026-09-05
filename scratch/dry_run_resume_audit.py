# -*- coding: utf-8 -*-
"""Epoch2 resume 正确性 dry-run（CPU，不加载真实 checkpoint、不改生产代码）。

复现 train.py 的 resume 机制并实测：
  A. epoch 语义：checkpoint 存 epoch=0 → resume --epochs 2 是否重复 epoch 0
  B. scheduler：last_epoch 重锚 + total_steps 重铺 → resume 首步 LR 突跳量
  C. AMP warning 根因：optimizer 未 step 而 scheduler 先 step → 复现警告；
     验证"仅 optimizer 真更新才 scheduler.step"逻辑下警告消失
  D. tokens_seen：partial batch 按满批计数 vs x.numel() 计数
"""
import math
import warnings

import torch

# ---------------- 与 train.py 相同的超参 ----------------
LR = 8e-4
WARMUP = 1000
MIN_RATIO = 0.0625
BATCH = 64
BLOCK = 512
STEPS_EPOCH1 = 30463          # epoch 1（batch 64）
N_SAMPLES = 1_949_625         # PackedDataset 样本数（floor(998208111/512)）
N_TRAIN_TOKENS = 998_208_111  # token 缓存总长

def lr_lambda_factory(total_steps):
    def f(step):
        if step < WARMUP:
            return (step + 1) / WARMUP
        t = (step - WARMUP) / max(1, total_steps - WARMUP)
        t = min(t, 1.0)
        return MIN_RATIO + 0.5 * (1.0 - MIN_RATIO) * (1.0 + math.cos(math.pi * t))
    return f

print('=' * 70)
print('A. EPOCH 语义：resume --epochs 2 是否重复 epoch 0')
print('=' * 70)
# train.py: save_checkpoint(full) 存 'epoch': epoch（循环变量）；epoch0 全程 epoch=0
# resume: start_epoch = ck['epoch']; for epoch in range(start_epoch, args.epochs)
ck_epoch = 0                    # checkpoint_latest.pt['epoch']（epoch 0 结束时保存）
start_epoch = ck_epoch
run_epochs = [e for e in range(start_epoch, 2)]
print(f'checkpoint 存 epoch={ck_epoch} → start_epoch={start_epoch}')
print(f'for epoch in range({start_epoch}, 2) → 将运行: {run_epochs}')
print('>>> 结论: epoch 0 会完整重复一遍（' + ('BUG 确认' if run_epochs == [0, 1] else 'OK') + '）')
print('    正确语义应存 next_epoch=1，range(1,2) 只跑 epoch 1\n')

print('=' * 70)
print('B. SCHEDULER：resume 后 LR 突跳实测')
print('=' * 70)
# ---- 真实流程：epoch1 完整跑 30463 次 scheduler.step ----
opt1 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(4))], lr=LR)
sch1 = torch.optim.lr_scheduler.LambdaLR(opt1, lr_lambda_factory(STEPS_EPOCH1))
for _ in range(STEPS_EPOCH1):
    sch1.step()
lr_end = opt1.param_groups[0]['lr']
print(f'epoch1 完整跑完 {STEPS_EPOCH1} 步 → optimizer lr = {lr_end:.3e}'
      f'  （scheduler.last_epoch = {sch1.last_epoch}；日志实测 5.0e-05）')

# ---- resume：epochs=2 → 新 LambdaLR 重锚 last_epoch = global_step0-1 ----
TOTAL2 = STEPS_EPOCH1 * 2
opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(4))], lr=LR)
sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda_factory(TOTAL2))
sch2.last_epoch = STEPS_EPOCH1 - 1    # train.py: scheduler.last_epoch = global_step0 - 1
lr_before_any_step = opt2.param_groups[0]['lr']   # 尚未 step，仍是构造值 8e-4
sch2.step()                            # resume 后第一个 scheduler.step()
lr_resume_step1 = opt2.param_groups[0]['lr']
print(f'新 total_steps = {TOTAL2}（cosine 按新总步数重铺）')
print(f'resume 后第一个 scheduler.step() → last_epoch={sch2.last_epoch} → '
      f'optimizer lr = {lr_resume_step1:.3e}')
print(f'>>> LR 突跳: {lr_end:.2e} → {lr_resume_step1:.2e}'
      f'  = ×{lr_resume_step1 / lr_end:.1f} 倍（隐式跳变，违反"禁止"要求）\n')

print('=' * 70)
print('C. AMP warning 根因复现')
print('=' * 70)
opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(4))], lr=1e-3)
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

# 场景 1：optimizer 从未 step（= AMP scaler 检测 overflow 跳过 optimizer.step），
#         但 scheduler 仍 step → PyTorch 首步自检触发 warning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    sch.step()
warns = [str(x.message) for x in w]
print(f'场景1 (scheduler.step 先于任何 optimizer.step): {len(warns)} 条警告')
for x in warns[:1]:
    print('  ', x[:90], '...')
print('  >>> 与训练日志警告同源 → 根因 = scaler 首步跳过 optimizer.step，scheduler 仍步进')

# 场景 2：修复逻辑——只有 optimizer 确实更新才 scheduler.step
# torch 2.11 AdamW 无 _step_count 属性；用 "optimizer state 里是否有 'step' 条目" 判断
# （AdamW 首次真正 step 才会为每个参数建 state['step']；scaler 跳过则无）
def has_stepped(optimizer):
    return any('step' in s for s in optimizer.state.values())

opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(4))], lr=1e-3)
sch2b = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 1.0)
_p = opt2.param_groups[0]['params'][0]

def fake_scaler_step(optimizer, skip):
    """模拟 GradScaler.step：skip=True 时（found_inf）不调 optimizer.step。"""
    _p.grad = torch.ones_like(_p)   # 每次造梯度，让 step 真实发生
    if not skip:
        optimizer.step()
    return has_stepped(optimizer)

for skip in (True, False):
    updated = fake_scaler_step(opt2, skip)
    if updated:
        sch2b.step()
    print(f'  模拟 scaler skip={skip}: optimizer 更新={updated}, '
          f'scheduler 步进={"是" if updated else "否（跳过）"} → 计数严格一致')

with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter('always')
    # 完整一轮：先 opt 后 sch（正常）
    opt3 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(4))], lr=1e-3)
    sch3 = torch.optim.lr_scheduler.LambdaLR(opt3, lambda s: 1.0)
    opt3.step()
    sch3.step()
print(f'场景2 (仅 optimizer 更新后 scheduler.step): {len(w2)} 条警告 → warning 消除\n')

print('=' * 70)
print('D. tokens_seen 精确计数')
print('=' * 70)
full_batches = N_SAMPLES // BATCH          # 30462
last_batch = N_SAMPLES - full_batches * BATCH   # 57
print(f'样本 {N_SAMPLES:,} / batch {BATCH} → {full_batches} 满批 + 尾批 {last_batch} 样本')
old_way = (full_batches + 1) * BATCH * BLOCK
new_way = full_batches * BATCH * BLOCK + last_batch * BLOCK
true_tokens = N_SAMPLES * BLOCK
print(f'旧算法 (步数×batch×block): {old_way:,}  (多算 {old_way - true_tokens:,})')
print(f'新算法 (Σ x.numel()):      {new_way:,}  (= 真实 {true_tokens:,})')
print(f'>>> 差 {old_way - new_way:,} = 3584，与用户观察一致; 修复: tokens_seen += x.numel()')
print(f'    终值相对缓存长度 {N_TRAIN_TOKENS:,}: 弃尾 {N_TRAIN_TOKENS - true_tokens} token\n')

print('=' * 70)
print('Epoch2 LR continuation 方案数值预估')
print('=' * 70)
f_new = lr_lambda_factory(TOTAL2)
print(f'现状（机械重铺）: resume 首步 {lr_resume_step1:.2e}，之后按 60926 步新 cosine '
      f'衰减到 {LR * MIN_RATIO:.2e}（中段即回 ~{LR * f_new(30000):.2e}）')
# 方案 B-1：epoch2 恒 min_lr（无跳变）
print(f'方案1（恒温续训）: epoch2 全程 lr = {LR * MIN_RATIO:.2e}（零跳变，最稳）')
# 方案 B-2：小峰值 warm restart：500 步 warmup 到 lr_peak2 再余弦回 min
for peak2_ratio in (0.05, 0.10, 0.25):
    peak2 = LR * peak2_ratio
    print(f'方案2（小重启 peak={LR*peak2_ratio:.1e}）: {LR*MIN_RATIO:.1e} → {peak2:.1e} '
          f'(warmup 500 步, ×{peak2/(LR*MIN_RATIO):.0f} 温和升) → 余弦回 {LR*MIN_RATIO:.1e}')
