# RESUME_AUDIT — 35M Epoch 2 resume 正确性审查报告

- 日期：2026-09-06
- 范围：**只审查，未修改任何代码、未启动训练**
- 对象：v2 35M（10L/512d/bs512/vocab6144/tie），Epoch 1 已完成（30463 步 / 998M token / best val 3.7076 / final lr 5e-5）
- 检查项：A epoch resume 语义 · B scheduler resume · C AMP warning · D tokens_seen 计数
- 材料：`docs/MiniGPT_Project_Status.md`、`docs/EXPERIMENT_LOG.md`、`train/train.py`（547 行全文）、`docs/experiment_config_v2.yaml`、本地 `log/35M参数+700Mtokens/`（run_config/val_history/step_history）、dry-run 实测
- checkpoint metadata：**云端实例已关机（22:17 训练结束后），未能直接读取**；字段值由代码路径高置信推断（见 §0）

## §0 checkpoint_latest.pt 字段推断（待开机验证）

`save_checkpoint(full=True)`（train.py L369-376）保存：`model / optimizer / scaler / step / epoch / best_val / no_improve`。
本 run 无 resume、无早停（no_improve 全程 0），最后保存点为 **epoch 0 结束**（step 30463，22:17:26）→ 推断：

| 字段 | 推断值 | 依据 |
|---|---|---|
| `epoch` | **0** | 循环变量 `for epoch in range(0, 1)` 全程为 0（L489） |
| `step` | 30463 | val_history 末行 step 30463 |
| `best_val` | 3.7076 | val_history is_best=1 行 |
| `no_improve` | 0 | 全程无未改善验证 |
| optimizer param_groups lr | 5e-5 | scheduler 已把 lr 写入 param_groups（dry-run 实测一致） |

> ⚠️ 开机后建议用 `scratch/cloud_read_ckpt_meta.py` 复核（已备好，只读不加载权重）。

---

## §1 A — epoch resume 语义：**存在重复 epoch 0 的 BUG（确认）**

### 现状代码

```python
# 保存（L370-376）——存的是"循环变量 epoch"
torch.save({'model': …, 'step': global_step, 'epoch': epoch, …}, ckpt_path)

# resume（L290-291）
start_epoch = int(ck.get('epoch', 0))
# 主循环（L489）
for epoch in range(start_epoch, args.epochs):
```

### 问题

epoch 0 结束时 checkpoint 存 `epoch=0` → `resume --epochs 2` 时 `start_epoch=0` →
`range(0, 2)` 会**完整重跑 epoch 0，再跑 epoch 1** → 同一 1B token 被训两遍（合计 3 遍）。

dry-run 实测：`range(0,2) → [0, 1]`，epoch 0 重复。

### 修复设计（未实施，待确认）

核心：checkpoint 的 epoch 字段必须表达**"下一个要开始的 epoch"**（next-epoch 语义），
并区分 **epoch 末保存**（该 epoch 已完成 → next = epoch+1）与 **mid-epoch 保存**（该 epoch 未完成 → next = epoch，重跑）。

1. `save_checkpoint(full=True)` 增加保存 `'next_epoch'`；由调用点语义决定取值：
   - 步级验证保存（`run_validation('step N')`）：`next_epoch = epoch`（当前 epoch 未完成）
   - epoch 末验证保存（`run_validation('epoch X 结束')`）：`next_epoch = epoch + 1`
2. resume：`start_epoch = int(ck.get('next_epoch', ck.get('epoch', 0)))`
   （向后兼容旧 checkpoint：旧 `epoch` 字段是循环变量，epoch 末存 0 → 语义退化与现状相同，属已知风险，仅兼容旧包用）
3. 主循环不变：`for epoch in range(start_epoch, args.epochs)`

验证矩阵（实施后 dry-run）：

| 场景 | checkpoint next_epoch | resume epochs | 应运行 | 现状行为 |
|---|---|---|---|---|
| epoch0 自然结束 → epoch2 | 1 | 2 | [1] | [0,1] ❌ 重复 |
| mid-epoch 中断（step 5000） | 0 | 2 | [0,1]（重跑 0） | [0,1] ✓ |
| epoch1 结束 → epoch3 | 2 | 3 | [2] | [1,2] ❌ |

> mid-epoch 语义澄清：当前代码**只在验证点/epoch 末保存 full 包**，无每步保存；
> 进程中断时损失 ≤ `val_every` 步（从上个验证点续）。"mid-epoch 保存"指 step 5000/10000… 验证点的 latest 包——它存的是**未完成 epoch 0**，next 应为 0。

---

## §2 B — scheduler resume：**cosine 被重铺，LR 隐式突跳 ×8.7（确认）**

### 现状机制

- resume 仅一行恢复（L294）：`scheduler.last_epoch = global_step0 - 1`
- 但 **LambdaLR 闭包捕获的 `total_steps` 是新的**（L236：`total_steps = steps_per_epoch * args.epochs`，epochs=2 → 60926）
- `lr_lambda(step)` 用 `(step - warmup) / (total_steps - warmup)` 重铺整段 cosine

### dry-run 实测（本地 torch 2.11 CPU，真实 LambdaLR 流程）

| 时刻 | lr |
|---|---|
| epoch1 最后一步（last_epoch=30462，total=30463） | **5.00e-05**（与日志一致） |
| resume 后第一个 `scheduler.step()`（last_epoch=30463，新 total=60926） | **4.35e-04** |

→ **5e-5 → 4.35e-4，×8.7 倍隐式跳变**。且新 cosine 中段（~step 45000）仍维持 ~1.7e-4
量级的大学习率——等于无意的 warm restart，会扰动 epoch1 尾部已精细收敛的权重（loss 预计回跳再降）。
Status 坑点 4 记录过"resume 会重锚 cosine（warm restart）"——此前是已知现象，本次按用户要求须消除。

### 为什么不能"简单交换/保留 last_epoch"解决

LambdaLR 的 lr = `initial_lr(8e-4) × lr_lambda(last_epoch)`，重铺是**结构性的**：
只要 total_steps 变大，同一步在曲线上的位置就变了。仅设 last_epoch 无法固定"从 5e-5 继续"。

### Epoch2 LR continuation 建议方案（数值见 §5）

| 方案 | 描述 | 跳变 | 适用 |
|---|---|---|---|
| **1 恒温续训（推荐）** | epoch2 全程 lr = 5e-5（epoch1 尾值），scheduler 退化为常值 | 0 | 同一数据第二遍，最稳，防止扰动已收敛权重 |
| **2 小 warm restart** | 显式：5e-5 →(warmup 500)→ peak →(cosine)→ 5e-5，peak 建议 ≤1e-4（0.125×峰值） | 温和×2（设计内） | 想给第二遍一点探索空间 |

两方案都不再使用"8e-4 起始的新 LambdaLR"。实施方式（待确认后定）：
- 方案1：resume 后不新建 cosine LambdaLR，改为显式 `optimizer.param_groups[0]['lr'] = 5e-5`
  并每步不再调 scheduler（或挂 ConstantLR）。
- 方案2：显式分段（warmup 500 + cosine 29963），实现为独立 scheduler 或在循环里手写 lr 更新。
- 关键约束：**lr 值永远显式写入并打印**，杜绝任何隐式来源。

---

## §3 C — AMP warning 根因：**scaler 跳过 optimizer.step 而 scheduler 仍步进（确认）**

### 根因链

1. 循环顺序本身正确：`scaler.step(optimizer) → scaler.update() → scheduler.step()`（L506-511）
2. 但 GradScaler 在检测到梯度 overflow（found_inf）时**跳过真正的 `optimizer.step()`**
3. scheduler 的 `step()` 自检（PyTorch 只在 `_step_count==1` 时检查一次）发现
   "optimizer 从未 step 过" → 触发该 warning
4. 启动即出现（loss≈323 的异常大首步与此吻合——首步梯度溢出是 AMP 常见瞬态）

dry-run 复现：手动"先 scheduler.step、optimizer 从未 step" → 1 条同文案警告；"仅 optimizer 更新后 scheduler.step" → 0 警告。

### 修复设计（未实施，待确认）

训练步改为"**仅当 optimizer 确实完成参数更新才推进一切计数**"：

```python
# 伪代码（实施时细化；检测用 optimizer state 是否出现 'step' 条目，
# 版本无关 —— 本地 torch 2.11 无 _step_count 属性）
had_update = _optimizer_updated(optimizer)      # scaler.step 前后比较
scaler.step(optimizer); scaler.update()
if had_update:
    scheduler.step()
    global_step += 1
    tokens_seen += x.numel()
    记 step 行 / tqdm
else:
    amp_skip_steps += 1                          # 记录：AMP skipped step 计数
```

- skipped 步：不推进 global_step / scheduler / tokens_seen / step CSV（该步不产生学习，不消耗进度）
- 需要小规模 GPU 冒烟（云端开机后跑 `--max-steps 200`）确认：warning 消失 + amp_skip_steps 打印 + loss 正常下降
- 若 skipped 计数异常高（>1%），提示 scale 初始值/首步 loss 问题

---

## §4 D — tokens_seen：**partial batch 多计 3584 token（确认）**

### 现状（L426）

```python
tok = global_step * args.batch_size * args.block_size   # 每步都按满批 64×512 计
```

### 实测

样本数 1,949,625 = 30462 满批(64) + 尾批 57 样本：
- 现状终值：30463 × 64 × 512 = **998,211,584**（多算 3,584）
- 实际：998,208,000（= 1,949,625 × 512，相对缓存 998,208,111 弃尾 111）

### 修复设计（未实施，待确认）

循环内维护累计量：`tokens_seen += x.numel()`（尾批 x.numel() = 57×512 = 29,184，天然正确）；
`append_step_row` 不再用 `global_step × batch × block` 反推。
注意 batch_64 的 epoch 进度 `ep = tok / len(train_tokens)` 也随之精确到 1.000。

---

## §5 Epoch2 预计 LR 曲线（数值）

epoch1 实际：8e-4 峰值（step 0→1000 warmup）→ cosine → step 30463 处 5.0e-5。

**现状（若直接 `--resume --epochs 2`，bug 行为）**：
```
step 30463:  5.0e-5          ← resume 前
step 30464:  4.35e-4  ← 突跳 ×8.7（新 total=60926 的 cosine 重铺）
step ~45000: ~1.7e-4         ← 新 cosine 中段仍高
step 60926:  5.0e-5
```

**方案 1（恒温，推荐）**：
```
step 30464 → 60926: 恒定 5.0e-5（平线，零跳变）
```

**方案 2（小 warm restart，peak=1e-4）**：
```
step 30464→30963: 5.0e-5 → 1.0e-4（线性 warmup 500 步，×2）
step 30964→60926: 1.0e-4 → 5.0e-5（cosine，29963 步）
```
（peak 可选 8e-5 / 1e-4 / 2e-4；>2e-4 即重回"隐性重启"风险区，不推荐）

### 建议与提示

- **同数据二遍提示**：epoch2 = 同一 shard0 第二遍（非新数据）。当前 gap 0.029 很小，
  但同分布二遍训练 gap 通常会上升、val 收益递减——若目标是真 scaling，更优路径是
  加入 shard1/2 新语料（属新实验变量，需另行确认）。
- 若选方案 1 且 val 二遍无改善（patience 2 触发早停），预计 ~step 40k 前自然停止，成本可控。

---

## §6 dry-run 测试结果（本地 scratch/dry_run_resume_audit.py）

| 项目 | 结果 |
|---|---|
| A epoch 重复 | `range(0,2)→[0,1]`，epoch 0 重复 **确认** |
| B LR 突跳 | 5.00e-5 → 4.35e-4（×8.7）**确认**（真实 LambdaLR 流程） |
| C warning 复现 | "scheduler 先于任何 optimizer.step" → 1 条同文案警告；仅更新后步进 → 0 警告 |
| C 修复逻辑 | skip=True 不步进 scheduler、skip=False 步进 → 计数严格一致 |
| D tokens_seen | 旧 998,211,584 vs 实际 998,208,000，差 3,584（=57 样本尾批按满批计）|

## §6b 修复实施 + 本地冒烟结果（2026-09-06，CPU mini 模型）

已按用户确认实施 A-D 四项（仅改 `train/train.py` 控制流，模型/权重/tokenizer/数据/超参未动），
mini 语料（152K 字符，2L/64d/bs32/batch16/vocab1024）两轮验证：

| 验证点 | 结果 |
|---|---|
| run1（epochs=1）checkpoint 字段 | `next_epoch=1`、`tokens_seen=64480`（精确，尾批 15 样本）、opt lr 5e-5 ✓ |
| run2（resume epochs=2, `--lr-scheme const`） | 日志 "Epoch 1"（**未重跑 epoch 0**）；lr 恒 **5.00e-05**（无跳变）✓ |
| step CSV 续接 | 行 127：epoch 列 1.008（tok_total 64480→64992 精确累计）、lr 5.00e-05 ✓ |
| AMP 门控（CPU） | else 分支 updated=True 正常；AMP skip 路径需云端 GPU 冒烟 |

> ⏳ 待云端开机后补：`--max-steps 200` GPU 冒烟——验证 warning 消失 + AMP skipped 计数打印。

---

## §7 待办 / 待确认清单

1. ~~云端开机后读 checkpoint metadata~~：实例关机未读；A-D 修复不依赖其字段值（代码路径已实测）
2. **Epoch2 LR 方案**：已确认 = 方案 1 恒温 5e-5（`--lr-scheme const`）✅
3. **修复范围**：A-D 四项已实施 ✅（2026-09-06，commit 见 git log）
   - [x] A：checkpoint 存 `next_epoch`（epoch 末 = epoch+1；步级 = epoch），resume `start_epoch=next_epoch`
   - [x] B：`--lr-scheme const`（恒 min_lr），弃 8e-4 重铺；cosine resume 保留但打印跳变警告
   - [x] C：仅 `_opt_step_count` 增加（optimizer 真更新）才 `scheduler.step / global_step / tok_total`；`amp_skip_steps` 计数
   - [x] D：`tok_total += x.numel()`；checkpoint 存 `tokens_seen`，resume 精确续接
4. **待云端开机**（用户明天操作时）：
   - 上传新 `train/train.py` → 云端 `/root/train/train.py`
   - 跑 `scratch/cloud_read_ckpt_meta.py` 复核 checkpoint 字段（epoch/step/lr/scaler）
   - GPU 冒烟：`--max-steps 200` 验证 AMP warning 消失 + skipped 计数打印
5. 正式 Epoch 2（resume latest + `--lr-scheme const --epochs 2`）需用户单独确认后启动

---

## 附录：GPT 权重初始化——未来改进项（不进本修复）

按用户指示记录（不属 Epoch2 resume 修复、不改权重）：模型初始化方案（`model/gpt.py` 内）
如需评审/改进，单独作为"从零训练版本"的改进项处理，与 resume 机制完全解耦。
