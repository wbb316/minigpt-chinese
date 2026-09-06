# EXPERIMENT_LOG — 训练实验日志

> 进入多实验阶段，每次训练完成后**在末尾追加一行**，保留可横向比对的历史轨迹。
> 规则：
> - commit 版本用训练时代码的 git 短哈希：`git rev-parse --short HEAD`
> - val loss 单位 **nats**，**词表不同不可横向比**（随机基线 ≈ ln(vocab_size)，ln(6144)≈8.72）
> - 过拟合判断看 `gap = val - train_eval`（eval 模式无 dropout），gap 明显为正且持续增大才说明过拟合

## 字段说明

| 字段 | 说明 | 获取方式 |
|---|---|---|
| 日期 | 训练完成/记录日期 YYYY-MM-DD | 训练结束时间 |
| commit 版本 | 训练所用代码的 git 短哈希 | `git rev-parse --short HEAD` |
| 模型配置 | n_layer / n_embd / n_head / block_size / vocab_size / tie | train 命令参数 |
| 数据版本 | 语料版本及清洗脚本（v0 / v1 / all / webnovel_v2…） | 数据文件名 |
| token 数量 | 训练集 token 总数 | 首次编码日志 / tokens_*.npy 缓存长度 |
| 训练时间 | 起止时间或总时长 | 训练起止 |
| best ckpt val loss | checkpoint_best.pt 对应的 val loss | 训练结束打印 |
| 备注 | 亮点 / 踩坑 / 与上次对比 | — |

## 历史记录（自 README 与结果目录回填，日期/commit 为近似）

| 日期 | commit 版本 | 模型配置 | 数据版本 | token 数量 | 训练时间 | best ckpt val loss | 备注 |
|---|---|---|---|---|---|---|---|
| ≈2026-08-23 | ≈85460d5 | 7L/8H/256d/bs128/vocab3256 | 百合 35 本 (2900 万字) | 待补 | 待补 | 3.79 | 百合基线 7.2M；按文件 90/10 划分修复后 4.5→3.79 |
| ≈2026-09-03 | ≈7a4bf61 | 6L/8H/256d/bs256/vocab6144/tie | 轻小说 v0 | 待补 | 待补 | 4.188 | LN 修复版 6.3M；标准 LayerNorm |
| ≈2026-09-04 | ≈20393f0 | 10L/8H/384d/bs256/vocab6144/tie | 轻小说 v0+v1 (all) | 4.16 亿 | 待补 | 3.636 | 20M 最终版；train_eval 3.616 / gap 0.020，未过拟合，可继续堆数据 |
| 2026-09-05 | 614d325 | 10L/8H/512d/bs512/vocab6144/tie | webnovel_v2 (shard0, 12.4亿字) | 9.98 亿 | 20:48→22:17 (1.5h) | 3.708 | v2 首跑 35M/998M；train_eval 3.679 / gap 0.029 未过拟合；vocab 同 6144 但 **tokenizer/语料不同，与 20M 3.636 不可直接比**；踩坑：`_split_points` O(n²) 卡死数小时→numpy 二分修复；batch 128 OOM→64（总 token 不变，步数 15232→30463） |
| 2026-09-05 | 614d325 | 10L/8H/512d/**实际 batch64**/vocab6144/tie | webnovel_v2 (shard0) | 9.98 亿 (tokens_seen 998,211,584) | 1:28:49 | 3.7076 | **v2_35M_ctx512_1B_E1 · Epoch 1 complete / Epoch 2 pending**；best step 30463 / gap 0.0289；7 次验证全刷 best；scheduler warning 待修（§详见下方 E1 阶段记录） |
| 2026-09-06 | 16f0c1a | 10L/8H/512d/bs512/vocab6144/tie (batch64) | webnovel_v2 shard0（同一份第二遍） | 累计 19.96 亿 (tokens_seen 1,996,059,136) | 12:18→13:49 (~1:30) | **3.6372** | **v2_35M_ctx512_1B_E2 · Epoch 2 complete**；const LR 5e-5 恒温续训 + resume 修复（§RESUME_AUDIT）；best step 60915 / gap 0.031；AMP skipped 11 步；与 20M 3.636 语料/tokenizer 不同不可直接比（数值几乎持平）（§详见下方 E2 阶段记录） |

### v2_35M_ctx512_1B_E1 — Epoch 1 阶段记录（2026-09-05）

- **Experiment ID**：`v2_35M_ctx512_1B_E1`
- **status**：`Epoch 1 complete / continuation pending`（Epoch 2 待训练诊断后决定；NOT FINAL MODEL REPORT）
- **date**：2026-09-05 ｜ **commit**：614d325
- **实际模型配置**：10L/8H/512d/bs512/vocab6144/tie/dropout0.1；parameters = 34,676,736
- **实际训练配置**：**batch=64**（首次配置 batch128，随后实际成功运行配置调整为 batch64；tokens/step=32,768；total_steps=30,463；epochs=1）
- **train tokens**：998,208,111 ｜ **tokens_seen（末步）**：998,211,584
- **best val**：3.7076 @ step 30463（全部 7 次验证均为 best，no_improve 恒 0）
- **gap**：val − train_eval = 0.0289（无过拟合信号）
- **训练时间**：1:28:49（约 5.72 step/s，≈187k tokens/s）
- **备注**：后半段改善放缓（-0.5829→-0.0290，按每 1,000 步标准化后 0.1166→0.0058）与 cosine LR 衰减至 5e-5 重合，不能单独归因容量耗尽；**scheduler warning 已记录为 Pending Fix，未破坏本轮结果**；源码顺序为 scaler.step→scaler.update→scheduler.step，根因待确认（优先检查 AMP GradScaler 初始 overflow 跳过 optimizer update 而 scheduler 仍 step）；Epoch 2 前通过 skipped-step 检测与 dry-run 确认，并检查 resume 后 LR 锚定。

### v2_35M_ctx512_1B_E2 — Epoch 2 阶段记录（2026-09-06）

- **Experiment ID**：`v2_35M_ctx512_1B_E2`
- **status**：`Epoch 2 complete`（v2 两轮合计 2B token 训练收官；NOT 与 20M 横向对比）
- **date**：2026-09-06 ｜ **commit**：16f0c1a（resume 修复：next_epoch / const LR / AMP 门控 / 精确 tokens_seen，见 `docs/RESUME_AUDIT.md`）
- **模型配置**：同 E1（10L/8H/512d/bs512/vocab6144/tie/dropout0.1；34,676,736 params）
- **训练配置**：batch=64；tokens/step=32,768；`--lr-scheme const`（**恒 5e-5**，不重铺 cosine）；epochs=2 + resume（checkpoint `next_epoch=1` → 只跑第二轮）；total_steps=60,926（实际到 60,915 自然结束）
- **train tokens**：E1+E2 累计 1,996,059,136（≈2B）｜ tokens_seen 精确累计（`x.numel()`，修复 D）
- **best val**：**3.6372** @ step 60,915（epoch 1 结束）——E1 3.7076 → E2 3.6372，**恒温二遍继续显著降 loss**
- **train_eval / gap**：3.606 / +0.031（gap 较 E1 0.029 仅微增，无过拟合信号）
- **训练时间**：12:18→13:49（~1:30，5.7 step/s）｜ **AMP skipped**：11 步（0.04%，门控正确不推进进度）
- **resume 修复验证**：启动日志 "Epoch 1"（未重跑 epoch 0）；lr 全程 5.00e-5（无 4.35e-4 突跳）；checkpoint `next_epoch=2` / `tokens_seen=1,996,059,136` 自动记录
- **备注**：同数据第二遍在恒温 5e-5 下仍显著改善（3.708→3.637），说明 E1 尾段未收敛完全（容量/步数尚有余量）；gap 微增提示再堆同数据收益递减——**下一步优先新数据（shard1/2）或更长训练**；与 20M 3.636 语料/tokenizer 不同**不可直接比**，数值几乎持平仅巧合。

### v3_50M_ctx512_1B — 50M 参数 scaling（2026-09-06）

- **Experiment ID**：`v3_50M_ctx512_1B`
- **status**：`complete`（★ 本项目首个严格可比参数对照：同 tokenizer/语料/val 集/batch/LR 调度，仅参数量不同）
- **date**：2026-09-06 ｜ **commit**：fffbffe（配置见 `docs/experiment_config_50M.yaml`）
- **模型配置**：**12L/9H/576d**/bs512/vocab6144/tie/dropout0.1；parameters = **51,411,840**（用户选定方案 B；head_dim=64 惯例保持）
- **训练配置**：batch=64；cosine（warmup 1000 → min 5e-5）；epochs=1；total_steps=30,454（自然结束）；**与 35M E1 逐项相同**
- **train tokens**：997,913,088（≈1B，shard0 单轮；严格 1:20 口径）
- **best val**：**3.6154** @ step 30,454（epoch 末；7 次验证全 best，无早停）
- **train_eval / gap**：3.578 / +0.038（无过拟合）
- **训练时间**：20:18→22:23（~2:05，4.15 step/s）｜ **AMP skipped**：9 步
- **核心结论（同一评估空间，直接可比）**：
  - 50M@1B **3.6154** < 35M@1B **3.7076**（同数据同配置，仅 +48% 参数 → 好 0.092）
  - 50M@1B **3.6154** < 35M@2B **3.6372**（单轮 1B 打赢 35M 两轮 2B）→ **参数 scaling 收益 > 重复数据收益**
  - 各同点对比（step 5k/10k/15k/20k/25k）50M 全面领先且差距扩大（0.02→0.09）
- **备注**：验证了「加深」（12L，前两代 10L 未变）方向有效；单轮 1B 未饱和迹象（末段仍在降），**下一步 = 50M + shard1/2 新数据**（参数与数据双升的真 scaling 主线）；与旧评估空间 20M 3.636 仍**不可比**（tokenizer/语料不同）。
