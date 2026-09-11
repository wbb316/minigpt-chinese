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

### v1 时代回顾：早期"过拟合"的真相与修复演进（2026-09-08 回填澄清）

> 背景：复盘 v1（旧空间：百合 / 轻小说）时期"模型老是过拟合"的记忆。按 README「关键优化点」与上方历史记录表回填，实际演进分三步，**最早的"高 val"不是过拟合，而是验证集切分泄漏**——不是靠"上大参数 + 上大数据"救回来的。

1. **7.2M 百合基线（≈2026-08-23，85460d5）：高 val 的真凶 = 评估泄漏，不是泛化问题**
   - 现象：val loss ≈ **4.5**，当时观感像"过拟合 / 泛化差"
   - 根因：验证集按 **token 顺序**切分 → val 可能包含训练**从未见过的整本书** → 评估天然偏高（数据泄漏，评估失真）
   - 修复：验证集改**按文件（按书）90/10 划分** → 4.5 → **3.79**（README 关键优化点 1，本表 85460d5 行）
   - 这一步**与参数/数据规模无关**——不是靠堆规模解决的

2. **6.3M LN 修复版（≈2026-09-03，7a4bf61）**：修的是 **LayerNorm 实现**（无偏方差 + 无 eps → 标准实现），val 4.188；属实现 bug 修复，与过拟合无关

3. **20M 最终版（≈2026-09-04，20393f0）：真正的"参数 + 数据双升" + 确立 gap 判据**
   - 规模：6.3M（6L/256d）→ **20M（10L/384d）**；语料 轻小说 v0 → **v0+v1（4.16 亿 token）**
   - 此时才引入**正确过拟合判据**：`train_eval`（eval 无 dropout）+ `gap = val − train_eval`
   - 终点：val **3.636** / train_eval 3.616 / **gap 0.020 → 判定"未过拟合，可继续堆数据"**（本表 20393f0 行；Status「已验证结论」）
   - 所以"堆参数 + 堆数据"确实发生了，但它是**继续压 loss 的手段**；**确认没过拟合靠的是 gap 判据**——堆规模本身不消除过拟合，是 4.16 亿 token 规模足够大 + 验证判据正确，才得到这个结论

4. **对 v1 记忆的澄清**：若记忆里的"过拟合"指早期 val 明显偏高（~4.5）——那是**泄漏 bug**（按文件划分修复）；若指"小模型小数据怕过拟合"——后期以 20M + 4.16 亿 token + gap 判据确认未过拟合。两者都不是"过拟合已发生、靠堆规模救回来"的故事。

5. **遗留到 v2 的连续性**：gap 判据沿用至今（35M E1 0.0289 / E2 0.031 / 50M 0.0378 均判"无过拟合"；50M gap 偏大是容量大、train_eval 压得更低的**预期伴随现象**，不是过拟合，见 `docs/report_output/v2_50M+1B/MiniGPT_v2_Scaling_Analysis.md` §4）。

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

### v2_50M_ctx512_1B — 50M 参数 scaling（2026-09-06）

- **Experiment ID**：`v2_50M_ctx512_1B`（v2 家族：与 35M 同 webnovel_v2 语料/tokenizer/评估空间）
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

### 位置编码对比短训（2026-09-06，非正式模型 run）

- **实验**：50M（12L/576d/9H）从零 + **新 GPT-2 init** + seed 42 + compile，sinusoidal vs rope 各 4000 步
- **结果**（同 step 对齐，唯一变量 = 位置编码）：

| step | sinusoidal | rope |
|---|---|---|
| 1000 | 6.443 | **4.331** |
| 2000 | 5.558 | **3.931** |
| 3000 | 4.788 | **3.780** |

- **结论**：**RoPE 胜出**（val 好 ~1.0-2.1 nats；代价速度 -6.3%）；起点 loss 均正常（新 init 消灭 300+ 爆炸，7.9 vs 基线 8.72）
- **决定**：默认 `position_encoding` 切 **rope**（GPT/train.py），sinusoidal 保留可切换（旧 checkpoint 推理自动检测）；旧 35M/50M（sinusoidal）结果不受影响
- commit：89f11a8（实现）+ e4a9a16（默认切 rope）

### v3_alpha_50M_ctx512_1B — 50M 新底层正式训练 v3_alpha（RoPE + GPT-2 init，2026-09-08）

- **Experiment ID**：`v3_alpha_50M_ctx512_1B`（**v3 = 50M 新底层系列**：GPT-2 init + RoPE；**alpha = 系列第 1 版（ReLU FFN）**；beta = 后续换 FFN 激活的变体；评估空间沿用 webnovel_v2/v6144/同 val 集 → **与 v2 系数字直接可比**）
- **status**：`complete`（★ 同评估空间**新 best：3.2097**，打破 v2 50M 3.6154）
- **date**：2026-09-08 ｜ **commit**：981d35f（rope 默认确定后；launcher `scratch/run_50m_rope_cloud.sh`）
- **模型配置**：**12L/9H/576d**/bs512/vocab6144/tie/dropout0.1/FFN=ReLU；parameters = **51,411,840**（与 v2 50M 完全同结构，纯底层替换）
- **与 v2 50M 的差异（双变量，非单变量）**：① **GPT-2 init**（N(0,0.02)+residual 缩放，commit ac7928d）② **position_encoding=rope**（89f11a8/e4a9a16）；rope-vs-sinusoidal 单变量短训已独立证明 rope 赢 ~1.0-2.1 nats（见上节）
- **训练配置**：batch=64；cosine（warmup 1000 → min 5e-5）；epochs=1；total_steps=30,455（自然结束）；**compile=True**
- **train tokens**：997,945,856（≈1B，shard0 单轮；严格 1:20 口径）
- **best val**：**3.2097** @ step 30,455（epoch 末；7 次验证全 best）
- **train_eval / gap**：3.142 / +0.068
- **训练时间**：16:54→18:08（~1:14，compile 加速后 ~7.2 step/s）｜ **AMP skipped**：8 步
- **同点对比 v3_alpha vs v2 50M**（同结构同数据同 batch/LR，底层升级，各 step val）：

| step | v2 50M (sinusoidal+旧init) | v3_alpha 50M (rope+新init) | 差 |
|---|---|---|---|
| 5000 | 4.747 | **3.632** | −1.115 |
| 10000 | 4.121 | **3.463** | −0.658 |
| 15000 | 3.871 | **3.364** | −0.507 |
| 20000 | 3.728 | **3.290** | −0.438 |
| 25000 | 3.652 | **3.236** | −0.415 |
| 30000 | 3.617 | **3.210** | −0.407 |
| 30454/455 | 3.6154 | **3.2097** | −0.406 |

- **核心结论（同一评估空间，直接可比）**：
  - v3_alpha 50M **3.2097** < v2 50M **3.6154**（底层升级 → 好 **0.406 nats**，~11% 相对；rope 短训同点差与正式 run 尾段收敛吻合）
  - 全梯度重排：**v3_alpha 50M 3.2097 < v2 50M 3.6154 < 35M E2 3.6372 < 35M E1 3.7076**
  - gap 0.068 较 v2 50M 0.038 偏大：train_eval 压得更低（3.142 vs 3.578）是大容量+新底层拟合更紧的**预期伴随现象**；val 单调下降无拐点 → **无过拟合信号**（沿用 v2 scaling 分析 §4 趋势判据）
- **备注**：compile 训练导致 checkpoint key 带 `_orig_mod.` 前缀 → 已修复（commit a31072b：A=推理工具自动剥离前缀，B=train.py 保存 raw_gpt 防复发）；产物归档 `log/50M参数_v3_alpha+998Mtokens/`（best/latest + tokenizer + run_config + step/val 历史 + 曲线图）；生成经 `python generate.py --ckpt ... --n-head 9` 验证正常；**下一步主线 = v3 系列 + shard1/2 新数据**（参数已定型，数据侧真 scaling；FFN 激活变体待 v3_beta）。

### v3 FFN 变体（relu / gelu / swiglu）— 短程对照完成（2026-09-08）

- **Experiment ID**：`v3_ffn_variants_shortrun`（配置：`docs/experiment_config_ffn.yaml`；launcher：`scratch/run_ffn_scratch_cloud.sh`）
- **status**：`complete`（★ 5000 步短程对照收官：**SwiGLU 胜出 → 已把默认 FFN 从 relu 改成 swiglu**）
- **目的**：在 v3_alpha 的底层（RoPE + GPT-2 init）上**只改 FFN**，测激活/结构对性能的影响
- **三个变体**：

| 变体 | FFN 结构 | 中间维 | FFN 层参数 | 总参数 | 初始化 |
|---|---|---|---|---|---|
| v3-relu | `fc1→ReLU→fc2` | 2304 (=4d) | 2,657,088 | 51,411,840 | 从零 + GPT-2 init（seed 42） |
| v3-gelu | `fc1→GELU→fc2` | 2304 (=4d) | 2,657,088 | 51,411,840 | 同上（参数名/形状逐位一致） |
| v3-swiglu | `gate/up→silu(gate)*up→down` | 1536 (≈8d/3) | 2,657,856 (+0.036%) | 51,421,056 (+0.018%) | 同上 seed，但张量形状不同 → 起点数值不同 |

- **对照条件（三者逐项相同，唯一变量 = `--ff-type`）**：
  - 从零随机初始化（**非**从 v3_alpha checkpoint 续训）+ 同 seed 42（数据流一致）
  - 同数据 shard0 / batch 64 / ctx 512 / 12L-576d-9H / vocab 6144 / tie / dropout 0.1
  - 同 lr **8e-4 cosine**（warmup 1000 → min 5e-5）/ 同 weight_decay 0.05 / fp16 / compile
  - 同 **5000 步** / val_every 1000 / 同 eval_batches / 同 val 集
  > **为何从零而非续训**：续训式对照（从 v3_alpha 的 relu 权重出发）里 relu 天然占优——换激活要花步数
  > 从扰动中恢复，5000 步可能只是恢复期；从零同 seed 才能公平回答「哪个激活更好」
  > （launcher `scratch/run_ffn_scratch_cloud.sh` 头部注释同此口径）。
  > 三个变体**同条件可比**（唯一变量 = ff-type）；与 v3_alpha 正式 run 的**绝对 loss 不完全可比**
  > （起点/调度细节不同），本实验只用于**三变体之间的排序**。
- **方法学口径（写进正式记录时勿省略）**：
  - **relu ↔ gelu 是严格单变量 ✅**：参数名/形状/参数量逐位一致（51,411,840 = 51,411,840），
    同 seed 初始化完全一致，只换激活函数——这组对比最干净。
  - **swiglu ↔ 前两者不是 100% 单变量 ⚠️**：① 参数量 +0.018%（h=1536 已按参数量对齐，
    可忽略但非逐位相同）；② 更重要——**结构不同 → 同 seed 下初始化数值并不一致**：
    seed 固定的是随机数流，swiglu 的 gate/up/down 与 relu 的 fc1/fc2 形状不同，
    随机流消耗方式不同 → 起点权重数值与 relu/gelu 不完全相同。
  - 准确表述：**relu↔gelu 是干净单变量；swiglu 属「FFN 结构变体」对比**（门控结构 + 参数量
    +0.018% + 同 seed 不同形），**非纯激活单变量**。
- **代码改动**（2026-09-08 对照后定版：`DEFAULT_FF_TYPE = 'swiglu'`，默认 = **swiglu**；relu/gelu 原实现保留未删，可随时回退）：
  - `model/layers.py`：新增 `DEFAULT_FF_TYPE = 'swiglu'` 与 `FF_TYPES = ('relu','gelu','swiglu')`；
    `FeedForward(dim, dropout, ff_type=DEFAULT_FF_TYPE, ff_hidden=None)`；`swiglu_hidden(d)` 解
    `3hd≈8d²`（h≈8d/3）；relu/gelu 参数名形状与 v3_alpha 完全一致
  - `model/gpt.py`：`GPT(..., ff_type=DEFAULT_FF_TYPE, ...)`；残差缩放 init 改为缩 `out_proj_weight()`
    （relu/gelu→fc2，swiglu→down_proj）
  - `model/ffn_adapter.py`（新）：relu↔swiglu checkpoint 权重映射（fc1 对半切→gate/up，
    fc2 前 h 列→down；**数值原样搬运、无随机数**），加载后无 missing/unexpected
  - `train/train.py`：`--ff-type` 默认 = swiglu；另有 `--ff-hidden/--ff-hidden-round/--init-from/--init-optimizer`；
    run config 记录 ff 信息；step CSV 新增 `tokens_per_sec` 列；结束打印吞吐 + 显存峰值
  - 推理侧（`generate.py` / `app/server.py` / `visualize_attention.py`）从 state_dict 自动检测
    ff_type（含 `gate_proj` → swiglu，否则 relu）——旧 checkpoint 不受默认值影响
  - 测试 `test/test_ffn.py`（12 项）：relu 数值回归、relu↔gelu 同名同形状、swiglu 参数对齐、
    适配器双向映射精确、三变体经适配器加载后起点 CE 量级相当、默认值 = swiglu 回归、
    默认 swiglu 端到端短训降 loss、CLI 参数可见性
- **结果（best val @5000 步，nats，webnovel_v2 评估空间）**：

| 排名 | 变体 | best val @5000 步 | 差值（相对 swiglu） |
|---|---|---|---|
| **1（胜出）** | **swiglu** | **3.655** | — |
| 2 | gelu | 3.688 | +0.033 |
| 3 | relu | 3.695 | +0.040 |

→ **SwiGLU 微弱胜出**（比 relu 好 **0.040 nats**），排名 **swiglu > gelu > relu**；
  领先从 **step 1000** 起即建立、**全程一致**；据此**已把默认 FFN 从 relu 改成 swiglu**
  （relu/gelu 保留可回退）。

- **备注**：本实验的默认值定版为 **swiglu**（`DEFAULT_FF_TYPE`），relu/gelu 保留可回退；
  `--init-from` / `--init-optimizer` / `model/ffn_adapter.py` 是本次一并实现的能力
  （让变体也能从同一 checkpoint 出发做续训式对照），**本轮 from-scratch 对照未使用它们**。
- **本地 CPU 结构性验证（`scratch/ffn_cpu_bench.py`，600 步，4L/128d/4H，小语料 ——
  **不是性能结论**，只证明代码路径/对照流程闭环）**：

| 变体 | FFN 层参数 | 总参数 | train EMA | val | train_eval | gap | init 映射 |
|---|---|---|---|---|---|---|---|
| relu（从零） | 131,712 | 1,051,344 | 6.235 | 5.504 | 5.447 | +0.058 | N/A |
| relu（同起点） | 131,712 | 1,051,344 | 4.967 | 4.873 | 4.842 | +0.031 | ok |
| gelu（同起点） | 131,712 | 1,051,344 | 5.025 | 4.906 | 4.877 | +0.029 | ok |
| swiglu（同起点） | 132,912 | 1,056,144 | 5.375 | 4.921 | 4.887 | +0.035 | ok |

  → relu↔gelu 零改写、relu→swiglu 经 adapter 映射均可正常从**同一 checkpoint** 出发
  （无 missing/unexpected），参数对齐符合设计（swiglu FFN 层 +0.9%、总参数 +0.5%，小 dim 下
  对齐粒度更粗）；600 步差异在噪声范围内——**本表仅验证代码路径（含 adapter 同起点能力），
  非性能结论**；本轮正式对照为**从零随机初始化**、未走 adapter 路径，**方向性结论见上方
  结果表**（云端 5000 步 50M run 已收官）。

- **下一步**：默认 FFN **已是 swiglu**（train `--ff-type` 默认值，无需再显式指定）；后续
  v3 正式训练（**新数据 shard1/2**）将用 swiglu 跑主线。

---

## v3 vocab 消融（tokenizer 侧 + LM 短测）— 完成（2026-09-10）

- **Experiment ID**：`v3_vocab_ablation`（tokenizer 侧）/ `v3_vocab_cmp_lm`（LM 短测）
- **status**：`complete`（★ 两阶段都收官 → **vocab 定版 8192**）
- **commit**：`a31072b`（`e22aa33` 之前；本实验未改模型代码）
- **配置**：`docs/tokenizer_sample_size_plan.md` §10（tokenizer 侧）；LM 短测见下方表

**阶段一（样本量）**：4M / 8M / 16M / 32M 字符，vocab 固定 6144。
结论：**4M→32M（8× 数据、9.4× 时间）压缩率只提升 0.623%**（chars/token 1.2429 → 1.2507）→
性价比极低，**维持 4M 训练样本**。详见 `tokenizer_sample_size_plan.md` §8。

**阶段二（vocab 大小，tokenizer 侧）**：4M 样本固定，vocab = 4096 / 6144 / 8192（12288 按指示取消）。
> 方法要点：BPE 顺序贪心合并 → 第 k 个 merge 的 id = `len(vocab)`（258 起连续分配），
> 因此**大词表截断 == 独立训练的小词表**（前缀性质）。只独立训练 8192（722.7s），
> 4096/6144 由它截取派生；**截取的 6144 与阶段一独立训练的 `tok_4M.pkl` 逐位一致**（已验证），
> 故三档唯一变量就是 vocab 大小，且排除了多次独立训练的合并路径随机差异。

| vocab | chars/token (B2) | 相对 4096 | 嵌入参数 | 词表利用率 |
|---|---|---|---|---|
| 4096 | 1.1662 | — | 2.36M | 91.4% |
| 6144 | 1.2429 | −6.17% | 3.54M | 88.6% |
| 8192 | 1.2872 | −9.40% | 4.72M | 85.0% |

边际收益（每 +2048 vocab）：4096→6144 省 21,168 token（−6.17%）；6144→8192 省 11,073（−3.44%，为第一档的 52.3%）。
**长尾分析**（40M 字符 val，外推 1B token 预算）：即使 8192，预计出现 <100 次的 token 仅 70 个、占训练槽位 ≈0.00%
→ **长尾不构成否决理由**；也**修正了阶段一的判断**（"6144 有 11.4% 冗余"是 400K 字符小样本的假象）。

**阶段三（LM 短测，唯一能定案的证据）**：intrinsic 指标无法裁定 6144 vs 8192 → 各跑 5000 步真实训练。

| 项 | 设定 |
|---|---|
| 模型 | 12L/576d/9H / swiglu / tie / rope / dropout 0.1 / block 512 |
| 数据 | shard0（train_webnovel_v2.txt），val = `val_webnovel_v2.txt`（**两边同一份**） |
| 训练 | batch 64 / lr 8e-4 / warmup 1000 / cosine（total_steps 维持 30463）/ `--max-steps 5050` / seed 42 / fp16+compile |
| 参数量 | 51,421,056 (v6144) / 52,602,752 (v8192) |

**全量 val（143,893,657 字符、全部 block）@ step 5000**：

| arm | vocab | val_loss (nats/tok) | chars/token | **bits/char** | bits/byte |
|---|---|---|---|---|---|
| v6144 | 6144 | 3.5999 | 1.2429 | 4.1784 | 1.4146 |
| v8192 | 8192 | 3.6906 | 1.2861 | **4.1398** | **1.4016** |

→ **Δ bits/char = −0.0386（−0.92%）**，Δ bits/byte = −0.0131（−0.92%）→ **8192 更好**。
**10/10 个验证点全部偏向 8192**（−0.99% ~ −1.49%，全程一致）；扣除"同 step 下 8192 多看 3.40% 文本"
的贡献后仍有 ≈ **−0.83%**。

- **⚠️ 方法学要点（本次最大产出之一）**：**per-token val_loss 跨 vocab 不可比**——
  词表越大单 token 承载信息越少，loss 天然更低。照 per-token 读会得出**完全相反**的结论
  （3.5999 vs 3.6906 看着 8192 差 2.5%）。必须归一化到 **bits/char 或 bits/byte**：
  `bits/char = (val_loss_nats / ln2) / chars_per_token`，其中 chars_per_token 取**完整 val 语料**实测值。
- **结论**：**vocab 定版 8192**；100M 主线 run 采用 8192。
- **产物**：`tok_exp/tok/tok_v{4096,6144,8192}*.pkl`、`tok_exp/eval_vocab.json`、`tok_exp/tail_vocab.json`、
  `tok_exp/tok_vocab_summary.png`、`log/vocab对照6144vs8192/`（两臂 train.log / step/val CSV / eval JSON / 汇总图）

---

## v3_beta 100M + shard1/2 新数据（2B）— 完成（2026-09-11）

- **Experiment ID**：`v3_beta_100M_ctx512_2B`
- **status**：`complete`
- **commit**：`e22aa33`（本 run **未改任何模型代码**，纯配置 + 数据）
- **配置**：`docs/experiment_config_100M.yaml` ｜ launcher：`scratch/cloud_run_100m.sh`

| 项 | 值 |
|---|---|
| 模型 | **16L / 704d / 11H**（head_dim 64）/ swiglu h=1888 / tie / rope / dropout 0.1 |
| 参数量 | **101,457,280**（实测；vocab 8192） |
| 数据 | **shard1+2**（`webnovel_{1,2}.jsonl` 清洗）→ train **2,489,862,301 字符 = 1,962,318,207 token**；val 291,364,644 字符 = 228,601,630 token |
| 训练 | batch 64 / block 512 / **lr 7e-4** / warmup 2000 / cosine（min 4.375e-5）/ wd 0.05 / grad-clip 1.0 / **epochs 1** / val_every 5000 / seed 42 / fp16+compile |
| 步数 | 59,886（完成 59,863，epoch 1.000） |
| **best val** | **3.0610** @ step 59863（epoch 末验证）；train_loss EMA 3.0780 |
| 耗时 | **265.5 分钟 = 4.42 小时** |
| 吞吐 | 3.76 step/s，123,136 tok/s（含验证/存盘） |
| 显存峰值 | allocated **18,403** / reserved **19,846** MiB（上限 24,564） |
| AMP skip | 23 / 59,863 步（0.04%，无害） |

**12 个验证点**（单调下降，无平台期/反弹）：

| step | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k | 45k | 50k | 55k | 59.9k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 3.6336 | 3.4535 | 3.3596 | 3.2999 | 3.2505 | 3.2087 | 3.1706 | 3.1368 | 3.1083 | 3.0846 | 3.0689 | **3.0610** |

- **备注（定位与口径，务必注意）**：
  - **非单变量实验**：本 run 同时动了 模型规模（51.4M→101.5M，1.97×）与 数据（1B→1.96B，且**完全不同的书**），
    另叠加 vocab（6144→8192，依据上方短测）与 FFN（relu→swiglu，依据 FFN 短程对照）
    → 属「综合升级」性质，**不要按单变量标准归因**（同 v2 家族定义）。
  - **★ 新评估空间**：本 run 的 val 是 **shard1+2 自己的 val**（新 split）。按「实验比较规则」，
    validation split 变化 → **loss 不可直接比较**。因此 **3.0610 不在 v2/v3 可比链内**，
    **不可与 v3_alpha 3.2097 排名**；它是新空间的第一个基准点。
  - 若只做量级参考（**不构成严格对比**）：换算 bits/char，v3_alpha ≈ 3.710 vs 本 run ≈ 3.465（−6.6%）。
  - 50M/1B 与 100M/1.96B 均约 19–20 token/参数，两边都落在 Chinchilla 附近，比例本身是正确的。
- **两个运行时坑（已写进脚本注释）**：
  1. **`ulimit -n 65535` 必须**：`ShardMemmap.__init__` 会一次性 `np.memmap` 打开**所有**分片，
     本语料 **1660 片 > 容器默认 soft limit 1024** → `OSError: [Errno 24] Too many open files`。
     shard0（828 片）时未暴露，语料翻倍才踩到。
  2. **自动关机在容器内不可用**：`/usr/bin/shutdown` 实际只 `kill supervisord`，
     容器 PID 1 会把它**重启** → 实例不关机。跑完需**人工在控制台关机**。
- **产物**：`log/100M参数_v3_2Btokens/`（`run_config.txt` / `step_history_train_webnovel_shard12.csv` /
  `val_history_train_webnovel_shard12.csv` / `train_100m.log` / `training_curve_100m.png`）；
  云端 `/root/result_100m/`（`checkpoint_best.pt` 389MB / `checkpoint_latest.pt` 1.2GB / tokenizer）
- **下一步**：本 run 末段（50k→59.9k 仅降 0.024）下降已明显放缓；数据侧继续扩（shard3+）或参数侧继续扩
  需先定新评估空间的比较口径（建议后续 run 固定用 shard1+2 的 val，形成新的可比链）。
