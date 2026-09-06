# MiniGPT Project Status — 项目状态文件

> **规则：修改任何代码前先读本文件**，了解当前阶段、约定与坑点。
> 训练完成后由助手根据用户口述在 `docs/EXPERIMENT_LOG.md` 追加实验记录。
> 完整工作流见文末「AI 操作规则 — 标准工作流（最终版）」。
> 最后更新：2026-09-05

## 一句话定位

从零 PyTorch 实现的中文小说 GPT（自研 BPE 分词器 + 手写 Transformer + 云端 4090 训练 + FastAPI demo），当前在 **B 组（能力深度）**：扩数据到 ~1B token + 更大模型 + 更长上下文。

## 当前阶段

- [x] **v2 50M 完成**（2026-09-06，`v2_50M_ctx512_1B`）：51.4M（12L/576d/9H）/ 1B token / ctx512，**best val 3.6154** @ step 30454（webnovel_v2 评估空间新 best，详见 EXPERIMENT_LOG）
- [x] **★ 首个严格可比参数对照成立**：50M@1B(**3.6154**) < 35M@1B(3.7076) < 35M@2B(3.6372) → **参数 scaling 收益 > 重复数据收益**（同 tokenizer/语料/val/batch/LR，仅参数不同）
- [x] **v2 35M 两轮训练完成**（2026-09-05→06，`v2_35M_ctx512_1B_E1+E2`）：35M / **2B token**（998M×2 同语料二遍）/ ctx512 / vocab6144，**v2 best = 35M v2 Epoch 2：val **3.6372** / train_eval **3.6063** / gap **0.0310** @ step 60915**（v2 evaluation space best，不与旧 20M raw loss 直接排名；详见 EXPERIMENT_LOG）
- [x] Epoch 2 resume 修复完成并验证：`docs/RESUME_AUDIT.md`（next_epoch 语义 / `--lr-scheme const` 恒温 5e-5 / AMP 更新门控 / 精确 tokens_seen），云端与本地冒烟均通过
- [x] 语料下载完成：`D:\小说\webnovel\webnovel_{0,1,2}.jsonl` 各 ~3.9GB（合计 11.7GB，~148 万行 / ~2790 本，均已验证无 JSON 错误）
- [x] 清洗完成：shard0 → `data/train|val_webnovel_v2.txt`（云端 `/root/autodl-tmp/data/`，train 12.4 亿字符 / val 1.44 亿字符）
- [x] v2 实验配置已跑（**已训练验证**）：`docs/experiment_config_v2.yaml`（**10L/512d ≈35M**/bs512/vocab6144/tie，webnovel ~1B token，pack，min_lr=5e-5）—— 首次配置 batch128，随后实际成功运行配置调整为 batch64（总 token 不变）
- [ ] 下一步（B 组主线）：**50M + shard1/2 新数据**（~2B 全新 token，参数与数据双升的真 scaling）——50M@1B 末段未饱和，新数据预期继续显著降

> **分别记录、不要混排**：
> - Historical best on old evaluation：20M val = 3.636（旧 tokenizer/语料/val 集/ctx256）
> - Current webnovel_v2 eval space best：50M val = **3.6154**（新 tokenizer/语料/val 集/ctx512；35M E2 3.6372 次之）
> 两者处于不同评估空间，**不构成同一排行榜**。

## Current Best Checkpoint

- **Best known model (old eval): 20M_final**（10L/8H/384d/bs256/vocab6144/tie）
- **Best val_loss (old eval): 3.636**（nats；旧评估空间：旧 tokenizer/语料/val 集/ctx256；本地归档: `result/20M参数+416Mtokens/checkpoint_best.pt`）
- **Do not overwrite unless a new experiment improves it.**
- v2 (35M, 2026-09-06, **Epoch 2**): val **3.6372** / train_eval **3.6063** / gap **0.0310**（**v2 评估空间 best**），**tokenizer/语料不同与 3.636 不可直接比**；checkpoint: `result/35M参数+998Mtokens/checkpoint_best.pt`（本地归档；云端 `/root/result_webnovel_v2/`）
- **v2 (50M, 2026-09-06): val 3.6154** / train_eval 3.5776 / gap 0.0378 —— **webnovel_v2 评估空间当前 best**（同空间与 35M 直接可比）；checkpoint: `result/50M参数+998Mtokens/checkpoint_best.pt`（本地归档；云端 `/root/result_50m/`）

> 最新 ≠ 最好：跑失败/半成品实验时，**不要覆盖 `checkpoint_best.pt`**，也不要以最新 checkpoint 当作最佳结论。

## 实验事实表

| 版本 | 参数 | 配置 | 语料 | val loss | commit |
|---|---|---|---|---|---|
| 百合基线 | 7.2M | 7L/8H/256d/bs128/vocab3256 | 百合 35 本 | 3.79 | ≈85460d5 |
| LN 修复 | 6.3M | 6L/8H/256d/bs256/vocab6144/tie | 轻小说 v0 | 4.188 | ≈7a4bf61 |
| 20M 最终 | 20M | 10L/8H/384d/bs256/vocab6144/tie | v0+v1 (4.16亿 token) | 3.636 | ≈20393f0 |
| v2 首跑 (Epoch 1) | 35M | 10L/8H/512d/bs512/vocab6144/tie（实际 batch64） | webnovel_v2 shard0 (9.98亿) | 3.7076 | 614d325 |
| v2 Epoch 2 | 35M | 同 E1（const LR 5e-5 恒温续训） | webnovel_v2 shard0（第二遍，累计 19.96 亿） | 3.6372 | 16f0c1a |
| v2 50M | 51.4M | 12L/9H/576d/bs512/vocab6144/tie | webnovel_v2 shard0 (9.98亿, 单轮) | **3.6154** | fffbffe |

> v2 严格可比链（同评估空间）：50M@1B 3.6154 < 35M@2B 3.6372 < 35M@1B 3.7076 → **参数 scaling 收益 > 重复数据收益**（详见 EXPERIMENT_LOG 阶段记录）。

> 完整记录在 `docs/EXPERIMENT_LOG.md`；豆包复盘报告在 `docs/report_output/`。
> 比较条件见「实验比较规则」一节（val loss 单位 nats）。

## 已验证结论

- **参数 scaling 收益 > 重复数据收益**：50M@1B(3.6154) < 35M@2B(3.6372) < 35M@1B(3.7076)——本项目首个**同 tokenizer/语料/val/batch/LR 严格可比**结论（2026-09-06）
- **增加参数 + 数据有效**：20M(3.636) < LN 修复版(4.188)（同 vocab6144/tokenizer 下可比）
- **验证集按文件/按书划分**：为避免数据泄漏（按 token 顺序切可能把整本书放进 val，评估失真）而设，使评估结果更可靠；不同实验版本的 loss 变化（如 4.5 → 3.79）**不能归因于单一因素**
- **标准 LayerNorm**（有偏方差 + eps）收敛更稳、val 更好
- **pack 模式**（stride=block_size）使 100M+ 语料每 epoch 从小时级降到分钟级
- **SDPA 训练路径**省显存，是 block_size > 256 的前提
- **gap = val − train_eval**（eval 无 dropout）是判断过拟合的正确指标；20M 版 gap 0.020，未过拟合
- **tie_embeddings + 权重去重**：大词表必备，省 ~833K 参数
- **KV cache 增量推理**与整序列重算逐位置 logits 一致（max diff <1e-4），~1.6-2.2x

## 当前未知问题

- v2 E1（3.7076）数值上未破 20M 纪录（3.636），但 **tokenizer/语料不同不可直接比**；同 tokenizer + 同 val 集下的 35M vs 20M 增益仍无可比数据
- 20M 模型继续堆数据（4.16亿 → 10 亿 token）是否继续降 loss 未验证
- block 512 相对 256 的长程收益未在同一语料上验证
- 更大模型（50M）相对 35M/20M 的增益未验证
- 中文生成质量无系统评估，目前仅主观观感

## 下一阶段计划

1. 恢复 B 组：数据扩到 ~1B token（webnovel_v2，先 1 分片）+ 模型 20M → 35M/50M + 上下文 512
2. v2 首跑（Epoch 1）已完成（配置见 `docs/experiment_config_v2.yaml`，best val 3.7076）
3. **Epoch 2（待训练诊断后决定）**：先修 scheduler warning（optimizer.step / lr_scheduler.step 顺序）、核对 resume 后 LR 锚定与 checkpoint 状态，再决定是否继续训练；目标先观察第二轮改善幅度与 gap 走势

### Scaling 原则（两阶段）

**第一阶段 — v2 综合升级实验（非单变量）**：
v2 同时升级三个维度——模型扩大（20M → 35M/50M）、context 增加（block 256 → 512）、数据增加（4.16亿 → ~1B token）。
目的：**验证整体 scaling 收益**（v2 有效 = 三个维度整体方向正确）。
> v2 是综合升级实验，**不属于严格单变量实验**，请勿要求它按单变量标准归因。

**第二阶段 — 单变量拆解**：
若 v2 有效，再通过单变量实验分别拆解各维度贡献：
1. 参数规模贡献
2. context 贡献
3. 数据规模贡献

**约束**：不要同时修改 tokenizer 和训练目标（tokenizer/vocab 改动独立于 scaling 实验，避免一次大改导致无法归因）。

## 代码约束（改代码时遵守）

- **采样默认值**：temperature 0.8 / top_p 0.9 / repetition_penalty 1.15（`model/sampling.py`）
- **LR 调度**：warmup 500 + cosine 衰减到 **max_lr×0.1**（`--min-lr-ratio 0.1`，不再到 0）
- **tie_embeddings 默认开**；vocab 6144；tokenizer 采样 4M 字符
- **数据模式**：小语料 slide（stride=1），100M+ 语料 pack（stride=block_size）
- **block_size 默认 256**；>256 需 SDPA（`model/attention.py` 已含 SDPA 训练路径）
- 生成必须走共享引擎 `model/generation.py`（KV cache + 超窗重建），不要另写循环
- 旧语料（百合/lightnovel v0/v1）不动，新语料用后缀区分（webnovel_v2 等）

## 实验比较规则

以下情况 **loss 不可直接比较**：
- vocab 变化
- tokenizer 变化
- validation split 变化
- loss 计算方式变化

**可比较**的条件：
- 同 tokenizer
- 同 val 集
- 同 loss 定义

## 关键坑点（踩过）

1. 验证集必须**按文件/按书划分**，不能按 token 顺序切——按 token 切可能造成数据泄漏（val 含训练未见的整本书）、高估 loss，评估不可靠
2. 判断过拟合看 `gap = val - train_eval`（eval 无 dropout），不是裸 val
3. tokenizer v2 哨兵：SEP=-1/DEAD=-2 是节点值不参与合并，别当邻居处理
4. resume 会重锚 cosine（warm restart，lr 跳回）——加轮次续训的正常现象
5. Windows 下 encode 并行退回单线程；云端多进程才有加速
6. tqdm 写 stderr 会让 pwsh 误报 exit 1——看输出内容而非退出码

## Experiment ID 规则

统一实验命名格式：

```
v{版本}_{模型}_ctx{context}_{数据规模}
```

示例：
- `v1_20M_ctx256_400M`（20M 最终版）
- `v2_35M_ctx512_1B`（v2 综合升级实验）

以下位置**统一使用该 ID**（新建实验时按格式命名）：
- checkpoint 输出目录
- log 文件（step/val 历史 CSV、run_config）
- `EXPERIMENT_LOG.md` 记录
- HTML 报告

## 日志规范

- **每次训练完成后**：在 `docs/EXPERIMENT_LOG.md` 末尾追加一行（日期/commit/模型配置/数据版本/token 数/训练时间/best val/备注）
- `log/step_history_{语料}.csv`：每步一行 `step,epoch,tokens_seen,train_loss,val_loss,lr,time`
  - **tokens_seen 跨 run 连续**，比较 400M/1B tokens / 不同 batch 看这列
- `log/val_history_{语料}.csv`：每次验证一行（含 gap/train_eval/is_best）
- `log/run_config.txt`：每次启动的完整配置快照（追加，resume 也单独一段）

## 环境信息

- 训练机：AutoDL 4090
- **Secrets（ssh 地址、key 目录、代理端口等）另行保存，不入库、不写本文件**——本文件可能上传 GitHub / 转发他人 / 交给 AI
- 本地不跑重训练；轻量 python/pytest 可以
- GitHub: `wbb316/minigpt-chinese`（本地 git 仓库，main 分支）

## AI 操作规则 — 标准工作流（最终版）

**修改代码前（顺序执行）：**
1. 读 `docs/MiniGPT_Project_Status.md`（本文件：当前阶段/约定/坑点）
2. 读 `docs/EXPERIMENT_LOG.md`（了解历史实验，避免重复/冲突）
3. 读本次实验配置 `docs/experiment_config_xxx.yaml`（确认当前实验目标与参数）
4. 说明预期影响
5. 未经确认**不得同时修改多个实验变量**

**修改代码 → 运行训练：**
6. 修改代码（遵守「代码约束」）
7. 运行训练
8. 保存产物：`log/`（step/val 历史 CSV + run_config.txt）、`checkpoint`（best/latest）、`config`（本次实验的 yaml）

**训练结束后（顺序执行）：**
9. 更新 `docs/EXPERIMENT_LOG.md`（末尾追加一行：日期/commit/模型配置/数据版本/token 数/训练时间/best val/备注）
10. 更新 `docs/MiniGPT_Project_Status.md`（**仅当**产生新 best 或阶段变化：更新 Current Best Checkpoint / 实验事实表 / 当前阶段）
