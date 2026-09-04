# MiniGPT Project Status — 项目状态文件

> **规则：修改任何代码前先读本文件**，了解当前阶段、约定与坑点。
> 训练完成后由助手根据用户口述在 `docs/EXPERIMENT_LOG.md` 追加实验记录。
> 完整工作流见文末「AI 操作规则 — 标准工作流（最终版）」。
> 最后更新：2026-09-05

## 一句话定位

从零 PyTorch 实现的中文小说 GPT（自研 BPE 分词器 + 手写 Transformer + 云端 4090 训练 + FastAPI demo），当前在 **B 组（能力深度）**：扩数据到 ~1B token + 更大模型 + 更长上下文。

## 当前阶段

- [ ] **B 组暂停中**（用户决定过几天再训）→ 恢复步骤见 `RESUME_B_PLAN.md`（根目录）
- [x] 语料下载完成：`D:\小说\webnovel\webnovel_{0,1,2}.jsonl` 各 ~3.9GB（合计 11.7GB，~148 万行 / ~2790 本，均已验证无 JSON 错误）
- [ ] 待办：本地清洗 shard0 → `data/train|val_webnovel_v2.txt`（`python data/prepare_webnovel.py --shards 0`）
- [ ] v2 实验配置（**候选冻结**，状态：**待训练验证**）：`docs/experiment_config_v2.yaml`（10L/384d/bs512/vocab6144/tie，webnovel ~1B token，pack，min_lr=5e-5）—— **仅为计划，尚未验证有效，不要当作已确定结论**
- [ ] 训练目标：v2 用 webnovel_v2 语料（1 分片 ≈ 10 亿 token），期望 val < 3.636（20M 版纪录）

## Current Best Checkpoint

- **Best known model: 20M_final**（10L/8H/384d/bs256/vocab6144/tie）
- **Best val_loss: 3.636**（nats；checkpoint: `result_20m_all/checkpoint_best.pt`）
- **Do not overwrite unless a new experiment improves it.**

> 最新 ≠ 最好：跑失败/半成品实验时，**不要覆盖 `checkpoint_best.pt`**，也不要以最新 checkpoint 当作最佳结论。

## 实验事实表

| 版本 | 参数 | 配置 | 语料 | val loss | commit |
|---|---|---|---|---|---|
| 百合基线 | 7.2M | 7L/8H/256d/bs128/vocab3256 | 百合 35 本 | 3.79 | ≈85460d5 |
| LN 修复 | 6.3M | 6L/8H/256d/bs256/vocab6144/tie | 轻小说 v0 | 4.188 | ≈7a4bf61 |
| 20M 最终 | 20M | 10L/8H/384d/bs256/vocab6144/tie | v0+v1 (4.16亿 token) | 3.636 | ≈20393f0 |

> 完整记录在 `docs/EXPERIMENT_LOG.md`；豆包复盘报告在 `docs/report_output/`。
> 比较条件见「实验比较规则」一节（val loss 单位 nats）。

## 已验证结论

- **增加参数 + 数据有效**：20M(3.636) < LN 修复版(4.188)（同 vocab6144/tokenizer 下可比）
- **验证集按文件/按书划分**显著降低 val loss（4.5 → 3.79，百合基线）
- **标准 LayerNorm**（有偏方差 + eps）收敛更稳、val 更好
- **pack 模式**（stride=block_size）使 100M+ 语料每 epoch 从小时级降到分钟级
- **SDPA 训练路径**省显存，是 block_size > 256 的前提
- **gap = val − train_eval**（eval 无 dropout）是判断过拟合的正确指标；20M 版 gap 0.020，未过拟合
- **tie_embeddings + 权重去重**：大词表必备，省 ~833K 参数
- **KV cache 增量推理**与整序列重算逐位置 logits 一致（max diff <1e-4），~1.6-2.2x

## 当前未知问题

- v2 配置（1B token / block 512 / batch 128）**尚未训练验证**，能否达到 val < 3.636 未知
- 20M 模型继续堆数据（4.16亿 → 10 亿 token）是否继续降 loss 未验证
- block 512 在 SDPA 下的显存占用、训练速度、长程收益未实测
- 更大模型（35M/50M）相对 20M 的增益未验证
- 中文生成质量无系统评估，目前仅主观观感

## 下一阶段计划

1. 恢复 B 组：数据扩到 ~1B token（webnovel_v2，先 1 分片）+ 模型 20M → 35M/50M + 上下文 512
2. v2 首跑验证（配置见 `docs/experiment_config_v2.yaml`），目标 val < 3.636
3. 按下方 Scaling 原则**单变量**逐步验证，一次只动一个变量

### Scaling 原则

当前已验证：**增加参数有效**（20M < LN 修复版）。
下一阶段优先验证，按顺序：
1. 参数规模（20M → 35M/50M）
2. context 长度（block 512）
3. 数据规模（4.16亿 → 10 亿 token）

**不要同时修改 tokenizer 和训练目标**，避免一次大改导致无法归因。

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

1. 验证集必须**按文件/按书划分**，不能按 token 顺序切（否则高估 loss）
2. 判断过拟合看 `gap = val - train_eval`（eval 无 dropout），不是裸 val
3. tokenizer v2 哨兵：SEP=-1/DEAD=-2 是节点值不参与合并，别当邻居处理
4. resume 会重锚 cosine（warm restart，lr 跳回）——加轮次续训的正常现象
5. Windows 下 encode 并行退回单线程；云端多进程才有加速
6. tqdm 写 stderr 会让 pwsh 误报 exit 1——看输出内容而非退出码

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
