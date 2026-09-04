# MiniGPT Project Status — 项目状态文件

> **规则：修改任何代码前先读本文件**，了解当前阶段、约定与坑点。
> 训练完成后由助手根据用户口述在 `docs/EXPERIMENT_LOG.md` 追加实验记录。
> 最后更新：2026-09-04

## 一句话定位

从零 PyTorch 实现的中文小说 GPT（自研 BPE 分词器 + 手写 Transformer + 云端 4090 训练 + FastAPI demo），当前在 **B 组（能力深度）**：扩数据到 ~1B token + 更大模型 + 更长上下文。

## 当前阶段与进行中任务

- [ ] **B 组暂停中**（用户决定过几天再训）→ 恢复步骤见 `RESUME_B_PLAN.md`（根目录）
- [ ] 语料下载：3 个分片 `D:\小说\webnovel\webnovel_{0,1,2}.jsonl`（各 ~3.9GB，后台任务曾运行）
- [ ] v2 实验配置已冻结：`docs/experiment_config_v2.yaml`（10L/384d/bs512/vocab6144/tie，webnovel ~1B token，pack，min_lr=5e-5）
- [ ] 训练目标：v2 用 webnovel_v2 语料（1 分片 ≈ 10 亿 token），期望 val < 3.636（20M 版纪录）

## 模型/实验事实清单（最新在后）

| 版本 | 参数 | 配置 | 语料 | val loss | commit |
|---|---|---|---|---|---|
| 百合基线 | 7.2M | 7L/8H/256d/bs128/vocab3256 | 百合 35 本 | 3.79 | ≈85460d5 |
| LN 修复 | 6.3M | 6L/8H/256d/bs256/vocab6144/tie | 轻小说 v0 | 4.188 | ≈7a4bf61 |
| 20M 最终 | 20M | 10L/8H/384d/bs256/vocab6144/tie | v0+v1 (4.16亿 token) | 3.636 | ≈20393f0 |

> 完整记录在 `docs/EXPERIMENT_LOG.md`；豆包复盘报告在 `docs/report_output/`。
> val loss 单位 nats，**词表不同不可横向比**（随机基线 ≈ ln(vocab)，ln(6144)≈8.72）。

## 代码约定（改代码时遵守）

- **采样默认值**：temperature 0.8 / top_p 0.9 / repetition_penalty 1.15（`model/sampling.py`）
- **LR 调度**：warmup 500 + cosine 衰减到 **max_lr×0.1**（`--min-lr-ratio 0.1`，不再到 0）
- **tie_embeddings 默认开**；vocab 6144；tokenizer 采样 4M 字符
- **数据模式**：小语料 slide（stride=1），100M+ 语料 pack（stride=block_size）
- **block_size 默认 256**；>256 需 SDPA（`model/attention.py` 已含 SDPA 训练路径）
- 生成必须走共享引擎 `model/generation.py`（KV cache + 超窗重建），不要另写循环
- 旧语料（百合/lightnovel v0/v1）不动，新语料用后缀区分（webnovel_v2 等）

## 关键坑点（踩过）

1. 验证集必须**按文件/按书划分**，不能按 token 顺序切（否则高估 loss）
2. 判断过拟合看 `gap = val - train_eval`（eval 无 dropout），不是裸 val
3. tokenizer v2 哨兵：SEP=-1/DEAD=-2 是节点值不参与合并，别当邻居处理
4. resume 会重锚 cosine（warm restart，lr 跳回）——加轮次续训的正常现象
5. Windows 下 encode 并行退回单线程；云端多进程才有加速
6. tqdm 写 stderr 会让 pwsh 误报 exit 1——看输出内容而非退出码

## 日志文件（云端训练时自动生成）

- `log/step_history_{语料}.csv`：每步一行 `step,epoch,tokens_seen,train_loss,val_loss,lr,time`
  - **tokens_seen 跨 run 连续**，比较 400M/1B tokens / 不同 batch 看这列
- `log/val_history_{语料}.csv`：每次验证一行（含 gap/train_eval/is_best）
- `log/run_config.txt`：每次启动的完整配置快照（追加，resume 也单独一段）

## 环境速查

- 训练机：AutoDL 4090（关机中；`ssh -p 24932 root@connect.westb.seetacloud.com`，key 在 `.autodl_key/`）
- 本地不跑重训练；轻量 python/pytest 可以
- 梯子代理：`127.0.0.1:7897`（下载 HF 语料需走代理）
- GitHub: `wbb316/minigpt-chinese`，本地已是 git 仓库（main 分支）
