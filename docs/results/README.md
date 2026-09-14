# 训练结果归档：100M（v3_beta）与 150M（v3_gamma）

> 归档日期：2026-09-14 ｜ 来源：云端训练产物，**逐字节拷贝**（本地与云端 md5 一致）
> 完整实验记录见 [`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md) ｜ 项目状态见 [`../MiniGPT_Project_Status.md`](../MiniGPT_Project_Status.md)
> **权重（`*.pt` / `*.pkl`）不入 git**：100M best 388.8 MB、150M best 594.3 MB，均超出 GitHub 单文件 100 MB 硬限；
> 权重只归档在本地 `result/` 与云端 `/root/result_*/`。

---

## ⚠️ 先读这条：这两个 run 属于**第三个评估空间**

本项目有 **3 个互不可比的评估空间**（tokenizer / 语料 / validation split 均不同）：

| 空间 | train / val | vocab | 成员 |
|---|---|---|---|
| ① 旧空间 | 百合+轻小说 v0/v1，ctx256 | 3256 / 6144 | 7.2M、6.3M、20M |
| ② webnovel_v2 空间 | shard0，ctx512 | 6144 | v2 35M/50M、**v3_alpha 50M** |
| ③ **shard1+2 空间** | shard1+2 自有 split，ctx512 | **8192** | **v3_beta 100M、v3_gamma 150M（本目录）** |

**本目录的两个数字只能和彼此比，不能与 v3_alpha（3.2097）或 v2 系排名。** 跨空间只能看量级趋势。

---

## 两个 run 的关键指标（全部实测）

| | **100M（v3_beta）** | **150M（v3_gamma）** |
|---|---|---|
| 实验 ID | `v3_beta_100M_ctx512_2B` | `v3_gamma_150M_ctx512_3B` |
| 参数量 | **101,457,280** | **148,079,104** |
| 结构 | 16L / 704d / 11H（head_dim 64） | 20L / 768d / 12H（head_dim 64），SwiGLU h=2048 |
| 训练 token | **1.96B** | **2,931,494,589（2.93B）** |
| 语料 | shard1+2（全新 token，单轮） | shard0+1+2（单轮） |
| vocab / ctx / dropout | 8192 / 512 / 0.1 | 8192 / 512 / 0.1 |
| 有效 batch | 64 | 64（micro 32 × grad-accum 2；batch64 实测 OOM） |
| lr / warmup / min_lr | 7e-4 / 2000 / 4.38e-5 | 6.0e-4 / 2500 / 3.75e-5 |
| total_steps | 59,886 | 89,463（**成功 update 89,432**，31 次 AMP skip） |
| **best val** | **3.0610** @ step 59,863 | **2.9675** @ step 89,432 |
| train_eval / gap | 2.9995 / **+0.0616** | 2.9109 / **+0.0566** |
| 墙钟 / 吞吐 | 4.42 h / 3.76 step/s | 8.92 h（534.5 min）/ 2.79 step/s |
| 显存峰值 | — | allocated 13,539 / reserved 13,934 MiB |
| 验证点 | 12 个 | 18 个 |
| tokenizer | 冻结 v8192（未重训） | 同左（开跑前 md5 校验一致） |

---

## 结论（严格口径）

**shard1+2 空间的可比链（同 tokenizer / 同 val / 同 tokens-per-update，终点 vs 终点）：**

```
v3_gamma 150M/2.93B  2.9675  <  v3_beta 100M/1.96B  3.0610
Δ = −0.0935 nats（−3.05%）
```

**但这个差值不能单独归因于 scaling**，它至少混了四件事：

1. **参数量 1.46×**（101.5M → 148.1M）
2. **数据量 1.49×**（1.96B → 2.93B）
3. **lr 7e-4 → 6e-4**（用户规定）
4. **warmup 2000 → 2500**（用户规定）

要拆开各维度贡献，需要「同 token 预算 + 同 lr 调度」的对照实验（尚未做）。

### ⚠️ 方法学：这两个 run **不能按同 step 比**

150M 的 cosine 铺在 89,463 步上、100M 铺在 59,886 步上 → **同一 step 的 lr 并不相等**（150M 更高）。
实测在中段（40k–50k）150M 反而略差 **+0.0007 ~ +0.0038 nats**，到 55k 才反超：

| step | 100M val | 150M val | Δ | 100M lr | 150M lr |
|---|---|---|---|---|---|
| 40000 | 3.1368 | 3.1375 | +0.0007 | 2.17e-4 | 3.79e-4 |
| 45000 | 3.1083 | 3.1121 | +0.0038 | 1.45e-4 | 3.29e-4 |
| 50000 | 3.0846 | 3.0876 | +0.0030 | 8.99e-5 | 2.78e-4 |
| 55000 | 3.0689 | 3.0637 | **−0.0052** | 5.52e-5 | 2.29e-4 |

**中段同 step 的胜负主要是调度错配，不是模型质量差。** 自洽的比法是**终点 vs 终点**（双方 cosine 都走到 min_lr）。

---

## 文件说明

| 文件 | 内容 | 数据行数 |
|---|---|---|
| `100M_v3_beta_val_history.csv` | 100M 的 **12 个验证点**（`step,where,val,train_eval,gap,lr,is_best,no_improve,best_val,wall_time`） | 12 |
| `100M_v3_beta_step_history.csv` | 100M 训练曲线（每 20 步一行，含 `tokens_seen / train_loss / lr / tokens_per_sec`） | 3,005 |
| `150M_v3_gamma_val_history.csv` | 150M 的 **18 个验证点**（比 100M 多一列 `time_unix`） | 18 |
| `150M_v3_gamma_step_history.csv` | 150M 训练曲线（末尾若干行带 `val_loss` 的是验证追加行） | 4,490 |

- **loss 单位是 nats/token**；随机基线 = `ln(8192)` = **9.01**。
- `gap = val − train_eval`（eval 模式、无 dropout）用来判断过拟合：两次都全程为正且 < 0.07，**未过拟合**。
- 更早的空间（v3_alpha / v2 系）不在此目录，见 `log/` 本地归档（`log/` 被 `.gitignore` 排除，不入库）。

## 复现对比

```powershell
# 同 step 对照表 + 终点差（脚本末尾自动打印 −0.0935 nats）
python scratch\compare_150m_vs_100m.py
```

## 训练配置

见 `docs/experiment_config_100M.yaml` 与 `docs/experiment_config_150M.yaml`（后者含 smoke 实测整节）。
