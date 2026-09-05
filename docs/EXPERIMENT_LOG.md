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
