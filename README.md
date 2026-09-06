# MiniGPT-Chinese

从零用 PyTorch 实现的迷你中文 GPT 文本生成模型 —— 一个从"语料 → 自研分词器 → 模型 → 训练 → 生成"完整链路的中文小说模型项目。

> 核心思想：**不依赖任何现成 NLP 库**（不用 HuggingFace / tiktoken / 预训练模型），
> BPE 分词器、Transformer、训练循环全部手写，在云端 4090 上从零训出自己的中文小说续写模型。

## 🎯 项目亮点

- **自研 BPE 分词器**：字节级 BPE，正则预分词 + 链表增量合并，4M 字符训练分钟级；大语料并行编码（uint16 分片缓存 + memmap，3.6GB 文本 → 998M token 几分钟）
- **从零实现 GPT**：LayerNorm / 正弦位置编码 / FeedForward / 多头因果注意力（SDPA + 手写双路径）/ 主模型拼装
- **完整训练链路**：语料清洗 → BPE → 数据切片（pack 打包）→ 训练（warmup+cosine、AMP、梯度裁剪、step 级验证、早停、断点续训）→ Web 生成
- **多代演进记录**：7.2M 百合单语料 → 6.3M LayerNorm 修复版 → 20M 双语料 → **v2 35M（webnovel 1B×2）→ v2 50M**

## 📊 模型演进与最佳结果

> loss 单位 nats。**评估空间不同不可直接横向比**：
> 旧空间 = 轻小说语料 + 旧 tokenizer；新空间（v2）= webnovel_v2 shard0 + 新 tokenizer v6144（两者同 tokenizer/语料/val 集，**可直接比**）。

| 版本 | 参数量 | 结构 | 语料 | val loss | 评估空间 |
|---|---|---|---|---|---|
| 百合基线 | 7.2M | 7L/8H/256d | 百合 35 本 | 3.79 | 旧 |
| LN 修复版 | 6.3M | 6L/8H/256d | 轻小说 v0 | 4.188 | 旧 |
| 20M 最终版 | 20M | 10L/8H/384d | 轻小说 v0+v1 (4.16亿 token) | 3.636 | 旧 |
| v2 35M E1 | 34.7M | 10L/8H/512d | webnovel_v2 shard0 (1B) | 3.7076 | **v2** |
| v2 35M E2 | 34.7M | 同 E1（恒温 5e-5 续训） | shard0 二遍（累计 2B） | 3.6372 | **v2** |
| **v2 50M** | **51.4M** | **12L/9H/576d** | webnovel_v2 shard0 (1B) | **3.6154** | **v2** |

### 🏆 核心 scaling 结论（2026-09-06，首个严格可比对照）

同 tokenizer/语料/val/batch/LR，仅参数量不同：

```
50M @ 1B (3.6154)  <  35M @ 2B (3.6372)  <  35M @ 1B (3.7076)
```

**参数 scaling 收益 > 重复数据收益**：50M 单轮 1B 打赢 35M 把同批数据训两遍的 2B。
（35M 各代 checkpoint 归档于 `result/`，见「项目结构」）

## 🧠 关键优化点（踩坑记录）

### 1. 验证集划分方式（最先修复的 bug）
按 token 顺序切分验证集会**高估 loss**（验证集可能包含某本训练没见过的整本书）。
改成**按文件划分**，val loss 从 4.5 → 3.79。

### 2. LayerNorm 标准化
早期用了无偏方差 + 不带 eps 除法的错误实现。改为标准实现
`(x - mean) / sqrt(var_biased + eps) * gamma + beta`，同规模下 val loss 明显改善。

### 3. BPE 分词器 v2（正则预分词 + 链表增量合并）
- 预分词 `\w+|[^\w\s]|\s+`：合并**永不跨标点/空白**，文本无缝覆盖、无片间缝隙
- 训练后端 `fast`（惰性最大堆）：vocab≥3256 时 2.5-6.5x 加速，与 legacy **逐 token 等价**（15 项等价性测试）
- 训练样本从语料**均匀撒 100 段**（采样 bug 修复：之前误取开头 2M）

### 4. 大语料编码管线（uint16 分片缓存）
- 3.6GB 文本的切分点**严格落在预分词空白块内部** → 并行分片编码与整段编码逐 token 一致（100M 级验证精确相等）
- `_split_points` 曾因 O(n_parts×n_ws) 双重循环在 12.4 亿字符（2390 万空白区段）上卡死数小时 → 改 numpy 二分（毫秒级）
- worker 直写 uint16 `.bin` 分片 + `ShardMemmap` 只读视图：998M token 从 int64 5.6GB → uint16 1.4GB，训练内存占用近零

### 5. 数据模式：slide vs pack
- `slide`（stride=1）：重叠滑窗，适合小语料
- `pack`（stride=block_size）：不重叠打包，100M+ 语料每 epoch 从小时级 → 分钟级

### 6. 过拟合判断修正（gap 指标）
验证时**额外算 eval 模式（无 dropout）的 train loss**，打印 `gap = val - train_eval` —— 这才是正确指标。

### 7. tie_embeddings + 权重去重
输入/输出嵌入共享权重，`dedupe_params` 避免被优化两次。

### 8. KV cache 增量推理
与整序列重算**逐位置 logits 一致**（max diff < 1e-4），~1.6-2.2x；`model/generation.py` 超窗自动重建。

### 9. SDPA（Flash Attention）训练加速
训练路径用 `F.scaled_dot_product_attention(is_causal=True)`，block 256+ 显存省 ~50%（block 512 的前提）。

### 10. 生成采样三参数（观感质量飞跃）
`model/sampling.py`：temperature 0.8 / top_p 0.9 / repetition_penalty 1.15（网页三滑块可调）。

### 11. 断点续训语义修复（v2 Epoch 2 前，详见 `docs/RESUME_AUDIT.md`）
- **epoch 语义**：checkpoint 存 `next_epoch`（epoch 末=epoch+1，步级=当前）——修复旧版 resume 会重复已完成 epoch 的 bug
- **LR 续跑**：`--lr-scheme const`（恒 min_lr 5e-5，continuation 用）——修复 resume 后 cosine 按新总步数重铺、LR 从 5e-5 突跳回 4.35e-4 的隐式跳变
- **AMP 更新门控**：仅 optimizer 真正完成参数更新才推进 scheduler/global_step/tokens_seen（消除 `lr_scheduler.step() before optimizer.step()` warning，记录 skipped 步数）
- **精确 tokens_seen**：`+= x.numel()`（修复末批 partial batch 多计 3,584 token）

## 🏗️ 项目结构

```
minigpt-chinese/
├── model/
│   ├── layers.py       # LayerNorm、正弦位置编码、FeedForward
│   ├── attention.py    # 多头因果注意力（SDPA 训练路径 + KV cache 手写路径）
│   ├── gpt.py          # GPT 主模型（tie_embeddings、KV cache、去重参数计数）
│   ├── sampling.py     # 采样工具：temperature / top_p / repetition_penalty
│   └── generation.py   # 共享自回归引擎（KV cache + 超窗重建）
├── data/
│   ├── tokenizer.py    # 自研 BPE（v2：预分词 + 链表增量合并 + fast 后端）
│   ├── token_cache.py  # uint16 分片缓存 + ShardMemmap（大语料编码管线）
│   ├── dataset.py      # TextDataset（slide）/ PackedDataset（pack）
│   ├── prepare_corpus.py / prepare_lightnovel.py / prepare_webnovel.py
├── train/
│   └── train.py        # 主训练脚本（resume/lr-scheme/AMP 门控/tokens_seen 精确计数）
├── app/
│   ├── server.py       # FastAPI 后端（默认加载 v2 35M，架构自动推断）
│   └── templates/index.html  # 前端（温度/top_p/重复惩罚滑块）
├── test/               # pytest（56 个用例，含 BPE 等价性 / encode 等价性）
├── generate.py         # 生成（默认 v2 35M）+ KV cache 一致性验证
├── visualize_attention.py  # 注意力热力图
├── result/             # 模型产物归档（按 参数量+token数 分子目录）
│   ├── 35M参数+998Mtokens/   # v2 35M（E1+E2 完整）
│   ├── 50M参数+998Mtokens/   # v2 50M ← 当前最优（checkpoint + 训练曲线 + 热力图）
│   ├── 20M参数+416Mtokens/   # 20M 最终（旧空间）
│   └── ...
├── log/                # 训练日志归档（按实验分子目录：step/val 历史 CSV、run_config）
└── docs/
    ├── MiniGPT_Project_Status.md  # 项目状态文件——改代码前先读
    ├── EXPERIMENT_LOG.md          # 训练实验日志（含各阶段详细记录）
    ├── RESUME_AUDIT.md            # resume 语义审查 + 修复报告
    ├── experiment_config_v2.yaml / experiment_config_50M.yaml
    └── report_output/             # 实验复盘报告（HTML/md）
```

## 🚀 快速开始

```bash
pip install -r requirements.txt

# 测试（56 个用例）
python -m pytest test/ -v

# 生成测试（默认加载 v2 35M；用 50M 就传 --ckpt）
python generate.py --demo

# Web demo（浏览器打开 http://127.0.0.1:8000）
python app/server.py

# 指定其他模型（如 v2 50M）
python generate.py --ckpt result/50M参数+998Mtokens/checkpoint_best.pt \
                   --tokenizer result/50M参数+998Mtokens/tokenizer_best.pkl
```

## 🧪 测试

56 个用例全部通过：BPE（训练 fast/legacy 等价性、encode 等价性 15 项）、编码管线（分片一致性、memmap 切片）、模型/注意力、数据集、采样器、过拟合冒烟等。

## 📅 Roadmap

- [x] 阶段1：PyTorch 复习（CIFAR/MNIST）
- [x] 阶段2：从零写 GPT 模型
- [x] 阶段3：数据管线 + 自研 BPE tokenizer
- [x] 阶段4：训练 + 可视化（loss 曲线）
- [x] 阶段5：实验体系（对比实验 + 分析）
- [x] 阶段6：后端 API + 前端页面
- [x] 阶段7：优化迭代（LN 修复、pack 模式、SDPA、KV cache、采样参数、20M 训练）
- [x] 阶段8：v2 综合升级（webnovel 1B×2、35M→50M、block 512、resume 语义修复）—— 参数 scaling 结论成立
- [ ] 阶段9：v2 50M + shard1/2 新数据（~2B 全新 token，参数×数据双升）
- [ ] 阶段10：部署上线

## 📜 数据说明

- webnovel_v2：`D:\小说\webnovel\webnovel_{0,1,2}.jsonl`（3 shard × ~3.9GB，清洗后 shard0 → train/val_webnovel_v2.txt，12.4 亿字符 / 1.44 亿字符）
- 轻小说语料 v0/v1（旧空间）、百合 35 本（更早）
- 语料与 token 缓存为 GB 级，均被 .gitignore 排除，不入仓库；训练在云端 AutoDL 4090 进行
