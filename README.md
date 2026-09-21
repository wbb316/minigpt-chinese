# MiniGPT-Chinese

从零用 PyTorch 实现的迷你中文 GPT 文本生成模型 —— 一个从"语料 → 自研分词器 → 模型 → 训练 → 生成"完整链路的中文小说模型项目。

> 核心思想：**不依赖任何现成 NLP 库**（不用 HuggingFace / tiktoken / 预训练模型），
> BPE 分词器、Transformer、训练循环全部手写，在云端 4090 上从零训出自己的中文小说续写模型。

## 🎯 项目亮点

- **自研 BPE 分词器**：字节级 BPE，正则预分词 + 链表增量合并，4M 字符训练分钟级；大语料并行编码（uint16 分片缓存 + memmap，3.6GB 文本 → 998M token 几分钟）
- **从零实现 GPT**：LayerNorm / 正弦位置编码 / RoPE / SwiGLU / 多头因果注意力（SDPA + 手写双路径）/ 主模型拼装
- **完整训练链路**：语料清洗 → BPE → 数据切片（pack 打包）→ 训练（warmup+cosine、AMP、梯度裁剪、**梯度累积**、step 级验证、早停、断点续训）→ Web 生成
- **多代演进记录**：7.2M 百合单语料 → 6.3M LayerNorm 修复版 → 20M 双语料 → v2 35M（webnovel 1B×2）→ v2 50M → v3_alpha 50M（rope+新 init）→ **v3_beta 100M → v3_gamma 150M（2.93B token，本仓库目前最大规模）**

## 📊 模型演进与最佳结果

> loss 单位 **nats/token**（随机基线 = `ln(vocab)`，vocab 8192 时 = 9.0109）。
> **三个评估空间互不可比，禁止横向排名**：
> ① **旧空间** = 百合/轻小说语料 + 旧 tokenizer（vocab 3256/6144，ctx256）
> ② **v2 空间** = webnovel_v2 shard0 + vocab 6144（ctx512）
> ③ **shard1+2 空间** = webnovel shard1+2 + **vocab 8192**（ctx512，自己的 val split）
>
> 同一空间内可直接比；跨空间只能看量级趋势。

| 版本 | 参数量 | 结构 | 语料 | val loss | 评估空间 |
|---|---|---|---|---|---|
| 百合基线 | 7.2M | 7L/8H/256d | 百合 35 本 | 3.79 | ① |
| LN 修复版 | 6.3M | 6L/8H/256d | 轻小说 v0 | 4.188 | ① |
| 20M 最终版 | 20M | 10L/8H/384d | 轻小说 v0+v1 (4.16亿 token) | 3.636 | ① |
| v2 35M E1 | 34.7M | 10L/8H/512d | webnovel_v2 shard0 (1B) | 3.7076 | ② |
| v2 35M E2 | 34.7M | 同 E1（恒温 5e-5 续训） | shard0 二遍（累计 2B） | 3.6372 | ② |
| v2 50M | 51.4M | 12L/9H/576d | webnovel_v2 shard0 (1B) | 3.6154 | ② |
| **v3_alpha 50M** | **51.4M** | **12L/9H/576d（rope+GPT-2 init）** | webnovel_v2 shard0 (1B) | **3.2097** | ② |
| **v3_beta 100M** | **101.5M** | **16L/11H/704d + SwiGLU + vocab 8192** | **shard1+2（1.96B 全新 token）** | **3.0610** | **③** |
| **v3_gamma 150M** | **148.1M** | **20L/12H/768d + SwiGLU h=2048 + vocab 8192** | **shard0+1+2（2.93B token，单轮）** | **2.9675** | **③** |

📁 **③ 空间两个 run 的完整曲线 CSV 与指标对照见 [`docs/results/`](docs/results/README.md)**
（val / step 历史曲线，逐字节取自云端产物，可直接在 GitHub 上查看）。

### 🏆 核心 scaling 结论

**① 参数对照（2026-09-06，② 空间，同 tokenizer/语料/val/batch/LR，仅参数量不同）**：

```
v2 50M @ 1B (3.6154)  <  35M @ 2B (3.6372)  <  35M @ 1B (3.7076)
```

**参数 scaling 收益 > 重复数据收益**：50M 单轮 1B 打赢 35M 把同批数据训两遍的 2B。

**② 底层升级（2026-09-08，v3_alpha）**：同结构同数据把位置编码换 RoPE + GPT-2 式 init（双变量），
val **3.6154 → 3.2097**（−0.406 nats，~11%）；rope 单变量短训独立支撑该方向。

**③ 数据 + 参数继续放大（2026-09-11 → 09-14，③ 空间）**：

```
v3_gamma 150M @ 2.93B (2.9675)  <  v3_beta 100M @ 1.96B (3.0610)
Δ = −0.0935 nats（−3.05%）
```

⚠️ **这个差值不能单独归因于 scaling**，它至少混了四件事：参数量 1.46×、数据量 1.49×，
以及两个**规定共变量**（lr 7e-4→6e-4、warmup 2000→2500）。拆开各维度贡献需要
「同 token 预算 + 同 lr 调度」的对照实验（**尚未做**）。

⚠️ **两个 run 不能按同 step 比**：150M 的 cosine 铺在 89,463 步、100M 铺在 59,886 步，
同一 step 的 lr 并不相等 → 实测中段（40k–50k）150M 反而差 **+0.0007 ~ +0.0038 nats**，到 55k 才反超。
**自洽的比法是终点 vs 终点**（双方 cosine 都走到 min_lr）。详见 `docs/EXPERIMENT_LOG.md`。

（各代 checkpoint 归档于 `result/` 与 `log/`，见「项目结构」；权重文件不入 git）

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
- ⚠️ `ShardMemmap` 会**同时打开所有分片**：1660 片 > 容器默认 soft limit 1024 → 需 `ulimit -n 65535`，否则 `OSError: [Errno 24] Too many open files`

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

### 12. GPT-2 式权重初始化（v3_alpha 前，消灭起点 loss 爆炸）
早期 `nn.Embedding` 默认 N(0,1) + tie → head 权重 std=1，初始 CE 300+（正常应 ≈ln(vocab)=8.72）。
修复：所有权重 N(0,0.02)，残差层 0.02/√(2·n_layer)；初始 CE 96.5→6.97，同配置最终 loss 好 ~0.2 nats。

### 13. RoPE 旋转位置编码（默认，v3_alpha 起）
sinusoidal（位置加到 embedding）与 rope（旋转 Q/K）二选一，`--position-encoding` 可切换，推理工具自动检测。
50M/4000 步单变量对比：rope 全面胜出（val 3.780 vs 4.788 @3000 步），正式训练 v3_alpha 50M → 3.2097（② 空间新 best）。

### 14. torch.compile 兼容（checkpoint 前缀坑）
`--compile` 训练吞吐 +75%（137.7k→240.9k tok/s）。但 compile 包装使 `gpt.state_dict()` 带 `_orig_mod.` 前缀，
导致推理工具解析失败——修复：train.py 保存用 raw 模型（无前缀），generate/server/visualize 加载端兼容剥离（双保险）。
2026-09-13 在 100M 上重测（独立 A/B，可复现）：**+88.1% 吞吐、−8.7 GB 显存**，见 `docs/TRAINING_PERFORMANCE_AUDIT.md` §7。

### 15. SwiGLU FFN（默认，v3_beta 起）
`--ff-type relu|gelu|swiglu`，`ff_hidden` 默认由 `swiglu_hidden(n_embd)` 自动求解（如 768 → 2048）。
从零短程对照（各 5000 步，同 seed / 唯一变量为激活）：**swiglu 3.655 < gelu 3.688 < relu 3.695**。

### 16. 梯度累积（`--grad-accum`，2026-09-13）
150M 模型在 **micro-batch 64 下第一步 backward 即 OOM**（23.16 GiB），需要 micro 32 × accum 2 凑出 effective batch 64。
门控语义严格：`global_step` / LR scheduler / `tokens_seen` / 验证节奏**只在成功的 optimizer update 后推进**
（AMP overflow 跳过的周期同样不推进）；`loss/accum` 之后再 `scaler.scale(...).backward()`；`total_steps` 按 **effective batch** 折算；
epoch 末尾 flush 残余累积；**默认 1 = 与旧行为逐位一致**。
实测：500 次 update 后 `tokens_seen = 16,384,000 = 500 × 32,768` 无零头，resume 后计数连续。
⚠️ `--grad-accum>1` 时进度条数的是 **micro-batch**（2×），落盘计数仍是 update 口径。

## 🏗️ 项目结构

```
minigpt-chinese/
├── model/
│   ├── layers.py       # LayerNorm、正弦位置编码、FeedForward
│   ├── attention.py    # 多头因果注意力（SDPA 训练路径 + KV cache 手写路径）
│   ├── rope.py         # RoPE 旋转位置编码
│   ├── gpt.py          # GPT 主模型（tie_embeddings、KV cache、去重参数计数）
│   ├── ffn_adapter.py  # FFN 变体适配（relu / gelu / swiglu）
│   ├── sampling.py     # 采样工具：temperature / top_p / repetition_penalty
│   └── generation.py   # 共享自回归引擎（KV cache + 超窗重建）
├── data/
│   ├── tokenizer.py    # 自研 BPE（v2：预分词 + 链表增量合并 + fast 后端）
│   ├── token_cache.py  # uint16 分片缓存 + ShardMemmap（大语料编码管线）
│   ├── dataset.py      # TextDataset（slide）/ PackedDataset（pack）
│   ├── prepare_corpus.py / prepare_lightnovel.py / prepare_webnovel.py
├── train/
│   └── train.py        # 主训练脚本（resume / lr-scheme / AMP 门控 / grad-accum / 精确 tokens_seen）
├── app/
│   ├── server.py       # FastAPI 后端（默认加载 v3_gamma 150M；架构由 RoPE head_dim 自动推断；
│   │                   #   GET /model-info 暴露当前实际加载的模型规格与是否走了回退）
│   └── templates/index.html  # 前端（非模板引擎，open() 原样返回；解码参数滑块 / 模型信息条 / 暗色模式）
├── test/               # pytest：124 个用例（BPE 与 encode 等价性 / 模型与注意力 / RoPE / FFN 变体 /
│                       #   数据集与采样器 / 过拟合冒烟 / perf_bench 解析与变体矩阵）
├── benchmark/          # 性能基准（perf_bench.py + performance_results.csv + performance_summary.csv）
├── generate.py         # 生成（默认 v2 35M）+ KV cache 一致性验证
├── visualize_attention.py  # 注意力热力图
├── result/             # 模型产物归档（按 参数量+token数 分子目录；*.pt/*.pkl 不入 git）
│   ├── 35M参数+998Mtokens/      # v2 35M（E1+E2 完整）
│   ├── 50M参数+998Mtokens/      # v2 50M（strict 参数对照基准）
│   ├── 100M参数v3+2Btokens/     # v3_beta 100M
│   ├── 150M参数v3+1.5Btokens/   # v3_gamma 150M ← 当前默认推理模型（目录名为拉取时旧名，实为 2.93B token，详见 app/server.py 注释）
│   ├── 20M参数+416Mtokens/      # 20M 最终（① 空间）
│   └── ...
├── log/                # 训练日志归档（按实验分子目录：step/val 历史 CSV、run_config）
│   ├── 50M参数_v3_alpha+998Mtokens/        # v3_alpha 50M（② 空间当前最优）
│   ├── 100M参数_v3_2Btokens/               # v3_beta 100M
│   └── 150M参数_v3_gamma+2.93Btokens/      # v3_gamma 150M（③ 空间当前最优，扁平布局）
└── docs/
    ├── MiniGPT_Project_Status.md  # 项目状态文件——改代码前先读
    ├── EXPERIMENT_LOG.md          # 训练实验日志（含各阶段详细记录）
    ├── TRAINING_PERFORMANCE_AUDIT.md  # 训练性能审计（含 torch.compile A/B）
    ├── RESUME_AUDIT.md            # resume 语义审查 + 修复报告
    ├── results/                   # ★ ③ 空间两个 run 的曲线 CSV + 指标汇总（入库）
    ├── experiment_config_v2.yaml / experiment_config_100M.yaml / experiment_config_150M.yaml
    └── report_output/             # 实验复盘报告（HTML/md）
```

## 🚀 快速开始

```bash
pip install -r requirements.txt

# 测试
python -m pytest test/ -v

# 生成测试（默认加载 v2 35M；用 v3_gamma 150M / v3_beta 100M 就传 --ckpt --tokenizer）
python generate.py --demo

# Web demo（默认加载 v3_gamma 150M，20L/768d/12H；浏览器打开 http://127.0.0.1:8000）
python app/server.py

# 页面能力：模型信息条（显示当前实际加载的规格，权重缺失走回退时会显式提示）
#           解码参数四个滑块（温度 / top_p / 重复惩罚 / 生成长度）
#           **流式逐字输出**（SSE，首字 ~25ms 出现，不用等整段算完）
#           停止生成（客户端断开 → 服务端一步都不多算）· 接着写 · 历史记录（localStorage，20 条）
#           暗色模式（跟随系统）· Cmd/Ctrl+Enter 提交 · 页内 toast · 剪贴板降级
# 接口：GET /health · GET /model-info（当前模型规格）
#       POST /generate        一次性返回（契约与历史一致，响应字段 elapsed_ms / new_tokens）
#       POST /generate/stream 流式 SSE：`data: {"delta": "..."}` 逐 token 推文本，
#                             末尾 `event: done` + {"elapsed_ms","new_tokens"}，出错 `event: error`
#       两者请求体字段相同（prompt/max_tokens/temperature/top_p/repetition_penalty）

# 想用 100M v3_beta 或别的存档：显式传参覆盖默认即可
python app/server.py --ckpt result/100M参数v3+2Btokens/checkpoint_best.pt \
                     --tokenizer result/100M参数v3+2Btokens/tokenizer_best.pkl

# 指定其他模型（v3_alpha 50M：rope+新init，9 头）
python generate.py --ckpt log/50M参数_v3_alpha+998Mtokens/checkpoint_best.pt \
                   --tokenizer log/50M参数_v3_alpha+998Mtokens/tokenizer_best.pkl \
                   --n-head 9
```

> ⚠️ 权重文件（`*.pt` / `*.pkl`）不入 git：v3_beta 100M best = 388.8 MB、v3_gamma 150M best = 594.3 MB，
> 均超出 GitHub 单文件 100 MB 硬限。需要时从本地归档 `result/` 或云端 `/root/result_*/` 取。

## 🧪 测试

**124 个用例**：BPE（训练 fast/legacy 等价性、encode 等价性 15 项）、编码管线（分片一致性、memmap 切片）、
模型 / 注意力 / RoPE / FFN 变体、数据集、采样器、过拟合冒烟、perf_bench（计时精度 / telemetry 窗口 /
变体矩阵 / 100M 配置档）等。

> ⚠️ **平台差异**：`test/test_encode_equivalence.py::test_real_webnovel_equivalence` 用 `n_procs=4`
> 多进程编码，**在禁止多进程命名管道的受限沙箱里会 `PermissionError`**（此时为 123 passed / 1 failed，
> 测试逻辑本身无误）。普通 Windows / Linux 环境下 124 个全过。

## 📅 Roadmap

- [x] 阶段1：PyTorch 复习（CIFAR/MNIST）
- [x] 阶段2：从零写 GPT 模型
- [x] 阶段3：数据管线 + 自研 BPE tokenizer
- [x] 阶段4：训练 + 可视化（loss 曲线）
- [x] 阶段5：实验体系（对比实验 + 分析）
- [x] 阶段6：后端 API + 前端页面
- [x] 阶段7：优化迭代（LN 修复、pack 模式、SDPA、KV cache、采样参数、20M 训练）
- [x] 阶段8：v2 综合升级（webnovel 1B×2、35M→50M、block 512、resume 语义修复）—— 参数 scaling 结论成立
- [x] 阶段8.5：v3_alpha 底层升级（2026-09-08：GPT-2 init + RoPE）—— 同结构 val 3.6154 → 3.2097（−0.406 nats）
- [x] 阶段9：③ 空间（shard1+2 + vocab 8192）数据侧 scaling —— **v3_beta 100M/1.96B → 3.0610**、**v3_gamma 150M/2.93B → 2.9675**（−0.0935 nats，含共变量）
- [ ] 阶段10：拆分 scaling 归因（同 token 预算 + 同 lr 调度下的 100M vs 150M 对照；可选 2×2 交叉评估）
- [ ] 阶段11：部署上线

## 📜 数据说明

- webnovel_v2：`D:\小说\webnovel\webnovel_{0,1,2}.jsonl`（3 shard × ~3.9GB）
  - shard0 → ② 空间 `train/val_webnovel_v2.txt`（12.4 亿字符 / 1.44 亿字符，vocab 6144）
  - shard1+2 → ③ 空间训练集（vocab 8192）；shard0+1+2 合并 → v3_gamma 150M 的 2.93B token
- 轻小说语料 v0/v1（① 空间）、百合 35 本（更早）
- 语料与 token 缓存为 GB 级，均被 .gitignore 排除，不入仓库；训练在云端 AutoDL 4090 进行
  - ⚠️ 云端**关机不丢数据**，但**「释放实例」会连 `/root/autodl-tmp` 一起清空**
