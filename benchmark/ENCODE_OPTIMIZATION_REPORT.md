# ENCODE_OPTIMIZATION_REPORT — Encode Pipeline 专项优化

> 日期：2026-09-05 ｜ 范围：仅 encode pipeline（tokenizer train/merges/vocab、GPT、训练配置均未改）
> 前置：BPE trainer 已优化（见 `BENCHMARK_REPORT.md`）；本报告只讲 encode。

## 1. 背景与目标

v2 语料 webnovel_v2（~10.5 亿 token）首次编码需 raw text → token cache。
旧管线：单文件 int64 npy（700M×8B≈5.6GB），多进程 worker 把大数组经 IPC 传回主进程
再 np.concatenate——内存/传输浪费。任务目标：端到端 ≥2x、期望 3-5x，正确性优先。

## 2. 改动清单

| 文件 | 改动 |
|---|---|
| `data/tokenizer.py` | `encode()` 热循环优化（局部变量绑定、免中间 ids list、栈顶合并精简）→ **1.5x**；train/merges/vocab/PRETOK_RE 均未动 |
| `data/token_cache.py`（新） | 并行编码：按**预分词块间隙**切分（每片独立编码 == 整段，逐 token 一致）；worker 直写 uint16 `.bin` 分片，只回传元数据；`ShardMemmap` 按片 memmap 只读 |
| `data/dataset.py` | `TextDataset`/`PackedDataset` 支持 ShardMemmap：按 batch 切片 uint16 → torch.long，不整载 |
| `train/train.py` | `--cache-format shards(默认)/legacy`；`get_tokens` 用新管线（legacy npy 路径保留） |

## 3. 关键设计

### 3.1 encode() 热循环（1.5x）
- 免中间 `ids` list，直接遍历 `raw` bytes
- `merges.get / out.extend / vocab` 绑定局部变量，消除属性查找
- 栈顶合并 `stack[-2]=nid; del stack[-1]`（替代 pop×2+append）
- 输出与旧实现逐 token 一致（10 类语料 + 2MB 真实文本测试）

### 3.2 并行分片（逐 token 一致的关键）
旧 `encode_text` 按字符均分切片 → 会切断 chunk，边界 token 有 <0.01% 差异。
新实现用 `_split_points()`：PRETOK_RE = `\w+|[^\w\s]|\s+` 对文本无缝覆盖、
chunk 之间无间隙，但 **`\s+` 空白 chunk 内部是安全切分区**（空白串独立编码、
merge 永不跨 chunk）——切分点取在空白区段中部，每片独立 encode 与整段
**完全一致**（500K 真实文本 8 片 == 整段验证；100M/86M token 全量一致）。

### 3.3 worker 直写 + IPC 修复
- worker 只收 (txt分片路径, bin路径)，**文本经文件传递不走 IPC**
- 实测发现：Windows spawn 下把大文本字符串塞进 job 参数会 pickle 传输极慢（100M 卡死）
- 主进程先把分片写成临时 .txt，worker 读文件编码后写 uint16 bin，回传 (path, counts)

### 3.4 uint16 + memmap
- vocab 6144 ≤ 65535 → token 落盘 uint16：700M ≈ 1.4GB（vs int64 5.6GB，省 4x）
- `ShardMemmap` 按片 memmap（mode='r'），训练按 batch 切片，不整载 1.4GB
- index.json 记录 dtype/vocab/tokenizer_hash/token_count/分片表

## 4. Benchmark（benchmark_encode.py → benchmark_encode_results.csv）

| chars | 单进程 (s) | 多进程管线 (s) | 加速 | 单进程 RSS | 管线 RSS | uint16 输出 |
|---|---|---|---|---|---|---|
| 1M | 0.5 | 0.6 | 0.9x | 8MB | 2MB | 2MB |
| 10M | 5.9 | 3.5 | **1.7x** | 71MB | 4MB | 17MB |
| 100M | 待补 | 待补 | ? | - | - | - |

> 单进程 encode 本身很快（~1.7M chars/s，全量 12.4 亿字符 ≈ 11 分钟），
> 并行收益受 Python GIL/进程启动影响有限；主要收益在**内存与磁盘**（RSS 71→4MB，
> uint16 输出省 4x）与缓存复用（二次运行秒级加载）。

## 5. 正确性（tests/test_encode_equivalence.py，15/15）

- 优化后 encode == legacy reference（10 类语料：中/英/数/标点/空白/混合/重复/段落/长文）
- 多进程分片拼接 == 单进程整段（逐 token）
- memmap 随机切片（含跨分片边界）正确
- PackedDataset/TextDataset(memmap) batch 与整段一致
- 2MB 真实 webnovel 文本逐 token 一致
- vocab > 65535 报错（uint16 溢出保护）
- 全套 pytest：56/56 通过

## 6. 端到端估算（v2 真实编码 12.4 亿字符 train）

- 单进程 encode：~11 分钟（1.85M chars/s 实测）
- 8 worker 管线：预计 ~3-6 分钟（10M 测 1.7x 外推保守）
- 缓存：uint16 ~1.4GB 分片 + index，二次训练秒级加载（无需重编码）
- 训练内存：memmap 按 batch 读，不再整载 5.6GB int64

## 7. 结论

- ✅ encode() 1.5x、管线内存省 ~18x、uint16 磁盘省 4x、memmap 免整载
- ✅ 逐 token 一致（15 项正确性测试全过）
- 端到端加速 1.7x（10M）+ 缓存复用；云端 Linux fork 下多进程开销更低，预期更好
- 建议 v2 正式编码在云端跑 `--cache-format shards --bpe-trainer fast`
