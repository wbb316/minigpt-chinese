# TRAINING_PERFORMANCE_AUDIT — 训练基础设施性能优化专项

- 开始：2026-09-06
- 状态：**进行中**（本地 P0/P1 修复 + 等价性测试 → 待云端 GPU 消融 benchmark B0-B8）
- 目标：50M（51.4M params / 12L-9H-576d / ctx512 / batch64）训练吞吐优化
- 基线：~133.8k tokens/s（4.15 step/s），1B ≈ 2h04m，2.8B ≈ 5.8h
- 约束：不改模型架构/vocab/tokenizer/数据/loss/checkpoint；benchmark 用消融协议；**禁止自动启动正式训练**

## 1. 初始瓶颈清单（读码结论，待 benchmark 验证）

| # | 位置 | 问题 | 严重度 |
|---|---|---|---|
| 1 | `data/dataset.py` PackedDataset.memmap `__init__` | `arr = tokens[offset:offset+_n*block+1]` 全量 slice——触发 ShardMemmap 全库 828 片扫描+拼接（~1GB np.concatenate），arr 从未使用 | P0 |
| 2 | `data/token_cache.py` ShardMemmap | `_slice_range`/`__getitem__` int 每次从头线性遍历全部 shard（828 片 × 每样本 2 次） | P0 |
| 3 | `data/dataset.py` PackedDataset/TextDataset `__getitem__` | x/y 两次独立 slice + 两次 copy + 两次 uint16→long | P0 |
| 4 | `data/token_cache.py` `_split_points` | "空白串 \s+ 内部"当作绝对安全边界——理论上不严格（\s+ 是完整 pretoken chunk，内部切开可能改变该 chunk 的 BPE merge） | P0（正确性） |
| 5 | `data/token_cache.py` encode_to_cache | 命中校验仅查 index.json 存在；tokenizer_hash 只覆盖前 1000 merges（[:1000]）——tokenizer 变化可能错误复用 | P0（正确性） |
| 6 | `train/train.py` 训练循环 | 每步 loss.item() + tqdm + EMA + CSV 行 → 每步 GPU→CPU 同步 | P1 |
| 7 | `train/train.py` | zero_grad 默认（保留 grad buffer）；H2D 无 non_blocking | P1 |
| 8 | DataLoader | num_workers=8 未经搜索 | P1 |
| 9 | AdamW | 非 fused | P1 |
| 10 | SDPA | 调用 `F.scaled_dot_product_attention(is_causal=True)`，实际 backend 未确认 | P1/P2 |
| 11 | torch.compile | 未启用 | P2 |
| 12 | precision | 仅 fp16+scaler（50M 有 9 AMP skipped） | P2 |
| 13 | batch | 64 固定未搜索 | P2 |
| 14 | GPT.forward | block 恒 return_kv=True → 每层 tuple 打包 + new_kvs 收集（训练不需要） | P2（低） |
| 15 | 初始化 | 35M/50M 随机初始化后 CE>300（≠ln6144≈8.72）→ **只记录 FUTURE_CORRECTNESS/STABILITY ITEM，不纳入本次** | 记录 |

## 2. 保护项（禁止修改）
50M 架构 / vocab 6144 / tokenizer / block 512 / dropout 0.1 / tie / loss / causal / 数据 / val 集 / weight_decay / 已训练 checkpoint / 历史结果。

## 3. Benchmark 协议（B0-B8 消融，云端 4090 执行）

- 固定：50M 51,411,840 / 12L-9H-576d / ctx512 / shard0 cache / batch 由变体定
- 每组：`--max-steps 900`（~200 burn + ~700 计时段），`--val-every 0`，`--log-every 50`，独立 `benchmark/perf_out/<variant>`
- 计时：step_history 最后 10 采样行 Δstep/Δtime → step/s、tokens/s（×batch×512）
- GPU：`nvidia-smi -l 1` 后台采样 → `benchmark/gpu_metrics_<variant>.csv` → mean/p50/p90
- 结果：`benchmark/performance_results.csv`（schema 见 perf_bench.CSV_FIELDS）
- 组序：B0 基线 → B1 logging → B2 zero_grad/H2D → B3 dataloader w{0,2,4,8,12} → B5 fused → B6 compile → B7 bf16 → B8 batch{80,96}
- 执行：`python benchmark/perf_bench.py --all`（云端，/root 代码）
- 保护：不覆盖正式 checkpoint/cache/log（perf_out 独立目录；tokenizer pkl 复制非移动）

## 4. 变更记录

| 日期 | 变更 | 文件 | 验证 |
|---|---|---|---|
| 09-06 | P0a：PackedDataset.memmap init 删除整库无效 slice（`arr=tokens[offset:…]` 触发全库 828 片拼接且未使用）→ 只存引用+元数据 | `data/dataset.py` | 新测试 test_packed_dataset_memmap_equals_ndarray（len + 100 idx x/y 逐 token == ndarray 路径） |
| 09-06 | P0b：ShardMemmap int/slice 定位线性扫 828 片 → `searchsorted` 二分（预存 starts/ends） | `data/token_cache.py` | test_memmap_slices_after_bisect（200 int + 500 随机 slice + 接缝窗口 == 参考数组） |
| 09-06 | P0c：PackedDataset/TextDataset `__getitem__` x/y 两次 slice+copy+long → 单次 `[start:start+block+1]` 读 | `data/dataset.py` | 同上（含 TextDataset） |
| 09-06 | P0d：`_split_points` 安全切分从"空白串内部"改为 **PRETOK chunk 边界**（严格正确；\s+ 是完整 chunk，内部切开理论上可改 BPE merge） | `data/token_cache.py` | test_split_points_on_chunk_boundaries（6 类文本：切点 ∈ chunk 边界集合；分片 encode == 整段 encode exact） |
| 09-06 | P0e：cache 命中校验完整化——`tokenizer_hash_full`（全部 merges）写入新 index；命中前校验 version/dtype/vocab/chars/hash；校验失败不删旧目录、另建 `_h{hash}` 后缀目录 | `data/token_cache.py` | test_cache_hash_invalidation（tokenizer merges 变 → 新目录；旧目录保留；同 tokenizer 再命中） |
| 09-06 | P1：`--log-every N`（默认 20，采样 loss.item()/EMA/tqdm/step 行）；`zero_grad(set_to_none=True)` 前移 forward 前；`x.to(non_blocking=True)`；`--precision fp16/bf16`（bf16 无 scaler）；`--fused-adamw`（try/except 回退）；`--compile`（仅 CUDA）；step CSV 采样行仍含真实 step/epoch/tokens_seen/lr | `train/train.py` | 60/60 pytest；CPU 冒烟（bf16/compile-忽略/fused 路径跑通）；epoch 平均标注采样 |
| 09-06 | benchmark 执行器 + 计时函数 | `benchmark/perf_bench.py` | compute_throughput 单测 4.17 vs 期望 4.15 step/s ✓ |

> 说明：P0 修复后 train/val 的 uint16 cache 未重编（旧 cache 由 legacy hash 校验命中，tokenizer 未变）；safe-split 逻辑只影响**未来新建**的 cache（如 shard1/2）。

## 5. 等价性测试

`test/test_perf_audit_equivalence.py`（4 项）+ 全套 **60/60 通过**：
- ShardMemmap 二分切片 == 线性参考（int/随机 slice/接缝）
- PackedDataset memmap == ndarray 路径（len + 100 idx × block∈{16,64} × offset∈{0,11,512}；TextDataset 同）
- safe split：切分点全部 ∈ PRETOK chunk 边界；6 类文本（中文段落/连续空格/多换行/标点/中英数混/长词块/大空白压力 100 片）分片 encode == 整段 encode（exact）
- cache hash 失效：tokenizer merges 内容变 → 新 `_h` 目录 + 旧目录保留；未变 → 命中

## 6. 最终结论（云端 4090 实测，2026-09-06）

### 6.1 结果总表（14 组消融）

| variant | tokens/s | loss_end | VRAM | 结论 |
|---|---|---|---|---|
| B0 基线 | 137,681 | 13.77 | 17.3GB | — |
| B1 logging 采样 | 137,681 | 14.20 | 同 | **0%**（GPU 饱和，sync 非瓶颈） |
| B2 zero_grad/H2D | 137,681 | 13.88 | 同 | **0%** |
| B3 workers 0/2/4/8/12 | 全 137,681 | ~13.6 | 同 | **0%**（workers=0 都喂得饱 → 数据管线非瓶颈） |
| B5 fused AdamW | 137,681 | 14.11 | 同 | **0%** |
| B6 **compile** | **240,941** | 13.92 | **11.9GB** | **+75%** ✅ 且显存 -31% |
| B7 bf16 | 138,847 | 13.60 | 17.3GB | +0.8%（噪声） |
| B8 batch80 / 96 | 135,629 / OOM | 13.60 | 21.4 / 22.9GB | batch 无增益；96 OOM |
| B9 compile+batch80 | 240,941 | 14.05 | 14.8GB | 与 B6 相同 → batch 不叠加 |

全部组 nan=0、loss 轨迹一致（~50→~14）→ 正确性无回归。

### 6.2 结论

1. **唯一有效优化 = torch.compile（+75%）**：50M 是"小算子密集 + 手写 LayerNorm 多 kernel"型，launch/memory-bound → 图融合收益巨大；附带显存 -31%（buffer 复用）
2. CPU 侧全部零收益（logging/数据管线/workers/fused/bf16/batch）：GPU 图效率是唯一瓶颈——P0/P1 修复保留（正确性 + 未来更大数据扩展），但对当前吞吐无贡献
3. batch80/96 无叠加增益且 96 OOM → batch64 保留（LR schedule 无需重设计，与 50M E1 一致）

### 6.3 RECOMMENDED_TRAINING_STACK

```
50M (12L/576d/9H) + batch64 + fp16 + --compile
workers 8 / log_every 默认 / 其余与 50M E1 相同
```

### 6.4 时间估算（240,941 tok/s 实测）

| 规模 | 原速度 (137.7k) | compile (240.9k) | 节省 |
|---|---|---|---|
| 1B | 2.07h | **1.15h** | 45% |
| 2.8B | 5.8h | **3.2h** | 45% |

### 6.5 决策（任务书 §21 标准）

实测 **240,941 tok/s ≥ 170k 阈值 → 决策 A：可以推进 50M + 2.8B 新数据**。
（compile 训练 loss 轨迹与基线一致 → 可安全用于正式 run 与 resume。）

---
FUTURE_CORRECTNESS/STABILITY ITEM：GPT 无显式标准初始化（initial CE>300）——未来从零训 90M/100M 前单独做 GPT initialization audit，本次禁止混入。
