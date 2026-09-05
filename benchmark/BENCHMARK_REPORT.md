# BENCHMARK_REPORT — BPE Tokenizer Trainer 性能专项重构

> 日期：2026-09-05 ｜ commit：`1dc7e8d`
> 范围：仅 tokenizer 训练工程（train.py / GPT / attention / dataset / sampling / generation 未动）
> 结论：**fast trainer 与 legacy 逐 merges/vocab 完全等价**，vocab 越大加速越明显
>       （200K 实测 1024:0.9x / 3256:2.5x / 6144:3.7x；4M/6144 预计 ~5-8min vs legacy ~55min）

## 1. 原实现瓶颈（profile 实证，非推测）

真实语料 100K 字符 / vocab 6144 / legacy，cProfile：

| 热点 | 耗时 | 占比 | 原因 |
|---|---|---|---|
| `max(counts, key=lambda…)` 的 lambda | 39.3s | 43% | 每轮 merge 全扫 counts dict（2.34 亿次 lambda 调用 / 6083 轮） |
| `builtins.max` 本身 | 39.2s | 43% | 同上，全 dict 线性扫描 |
| 其余（_inc/_dec 局部更新、flatnonzero 等） | ~12s | 13% | 链表增量本身不贵 |

**结论**：任务书预判的 `np.flatnonzero` 全扫节点不是瓶颈（向量化，6083 次仅 0.28s，
<1%）；真正瓶颈是 **Python 层每轮 max() 全扫 counts**（~86% 耗时）。

## 2. 新实现（train_fast）

- **lazy max-heap**：`(-count, a, b)` 最小堆 ≡ `max(key=(count, -a, -b))` 的语义
  （count 大优先 → a 小优先 → b 小优先），每轮 pop O(log P) 而非全扫 O(P)
- **count 快照校验**：inc/dec 后都 push 最新快照；pop 时若 `counts.get(pair) != 快照`
  视为过期丢弃（lazy invalidation）
- **保留 numpy flatnonzero occurrence 扫描**（实测 <1% 耗时，非瓶颈，不加复杂索引）

### 关键 bug（等价性测试抓出）
初版只在 count **增加**时 push、减少时靠 lazy 丢弃 → 一个仍存活但 count 刚降的 pair
会失去 heap 代表而被跳过（实测 divergence @ merge#28）。
**修复**：dec 后若 pair 仍存活也 push 最新快照（保证每个存活 pair 恒有代表）。
修复后 15/15 等价测试 + 全套 41/41 通过。

## 3. 时间复杂度变化

| | legacy | fast |
|---|---|---|
| 选 pair | O(P) 每轮（P = 存活 pair 数） | O(log P) 每轮 + 摊销的过期丢弃 |
| occurrence | numpy flatnonzero O(N) | 同左（未改） |
| 总体 | O(M × (P + N))，M=merge 数 | O(M × log P + M × N_vec)，N_vec 为向量化扫描 |

## 4. 内存变化

| | legacy | fast |
|---|---|---|
| counts dict | 同 | 同 |
| 额外 | 无 | heap（≤ 存活 pair 数的 3-int tuple） |
| 实测 200K/6144 | ~18 MB | ~78 MB（峰值含 heap 过期条目） |

fast 峰值内存比 legacy 高 ~4x，但绝对量很小（200K 78MB → 4M 估计 <1.5GB，
满足任务书 <4GB 要求）。若需进一步压低，可周期性 compact heap（丢弃过期条目）。

## 5. Benchmark 结果（benchmark_results.csv）

### Stage 1: 200K 字符（真实 webnovel 语料）
| vocab | legacy (s) | fast (s) | speedup | 等价 |
|---|---|---|---|---|
| 1024 | 6.6 | 7.6 | 0.9x | PASS |
| 3256 | 42.6 | 17.3 | **2.5x** | PASS |
| 6144 | 105.2 | 28.7 | **3.7x** | PASS |

### Stage 2: 1M 字符（真实 webnovel 语料）
| vocab | legacy (s) | fast (s) | speedup | 等价 | fast RSS |
|---|---|---|---|---|---|
| 1024 | 30.0 | 43.1 | 0.7x | PASS | 417 MB |
| 3256 | 179.7 | 109.9 | 1.6x | PASS | 408 MB |
| 6144 | 505.0 | 170.3 | **3.0x** | PASS | 405 MB |

### Stage 3: 4M 字符（**待跑**，用户推迟到后续）
只跑 fast（不跑 legacy）：目标实测 4M/6144 fast 生产耗时（预计 ~8-15 min）。

> vocab 1024 时 fast 略慢：merge 轮数少（766），max 全扫的 counts 小，
> heap 的 Python 层 push/pop 固定开销占主导。真实使用（3256/6144）均显著加速。

## 6. 正确性

- `test/test_tokenizer_train_equivalence.py`：15 个测试（10 类语料 + 随机 + 全 merge 1024/3256），
  `legacy.merges == fast.merges` 且 `legacy.vocab == fast.vocab`，encode 一致，decode roundtrip OK
- 全套 pytest：41/41 通过
- benchmark 每档都做全量 merges 等价对比（非抽样）

## 7. 结论

- ✅ **推荐替换默认 trainer**（train.py 增加 `--bpe-trainer fast` 选项，默认保持 legacy 至全面验证）
- 实测加速（等价 PASS）：200K/6144 = 3.7x；1M/6144 = 3.0x（legacy 505s → fast 170s）
- 4M/6144 预计：legacy ~55min → fast ~8-15min（Stage 3 待跑实测，用户推迟）
- 语义零变化：merges/vocab/encode/decode 与 legacy 完全一致
