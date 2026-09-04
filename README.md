# MiniGPT-Chinese

从零用 PyTorch 实现的迷你中文 GPT 文本生成模型 —— 一个从"语料 → 自研分词器 → 模型 → 训练 → 生成"完整链路的中文语言模型项目。

> 核心思想：**不依赖任何现成 NLP 库**（不用 HuggingFace / tiktoken / 预训练模型），
> BPE 分词器、Transformer、训练循环全部手写，在云端 4090 上从零训出自己的中文小说续写模型。

## 🎯 项目亮点

- **自研 BPE 分词器**：字节级 BPE，正则预分词 + 链表增量合并，4M 字符训练分钟级、487M 字符并行编码几分钟
- **从零实现 GPT**：LayerNorm / 正弦位置编码 / FeedForward / 多头因果注意力（SDPA + 手写双路径）/ 主模型拼装
- **完整训练链路**：语料清洗 → BPE → 数据切片 → 训练（warmup+cosine、AMP、梯度裁剪、step 级验证、早停）→ Web 生成
- **多代演进记录**：7.2M 百合单语料 → 6.3M LayerNorm 修复版 → **20M 双语料（v0+v1）最终版**

## 📊 最终模型（result_20m_all / result_ln）

| 版本 | 参数量 | 结构 | 语料 | 词表 | 验证 loss |
|---|---|---|---|---|---|
| 百合基线 | 7.2M | 7层/8头/256维 | 35 本百合轻小说 (2900万字) | 3256 | 3.79 |
| LN 修复版 | 6.3M | 6层/8头/256维 | 轻小说 v0 | 6144 | 4.188 |
| **20M 最终版** | **20M** | **10层/8头/384维** | **轻小说 v0+v1 (4.16亿 token)** | **6144 + 权重共享** | **3.636** |

> loss 单位 nats，词表大小不同**不可直接横向比**（ln(6144)≈8.72 只是随机基线）。
> 20M 版 train_eval 3.616 / gap 0.020 —— 未过拟合，还有继续堆数据的空间。

**训练配置（20M 版）**：`--n-layer 10 --n-embd 384 --vocab-size 6144 --tie-embeddings --block-size 256 --batch-size 256 --lr 8e-4 --warmup-steps 500 --weight-decay 0.05 --sample-mode pack`

## 🧠 关键优化点（踩坑记录）

### 1. 验证集划分方式（最先修复的 bug）
按 token 顺序切分验证集会**高估 loss**（验证集可能包含某本训练没见过的整本书）。
改成**按文件划分**（每本前 90% 训练、后 10% 验证），val loss 从 4.5 → 3.79。

### 2. LayerNorm 标准化
早期用了无偏方差 + 不带 eps 除法的错误实现，收敛慢、数值不稳。改为标准实现
`(x - mean) / sqrt(var_biased + eps) * gamma + beta`，同规模下验证 loss 明显改善。

### 3. BPE 分词器 v2（正则预分词 + 链表增量合并）
- 预分词 `\w+|[^\w\s]|\s+`：合并**永不跨标点/空白**，中文语义更干净
- 训练用**链表 + numpy 增量**统计相邻对，4M 字符从 ~50 分钟降到分钟级
- 并行 encode（16 进程）：487M 字符从 30-60 分钟 → 几分钟
- 采样 bug 修复：训练样本从语料**均匀撒 100 段 × 20K 字符**（之前误改成了只取开头 2M）

### 4. 数据模式：slide vs pack
- `slide`（stride=1）：重叠滑窗，样本多、训练稳，适合小语料
- `pack`（stride=block_size）：不重叠打包，100M+ 语料每 epoch 从小时级 → 分钟级
- 数据全程 numpy int64 零拷贝视图，内存 720MB → 160MB

### 5. 过拟合判断修正（gap 指标）
训练 loss 带 dropout 噪声 + epoch 平均偏高，不能直接与 val 比。
验证时**额外算 eval 模式（无 dropout）的 train loss**，打印 `gap = val - train_eval` —— 这才是正确指标。

### 6. tie_embeddings + 权重去重
输入/输出嵌入共享权重（省掉 head 的 ~833K 参数），`dedupe_params` 避免被优化两次。

### 7. KV cache 增量推理（生成加速）
- `attention.py` 缓存路径：每步只算新 token 的 q/k/v，与历史拼接
- 与原版整序列重算**逐位置 logits 完全一致**（max diff < 1e-4），~1.6-2.2x
- `model/generation.py` 加**超窗重建**：缓存超过 block_size 时用最近 block_size 个 token 重建，长生成不越界

### 8. SDPA（Flash Attention）训练加速
`attention.py` 训练路径用 `F.scaled_dot_product_attention(is_causal=True)`：
不物化 (B,H,T,T) 注意力矩阵，block 256+ 显存省 ~50%、速度更快（Flash / mem-efficient 自动选择）。
KV cache / 自定义 mask 路径保留手写实现（Flash 不支持）。

### 9. 生成采样三参数（观感质量飞跃）
`model/sampling.py` 统一实现，网页三个滑块可调：

| 参数 | 作用 | 推荐默认 |
|---|---|---|
| temperature | 温度，越低越保守 | 0.8 |
| top_p | Nucleus，只保留累积概率前 top_p 的 token | 0.9 |
| repetition_penalty | 抑制复读（logit>0 除以 penalty） | 1.15 |

同模型同 seed 对比：纯采样（t=1.0）会跳变复读（"傀儡师""轻易吐露的兴趣"），
三参数默认值下输出通顺、有对话体（实测 20M 版效果）。

### 10. 断点续训 + warm restart
`--resume <latest>` 恢复 optimizer/AMP/step，cosine 按全局 step 重新锚定（相当于 warm restart，LR 回升，可多跑几轮）。

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
│   ├── tokenizer.py    # 自研 BPE（v2：预分词 + 链表增量合并）
│   ├── dataset.py      # TextDataset（slide）/ PackedDataset（pack）
│   ├── prepare_corpus.py        # 百合语料清洗
│   └── prepare_lightnovel.py    # 网文语料切片（v0/v1/all）
├── train/
│   ├── train.py            # 主训练脚本
│   └── train_small_*.py    # 数据量/模型量对比实验
├── app/
│   ├── server.py           # FastAPI 后端（自动推断架构）
│   └── templates/index.html# 前端（温度/top_p/重复惩罚滑块）
├── test/                   # pytest（26 个用例）
├── generate.py             # 生成 + KV cache 一致性验证 + 计时
├── visualize_attention.py  # 注意力热力图（hook 版，不改模型）
├── plot_loss.py            # 训练曲线
├── scratch/                # 早期 PyTorch 学习脚本（CIFAR/MNIST 教程）
└── docs/                   # 项目管理文档（见下表）
```

### 📁 项目管理文档（docs/）

| 文件 | 用途 |
|---|---|
| `MiniGPT_Project_Status.md` | 项目状态文件——**改代码前先读**：当前阶段/约定/坑点/日志格式 |
| `EXPERIMENT_LOG.md` | 训练实验日志（每次训练完追加一行，含 commit/配置/val loss） |
| `experiment_config_v2.yaml` | v2 实验固定配置（webnovel ~1B token） |
| `report_output/` | 实验复盘 HTML 报告 |

## 🚀 快速开始

```bash
pip install -r requirements.txt

# 测试（26 个用例）
python -m pytest test/ -v

# 生成测试（KV cache 一致性验证 + 计时 + 示例）
python generate.py --ckpt result_20m_all/checkpoint_best.pt \
                   --tokenizer result_20m_all/tokenizer_best.pkl

# Web demo
python app/server.py --ckpt result_20m_all/checkpoint_best.pt \
                     --tokenizer result_20m_all/tokenizer_best.pkl
# 浏览器打开 http://127.0.0.1:8000
```

## 🧪 测试

26 个测试全部通过：LayerNorm、注意力（含 SDPA/缓存）、GPT 结构、过拟合冒烟、BPE（v1+v2）、数据集、采样器（11 个）。

## 📅 Roadmap

- [x] 阶段1：PyTorch 复习（CIFAR/MNIST）
- [x] 阶段2：从零写 GPT 模型
- [x] 阶段3：数据管线 + 自研 BPE tokenizer
- [x] 阶段4：训练 + 可视化（loss 曲线）
- [x] 阶段5：实验体系（对比实验 + 分析）
- [x] 阶段6：后端 API + 前端页面
- [x] 阶段7：优化迭代（LN 修复、pack 模式、SDPA、KV cache、采样参数、20M 训练）
- [ ] 阶段8：更大模型 + 更长上下文（30M+ / block 512）
- [ ] 阶段9：部署上线

## 📜 数据说明

- `D:\小说` 原始网文语料（v0 ~493MB / v1 ~1GB），经 `data/prepare_lightnovel.py` 清洗切分为 v0/v1/all
- 百合语料 35 本（旧版），目录 `data/train.txt / val.txt`
- 语料文件与 npy 缓存为 GB 级，均被 .gitignore 排除，不入仓库
