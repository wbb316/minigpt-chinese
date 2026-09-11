# MiniGPT Project Status — 项目状态文件

> **规则：修改任何代码前先读本文件**，了解当前阶段、约定与坑点。
> 训练完成后由助手根据用户口述在 `docs/EXPERIMENT_LOG.md` 追加实验记录。
> 完整工作流见文末「AI 操作规则 — 标准工作流（最终版）」。
> 最后更新：2026-09-11

## 一句话定位

从零 PyTorch 实现的中文小说 GPT（自研 BPE 分词器 + 手写 Transformer + 云端 4090 训练 + FastAPI demo），当前在 **B 组（能力深度）**：扩数据到 ~1B token + 更大模型 + 更长上下文。

## 当前阶段

- [x] **★ v3_beta 100M + shard1/2 新数据完成**（2026-09-11，`v3_beta_100M_ctx512_2B`）：**101,457,280**（16L/704d/11H，head_dim 64）/ **1.96B 全新 token**（shard1+2）/ ctx512 / vocab **8192** / FFN **swiglu**，**best val 3.0610** @ step 59863，耗时 **4.42 小时**（3.76 step/s）；**⚠️ 新评估空间**（val 是 shard1+2 自己的 split），**不在 v2/v3 可比链内**，不可与 v3_alpha 3.2097 排名；详见 EXPERIMENT_LOG
- [x] **★ vocab 消融收官 → 定版 8192**（2026-09-10，`v3_vocab_ablation` + `v3_vocab_cmp_lm`）：tokenizer 侧测 4096/6144/8192（4M 样本，前缀性质截取派生，截取的 6144 与独立训练逐位一致）；LM 短测（各 5000 步、同一份 val）全量 **bits/char 4.1784 vs 4.1398 → 8192 好 0.92%**，**10/10 验证点方向一致** → **vocab 定版 8192**；同时确认 **样本量 4M 足够**（4M→32M 仅 +0.623% 压缩率）；详见 EXPERIMENT_LOG 与 `tokenizer_sample_size_plan.md`
- [x] **★ v3_alpha 50M（新底层：RoPE + GPT-2 init）完成**（2026-09-08，`v3_alpha_50M_ctx512_1B`）：51.4M（12L/576d/9H）/ 1B token / ctx512，**best val 3.2097** @ step 30455（webnovel_v2 评估空间**新 best**；同结构较 v2 50M 3.6154 −0.406 nats，底层双变量升级；详见 EXPERIMENT_LOG）
- [x] **v2 50M 完成**（2026-09-06，`v2_50M_ctx512_1B`）：51.4M（12L/576d/9H）/ 1B token / ctx512，**best val 3.6154** @ step 30454（已由 v3_alpha 超越，仍为 strict 参数对照基准，详见 EXPERIMENT_LOG）
- [x] **★ 首个严格可比参数对照成立**：50M@1B(**3.6154**) < 35M@1B(3.7076) < 35M@2B(3.6372) → **参数 scaling 收益 > 重复数据收益**（同 tokenizer/语料/val/batch/LR，仅参数不同）
- [x] **v2 35M 两轮训练完成**（2026-09-05→06，`v2_35M_ctx512_1B_E1+E2`）：35M / **2B token**（998M×2 同语料二遍）/ ctx512 / vocab6144，**v2 best = 35M v2 Epoch 2：val **3.6372** / train_eval **3.6063** / gap **0.0310** @ step 60915**（v2 evaluation space best，不与旧 20M raw loss 直接排名；详见 EXPERIMENT_LOG）
- [x] Epoch 2 resume 修复完成并验证：`docs/RESUME_AUDIT.md`（next_epoch 语义 / `--lr-scheme const` 恒温 5e-5 / AMP 更新门控 / 精确 tokens_seen），云端与本地冒烟均通过
- [x] 语料下载完成：`D:\小说\webnovel\webnovel_{0,1,2}.jsonl` 各 ~3.9GB（合计 11.7GB，~148 万行 / ~2790 本，均已验证无 JSON 错误）
- [x] 清洗完成：shard0 → `data/train|val_webnovel_v2.txt`（云端 `/root/autodl-tmp/data/`，train 12.4 亿字符 / val 1.44 亿字符）
- [x] v2 实验配置已跑（**已训练验证**）：`docs/experiment_config_v2.yaml`（**10L/512d ≈35M**/bs512/vocab6144/tie，webnovel ~1B token，pack，min_lr=5e-5）—— 首次配置 batch128，随后实际成功运行配置调整为 batch64（总 token 不变）
- [ ] 下一步（B 组主线）：**v3 系列 + shard1/2 新数据**（~2B 全新 token，数据侧真 scaling）——v3_alpha@1B 末段未饱和，新数据预期继续显著降；FFN 默认已定 **swiglu**（见下条），正式 run 直接用 swiglu
- [x] **★ v3 FFN 变体短程对照完成**（2026-09-08，`v3_ffn_variants_shortrun`）：relu/gelu/swiglu 各 **5000 步**（同 seed 42 / **从零随机初始化** / **8e-4 cosine（warmup 1000 → min 5e-5）**，**唯一变量 = `--ff-type`**），best val @5000 步：**swiglu 3.655 < gelu 3.688 < relu 3.695** → **SwiGLU 胜出**（比 relu 好 0.040 nats，领先从 step 1000 起建立、全程一致）→ **已把默认 FFN 从 relu 改成 swiglu**（relu/gelu 保留可回退）；三变体短程值比 v3_alpha 正式 run 同点 val@5000=3.6321 略高——**本实验从零训练、与 v3_alpha 正式 run 的起点与调度不同**，**三变体之间同条件可比**；详见 EXPERIMENT_LOG

> **分别记录、不要混排**：
> - Historical best on old evaluation：20M val = 3.636（旧 tokenizer/语料/val 集/ctx256）
> - Current webnovel_v2 eval space best：**v3_alpha 50M val = 3.2097**（RoPE+GPT-2 init 新底层，ctx512；v2 50M 3.6154 次之）
> - **★ 新空间（2026-09-11 起）：shard1+2 空间** —— val = shard1+2 自己的 split（**与上面两个空间的 val 集不同，也与 shard0 的 val 不同**）；首个基准点 = **v3_beta 100M val 3.0610**（vocab 8192）
> - 旧空间与 webnovel_v2 空间处于不同评估空间，**不构成同一排行榜**；v3_alpha 与 v2 系同空间**可比**。
> - **shard1+2 空间的数字不可与 webnovel_v2 空间排名**（validation split 变化）。三空间各自成链。后续若在 shard1+2 上继续跑，**固定用该 val 集**以形成新的可比链。

## Current Best Checkpoint

- **Best known model (old eval): 20M_final**（10L/8H/384d/bs256/vocab6144/tie）
- **Best val_loss (old eval): 3.636**（nats；旧评估空间：旧 tokenizer/语料/val 集/ctx256；本地归档: `result/20M参数+416Mtokens/checkpoint_best.pt`）
- **Do not overwrite unless a new experiment improves it.**
- v2 (35M, 2026-09-06, **Epoch 2**): val **3.6372** / train_eval **3.6063** / gap **0.0310**，**tokenizer/语料不同与 3.636 不可直接比**；checkpoint: `result/35M参数+998Mtokens/checkpoint_best.pt`（本地归档；云端 `/root/result_webnovel_v2/`）
- **v2 (50M, 2026-09-06): val 3.6154** / train_eval 3.5776 / gap 0.0378 —— strict 参数对照基准（sinusoidal+旧 init）；checkpoint: `result/50M参数+998Mtokens/checkpoint_best.pt`（本地归档；云端 `/root/result_50m/`）
- **★ v3_alpha (50M, 2026-09-08, RoPE+GPT-2 init): val 3.2097** / train_eval 3.1416 / gap 0.0680 —— **webnovel_v2 评估空间当前 best**（同结构同数据较 v2 50M −0.406 nats）；checkpoint: `log/50M参数_v3_alpha+998Mtokens/checkpoint_best.pt`（本地归档；云端 `/root/result_50m_rope/`）
- **★ v3_beta (100M, 2026-09-11, 16L/704d/11H + vocab 8192): val 3.0610** / train_loss EMA 3.0780 —— **新空间（shard1+2 val）首个基准点**，**与上面所有数字不可排名**；checkpoint: 云端 `/root/result_100m/checkpoint_best.pt`（389MB，**尚未下载到本地**）；日志归档: `log/100M参数_v3_2Btokens/`

> **当前有 3 个互不可比的评估空间**：① 旧空间（tokenizer/语料/val 均不同）② webnovel_v2 空间（shard0 train / shard0 val / vocab6144）③ **shard1+2 空间**（shard1+2 train / shard1+2 val / vocab8192）。跨空间只能看量级趋势，**禁止排名**。

> 最新 ≠ 最好：跑失败/半成品实验时，**不要覆盖 `checkpoint_best.pt`**，也不要以最新 checkpoint 当作最佳结论。

## 实验事实表

| 版本 | 参数 | 配置 | 语料 | val loss | commit |
|---|---|---|---|---|---|
| 百合基线 | 7.2M | 7L/8H/256d/bs128/vocab3256 | 百合 35 本 | 3.79 | ≈85460d5 |
| LN 修复 | 6.3M | 6L/8H/256d/bs256/vocab6144/tie | 轻小说 v0 | 4.188 | ≈7a4bf61 |
| 20M 最终 | 20M | 10L/8H/384d/bs256/vocab6144/tie | v0+v1 (4.16亿 token) | 3.636 | ≈20393f0 |
| v2 首跑 (Epoch 1) | 35M | 10L/8H/512d/bs512/vocab6144/tie（实际 batch64） | webnovel_v2 shard0 (9.98亿) | 3.7076 | 614d325 |
| v2 Epoch 2 | 35M | 同 E1（const LR 5e-5 恒温续训） | webnovel_v2 shard0（第二遍，累计 19.96 亿） | 3.6372 | 16f0c1a |
| v2 50M | 51.4M | 12L/9H/576d/bs512/vocab6144/tie（sinusoidal+旧init） | webnovel_v2 shard0 (9.98亿, 单轮) | 3.6154 | fffbffe |
| **v3_alpha 50M** | 51.4M | 同结构 / **rope + GPT-2 init** / FFN=ReLU / compile | webnovel_v2 shard0 (9.98亿, 单轮) | **3.2097** | 981d35f |
| **v3_beta 100M** | **101.5M** | **16L/11H/704d** / rope + GPT-2 init / FFN=**swiglu** / vocab **8192** / compile | **shard1+2（1.96B 全新 token，单轮）** | **3.0610** ⚠️新空间 | e22aa33 |

> ⚠️ **v3_beta 那一行不在可比链里**：它的 val 是 shard1+2 自己的 split（新空间），与上表其余各行**不可排名**，列出仅为记录。

> v2/v3 可比链（同评估空间）：**v3_alpha 50M@1B 3.2097** < v2 50M@1B 3.6154 < 35M@2B 3.6372 < 35M@1B 3.7076。
> v2 50M < 35M 链 = strict 参数对照（仅参数不同）；v3_alpha 与 v2 50M 为**底层双变量升级**（rope+init），单变量贡献由 rope 短训单独支撑（详见 EXPERIMENT_LOG）。

> 完整记录在 `docs/EXPERIMENT_LOG.md`；豆包复盘报告在 `docs/report_output/`。
> 比较条件见「实验比较规则」一节（val loss 单位 nats）。

## 已验证结论

- **★ vocab 大小：8192 > 6144**（2026-09-10，`v3_vocab_cmp_lm`，各 5000 步、**同一份 val**、同一合并序列截取派生 → 严格单变量）：全量 val @step5000 **bits/char 4.1398 (8192) < 4.1784 (6144)**，**Δ = −0.92%**，**10/10 验证点方向一致**；扣除"同 step 多看 3.40% 文本"后仍 ≈ −0.83%。tokenizer 侧：4096 淘汰（压缩率损失 6.17% 只省 2.3% 参数）；6144→8192 边际收益为第一档的 52.3%。**→ vocab 定版 8192**
- **★ 方法学：per-token val_loss 跨 vocab 不可比**（同上）：词表越大单 token 信息越少，loss 天然更低——照 per-token 读会得出**完全相反**的结论（3.5999 vs 3.6906 看着 8192 差 2.5%）。跨 vocab 必须比 **bits/char** 或 **bits/byte** = `(val_loss/ln2) / chars_per_token`，chars_per_token 取完整 val 实测值
- **★ tokenizer 训练样本量：4M 已足够**（2026-09-10，阶段一）：4M→32M（**8× 数据、9.4× 时间**）压缩率只提升 **0.623%**（chars/token 1.2429→1.2507）→ 维持 4M
- **★ 增加参数 + 数据有效**（2026-09-11，v3_beta）：在**新空间**内，100M/1.96B 的 val 曲线从 3.6336 单调降到 3.0610，12 个验证点无一反弹，gap 仅 0.014→0.032（**未过拟合**，19.4 token/参数 ≈ Chinchilla 最优点）
- **参数 scaling 收益 > 重复数据收益**：50M@1B(3.6154) < 35M@2B(3.6372) < 35M@1B(3.7076)——本项目首个**同 tokenizer/语料/val/batch/LR 严格可比**结论（2026-09-06）
- **★ 新底层（RoPE + GPT-2 init）显著有效**（2026-09-08）：v3_alpha 50M(3.2097) < v2 50M(3.6154)，同结构同数据 −0.406 nats；rope 单变量短训独立支撑（−1.0~2.1 nats @ 同 step），GPT-2 init 修复起点 loss（300+ → 正常）
- **★ FFN：SwiGLU > GELU > ReLU**（2026-09-08，5000 步短程、同 seed / **从零** / cosine 8e-4（warmup 1000 → min 5e-5）、唯一变量 = `--ff-type`）：best val @5000 步 **3.655 / 3.688 / 3.695** nats → swiglu 比 relu 好 **0.040 nats**，微弱胜出（领先从 step 1000 起建立、全程一致）→ **已把默认 FFN 从 relu 改成 swiglu**（relu 保留可回退）；方法学口径：relu↔gelu 严格单变量（参数/形状/同 seed 初始化逐位一致）；swiglu 属「FFN 结构变体」对比（门控结构 + 参数量 +0.018% + 同 seed 不同形 → 起点数值不同），**非纯激活单变量**
- **增加参数 + 数据有效**：20M(3.636) < LN 修复版(4.188)（同 vocab6144/tokenizer 下可比）
- **验证集按文件/按书划分**：为避免数据泄漏（按 token 顺序切可能把整本书放进 val，评估失真）而设，使评估结果更可靠；不同实验版本的 loss 变化（如 4.5 → 3.79）**不能归因于单一因素**
- **标准 LayerNorm**（有偏方差 + eps）收敛更稳、val 更好
- **pack 模式**（stride=block_size）使 100M+ 语料每 epoch 从小时级降到分钟级
- **SDPA 训练路径**省显存，是 block_size > 256 的前提
- **gap = val − train_eval**（eval 无 dropout）是判断过拟合的正确指标；20M 版 gap 0.020，未过拟合
- **tie_embeddings + 权重去重**：大词表必备，省 ~833K 参数
- **KV cache 增量推理**与整序列重算逐位置 logits 一致（max diff <1e-4），~1.6-2.2x

## 当前未知问题

- v2 E1（3.7076）数值上未破 20M 纪录（3.636），但 **tokenizer/语料不同不可直接比**；同 tokenizer + 同 val 集下的 35M vs 20M 增益仍无可比数据
- 20M 模型继续堆数据（4.16亿 → 10 亿 token）是否继续降 loss 未验证
- block 512 相对 256 的长程收益未在同一语料上验证
- **v3 系列在更大数据（shard1/2 新 token）上是否延续 −0.4 nats 优势** → **已跑（2026-09-11，v3_beta 100M）**，但**因换了 val split 无法与 webnovel_v2 空间排名**；若要严格回答"新数据相对旧数据的增益"，需让两个空间**评在同一份 val 上**（尚未做）
- **shard1+2 空间尚无可比链**：只有 1 个点（v3_beta 100M 3.0610）。要形成链需在该 val 上再跑至少一个规模
- **v3_beta 末段下降放缓**（50k→59.9k 仅降 0.024）：是数据不够、参数不够、还是 lr 偏低（7e-4）尚无定论
- **FFN 激活能否再降 val → 已对照验证（2026-09-08，5000 步短程）**：**swiglu 3.655 < gelu 3.688 < relu 3.695**，swiglu 比 relu 好 0.040 nats → 已成为默认 FFN；v3_beta 100M 正式 run 已用 swiglu（长程收益得到间接支持）
- 中文生成质量无系统评估，目前仅主观观感

## 下一阶段计划

1. **★ 定 shard1+2 空间的比较口径（最优先）**：把 v3_alpha 的 checkpoint 也在 shard1+2 的 val 上评一遍（`scratch/eval_val_bits.py --val-txt ...`，两端都换算 bits/char），
   凑成 2×2 矩阵（两个模型 × 两个 val 集）——**这是唯一能严格回答"数据翻倍 + 参数翻倍"各贡献多少的做法**
2. **v3_beta 末段放缓的三个候选方向**（择一，勿同时动）：① 数据继续扩（shard3+，cloud 还有 shard3–9）② 参数继续扩 ③ lr 是否偏低（v3_beta 用 7e-4，v3_alpha 用 8e-4）
3. 生成质量评估：用 v3_beta / v3_alpha 跑一批示例 + 注意力热力图，主观抽检中文续写观感
4. ~~v3_beta（FFN 激活变体）~~ **已完成（2026-09-08）**：from-scratch 5000 步短程 swiglu 胜出 → 已并入主线；v3_beta 这个 ID 在 2026-09-11 被**复用为 100M 主线 run**（原 FFN 变体的正式 ID 是 `v3_ffn_variants_shortrun`）

### Scaling 原则（两阶段）

**第一阶段 — v2 综合升级实验（非单变量）**：
v2 同时升级三个维度——模型扩大（20M → 35M/50M）、context 增加（block 256 → 512）、数据增加（4.16亿 → ~1B token）。
目的：**验证整体 scaling 收益**（v2 有效 = 三个维度整体方向正确）。
> v2 是综合升级实验，**不属于严格单变量实验**，请勿要求它按单变量标准归因。

**第二阶段 — 单变量拆解**：
若 v2 有效，再通过单变量实验分别拆解各维度贡献：
1. 参数规模贡献
2. context 贡献
3. 数据规模贡献

**约束**：不要同时修改 tokenizer 和训练目标（tokenizer/vocab 改动独立于 scaling 实验，避免一次大改导致无法归因）。

## 代码约束（改代码时遵守）

- **采样默认值**：temperature 0.8 / top_p 0.9 / repetition_penalty 1.15（`model/sampling.py`）
- **LR 调度**：warmup 500 + cosine 衰减到 **max_lr×0.1**（`--min-lr-ratio 0.1`，不再到 0）
- **位置编码默认 rope**（`--position-encoding`，sinusoidal 保留可切换；推理工具自动检测旧 checkpoint）
- **FFN 默认 swiglu（2026-09-08 对照后定版）**：`--ff-type {relu,gelu,swiglu}`，**默认 `swiglu`**
  （2026-09-08 起；短程对照 swiglu 胜出后从 relu 切来；relu/gelu 原实现保留未删，可随时回退；回退点 = git commit 004ce65）
  - 默认值实现：`model/layers.py` `DEFAULT_FF_TYPE = 'swiglu'` / `FF_TYPES = ('relu','gelu','swiglu')`；
    `FeedForward(dim, dropout, ff_type=DEFAULT_FF_TYPE, ff_hidden=None)`；`train/train.py --ff-type` 默认 = swiglu
  - `relu`/`gelu`：参数名与形状**完全一致**（fc1/fc2）→ 旧 checkpoint 直接加载，单变量只换激活
  - `swiglu`：`gate_proj/up_proj → silu(gate)*up → down_proj`；中间维 h 由
    `swiglu_hidden(d)` 解 `3hd ≈ 8d²`（h≈8d/3，d=576 → **h=1536**），
    FFN 层参数与 4d 版差 **+0.036%**（层 2,657,856 vs 2,657,088）→ 参数公平
  - 残差分支输出投影的 GPT-2 缩放 init：relu/gelu 缩 fc2、swiglu 缩 down_proj
  - `--init-from <ckpt>`：**只加载模型权重**（优化器全新），用于让变体从同一 checkpoint 出发做
    **续训式**对照（**本轮 from-scratch 对照未使用**，保留备用）；
    relu↔swiglu 的权重由 `model/ffn_adapter.py` 改写（fc1 对半切→gate/up，
    fc2 前 h 列→down，**数值原样搬运、无随机数**），加载后无 missing/unexpected
  - 推理侧自动检测 ff_type：state_dict 含 `gate_proj` → swiglu，否则按 relu（旧 checkpoint
    不受默认值影响）；短程对照跑法记录见 EXPERIMENT_LOG `v3_ffn_variants_shortrun`
  - launcher：`scratch/run_ffn_scratch_cloud.sh`（云端三个变体从零顺序跑）；
    本地验证：`scratch/ffn_cpu_bench.py`；测试：`test/test_ffn.py`（12 项）
  - step CSV 新增 `tokens_per_sec` 列（实测吞吐，FFN 对比用）；run 末尾打印显存峰值
- **训练默认 --compile**（吞吐 +75%）；保存走 raw 模型 → checkpoint 无 `_orig_mod.` 前缀（推理工具 load 端已兼容前缀剥离，双保险）
- **推理工具头数**：generate/visualize 默认 n_head=8（35M）；**50M（v2 50M/v3_alpha 等）是 9 头，需显式 `--n-head 9`**
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

1. 验证集必须**按文件/按书划分**，不能按 token 顺序切——按 token 切可能造成数据泄漏（val 含训练未见的整本书）、高估 loss，评估不可靠
2. 判断过拟合看 `gap = val - train_eval`（eval 无 dropout），不是裸 val
3. tokenizer v2 哨兵：SEP=-1/DEAD=-2 是节点值不参与合并，别当邻居处理
4. resume 会重锚 cosine（warm restart，lr 跳回）——加轮次续训的正常现象
5. Windows 下 encode 并行退回单线程；云端多进程才有加速
6. tqdm 写 stderr 会让 pwsh 误报 exit 1——看输出内容而非退出码

## Experiment ID 规则

统一实验命名格式：

```
v{版本}_{模型}_ctx{context}_{数据规模}
```

**版本号语义（2026-09-08 起明确）**：
- **v1/v2 = 评估空间代际**：tokenizer/语料/val 集切换（v1 = 百合/轻小说旧空间；v2 = webnovel_v2/v6144/ctx512 空间）
- **v3 = 50M 新底层实验系列**（2026-09-08 起）：GPT-2 init + RoPE 的模型系列；**评估空间沿用 v2（webnovel_v2/v6144/同 val）** → v3 数字与 v2 系**直接可比**，不可误读为换了语料
- **v3 内后缀 = 同系列变体序号**（不是成熟度/质量排序）：`v3_alpha` = 系列第 1 版（50M / shard0 1B / ReLU FFN）；`v3_beta` = 系列第 2 版（**100M / shard1+2 1.96B / SwiGLU / vocab 8192**，2026-09-11）
  - ⚠️ 注意：FFN 激活消融那次的正式 ID 是 `v3_ffn_variants_shortrun`（**不是** v3_beta）；`v3_beta` 于 2026-09-11 用于 100M 主线 run

示例：
- `v1_20M_ctx256_400M`（20M 最终版）
- `v2_35M_ctx512_1B`（v2 综合升级实验）
- `v3_alpha_50M_ctx512_1B`（v3 系列第 1 版：rope+新init+ReLU，webnovel_v2 空间 best 3.2097）
- `v3_beta_100M_ctx512_2B`（v3 系列第 2 版：16L/704d/11H + SwiGLU + vocab8192，**shard1+2 新空间**首点 3.0610）

以下位置**统一使用该 ID**（新建实验时按格式命名）：
- checkpoint 输出目录
- log 文件（step/val 历史 CSV、run_config）
- `EXPERIMENT_LOG.md` 记录
- HTML 报告

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
