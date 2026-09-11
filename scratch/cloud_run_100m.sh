#!/bin/bash
# 100M 模型训练脚本（16L/704d/11H，vocab 8192，shard1+2 单遍 ≈1.94B token）
#
# 用法:
#   bash cloud_run_100m.sh smoke   # 200-500 步体检：显存/AMP skip/loss/吞吐
#   bash cloud_run_100m.sh full    # 正式 run（完整单遍，~59,246 步）
#
# 配置来源（2026-09-11 用户确认）：
#   16L/704d/11H, head_dim=64, RoPE + GPT-2 init + SwiGLU, block_size=512 固定,
#   peak LR 7e-4, batch 64, warmup 2000, cosine, min-lr-ratio 0.0625,
#   tie_embeddings, dropout 0.1, seed 42, fp16 + compile
#   vocab=8192（依据：6144 vs 8192 的 5000 步 LM 对照，bits/char 8192 好 0.92%，
#                10/10 个验证点方向一致，见 log/vocab对照6144vs8192/）
#
# 参数量：101,457,280（实测；vocab 8192）
#
# 自动关机：跑完（无论成功失败）调用平台提供的 /usr/bin/shutdown，避免
#   「人不在、任务跑完了、GPU 空转继续计费」。smoke 默认**不**关机（要立刻看结果）；
#   full 默认**关机**。可用 AUTOSHUTDOWN=0/1 显式覆盖。
set -u
cd /root

# ⚠️ 必须提高文件描述符上限：
#    ShardMemmap.__init__ 会把**所有**分片一次性 np.memmap 打开，
#    shard12 语料 2.49B 字符 → 1660 片，超过容器默认 soft limit 1024 →
#    OSError: [Errno 24] Too many open files（2026-09-11 首次 smoke 就栽在这）
#    硬上限 1048576，提到 65535 足够。
ulimit -n 65535 2>/dev/null || echo "⚠️ ulimit 提升失败，当前: $(ulimit -n)"

MODE=${1:-smoke}
if [ "$MODE" = "full" ]; then AUTOSHUTDOWN=${AUTOSHUTDOWN:-1}; else AUTOSHUTDOWN=${AUTOSHUTDOWN:-0}; fi

# 语料放**系统盘** /root/data（克隆会带走 → 抗丢），缓存/编码临时文件放**数据盘**。
# 好处：系统盘只承担 语料8.22G + checkpoint~1.7G；数据盘承担 缓存3.9G + 临时7.4G。
CORPUS=/root/data
DATA=/root/autodl-tmp/data
TRAIN=$CORPUS/train_webnovel_shard12.txt
VAL=$CORPUS/val_webnovel_shard12.txt
TOK_SRC=/root/result_vcmp/v8192/tokenizer_v8192_s4000000.pkl

if [ "$MODE" = "smoke" ]; then
  OUT=/root/result_100m_smoke
  LOG=/root/log_100m_smoke
  EXTRA="--max-steps 350 --val-every 300 --eval-batches 50 --warmup-steps 60"
else
  OUT=/root/result_100m
  LOG=/root/log_100m
  EXTRA="--val-every 5000 --eval-batches 100 --warmup-steps 2000"
fi
mkdir -p "$OUT" "$LOG"

# ---- 前置检查 ----
for f in "$TRAIN" "$VAL"; do
  [ -f "$f" ] || { echo "❌ 缺语料: $f"; exit 1; }
done
[ -f "$TOK_SRC" ] || { echo "❌ 缺 tokenizer: $TOK_SRC"; exit 1; }

# 预置 tokenizer：train.py 只在 out-dir 里找 tokenizer_v{V}_s{tokens_sample}.pkl，
# 文件名必须完全一致，否则它会自己去重训一个（浪费 12 分钟且不是同一个词表）
cp "$TOK_SRC" "$OUT/tokenizer_v8192_s4000000.pkl" || exit 1

echo "=================== 100M [$MODE] 开始 $(date +%F\ %T) ==================="
echo "语料: $TRAIN"
echo "输出: $OUT   日志: $LOG"

/root/miniconda3/bin/python -u train/train.py \
  --train-txt "$TRAIN" \
  --val-txt "$VAL" \
  --sample-mode pack --batch-size 64 --block-size 512 \
  --n-layer 16 --n-head 11 --n-embd 704 \
  --vocab-size 8192 --tie-embeddings --tokens-sample 4000000 \
  --bpe-trainer fast --cache-format shards \
  --lr 7e-4 --min-lr-ratio 0.0625 --weight-decay 0.05 \
  --epochs 1 --patience 5 \
  --encode-workers 16 --position-encoding rope --seed 42 \
  --compile \
  $EXTRA \
  --cache-dir "$DATA" --log-dir "$LOG" --out-dir "$OUT" 2>&1
RC=$?

echo "=================== 100M [$MODE] 结束 $(date +%F\ %T) exit=$RC ==================="
ls -l "$OUT" 2>/dev/null

# ---- 跑完自动关机（省掉「人不在还空转计费」）----
if [ "$AUTOSHUTDOWN" = "1" ]; then
  echo "AUTOSHUTDOWN=1 → 结果已落盘，准备关机（$(date +%F\ %T)）"
  sync
  sleep 5
  shutdown
fi
echo "AUTOSHUTDOWN=$AUTOSHUTDOWN，不关机。"
