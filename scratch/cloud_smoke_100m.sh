#!/bin/bash
# 100M 模型 smoke test：16L/704d/11H / vocab8192 / swiglu / rope / tie / block512 / batch64
#
# 为什么现在就能跑：训练语料用**云端已有**的 shard0（train_webnovel_v2.txt），
# 它的 v8192 token 缓存（train_v8192_len1241765192_e1）在 vocab 对照实验里刚建好，
# 所以无需等待 shard1+2 上传，立刻能验证架构本身。
#
# 目的（用户要求）：确认 显存 / AMP skip / loss / 吞吐 正常，再决定正式 run 的 batch。
# 语料不同不影响显存占用，所以这个 smoke test 对正式 run 有效。
set -u
cd /root
OUT=/root/result_100m_smoke
LOG=/root/log_100m_smoke
DATA=/root/autodl-tmp/data
mkdir -p "$OUT" "$LOG"

# 预置 tokenizer：文件名必须是 tokenizer_v{V}_s{tokens_sample}.pkl
cp /root/result_vcmp/v8192/tokenizer_v8192_s4000000.pkl "$OUT/" || { echo "❌ tokenizer 拷贝失败"; exit 1; }
echo "=================== smoke test 开始 $(date +%F\ %T) ==================="

/root/miniconda3/bin/python -u train/train.py \
  --train-txt "$DATA/train_webnovel_v2.txt" \
  --val-txt "$DATA/val_webnovel_v2.txt" \
  --sample-mode pack --batch-size 64 --block-size 512 \
  --n-layer 16 --n-head 11 --n-embd 704 \
  --vocab-size 8192 --tie-embeddings --tokens-sample 4000000 \
  --bpe-trainer fast --cache-format shards \
  --lr 7e-4 --min-lr-ratio 0.0625 --warmup-steps 60 --weight-decay 0.05 \
  --epochs 1 --max-steps 350 --val-every 300 --patience 99 --eval-batches 50 \
  --encode-workers 16 --position-encoding rope --seed 42 \
  --compile \
  --cache-dir "$DATA" --log-dir "$LOG" --out-dir "$OUT" 2>&1

echo "=================== smoke test 结束 $(date +%F\ %T) exit=$? ==================="
