#!/bin/bash
# RoPE vs sinusoidal 短训对比（50M 从零 / 新 GPT-2 init / 同 seed 42 / 各 4000 步）
# 固定：模型 12L-9H-576d / batch64 / lr 8e-4 / warmup 1000 / --compile（两组同栈公平）
# 输出：log/train_rope_cmp_{sinusoidal,rope}.log + rope_cmp_out/{pe}/ 独立产物
set -e
cd /root
mkdir -p log rope_cmp_out

for PE in sinusoidal rope; do
  mkdir -p /root/rope_cmp_out/$PE
  cp /root/result_50m/tokenizer_v6144_s4000000.pkl /root/rope_cmp_out/$PE/ 2>/dev/null || true
  echo "===== 开始 $PE ($(date +%H:%M:%S)) ====="
  /root/miniconda3/bin/python -u train/train.py \
    --train-txt /root/autodl-tmp/data/train_webnovel_v2.txt \
    --val-txt /root/autodl-tmp/data/val_webnovel_v2.txt \
    --sample-mode pack --batch-size 64 --block-size 512 \
    --n-layer 12 --n-head 9 --n-embd 576 \
    --vocab-size 6144 --tie-embeddings --tokens-sample 4000000 \
    --bpe-trainer fast --cache-format shards \
    --lr 8e-4 --min-lr-ratio 0.0625 --warmup-steps 1000 --weight-decay 0.05 \
    --epochs 1 --val-every 1000 --patience 10 --eval-batches 100 \
    --encode-workers 16 --position-encoding $PE --seed 42 \
    --compile --max-steps 4000 \
    --out-dir /root/rope_cmp_out/$PE \
    --cache-dir /root/autodl-tmp/data \
    --log-dir /root/log \
    > /root/log/train_rope_cmp_${PE}.log 2>&1
  echo "===== $PE 完成 ($(date +%H:%M:%S)) ====="
done
echo "ALL DONE — 对比日志: log/train_rope_cmp_sinusoidal.log / train_rope_cmp_rope.log"
