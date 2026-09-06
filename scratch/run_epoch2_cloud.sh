#!/bin/bash
# Epoch 2 续跑（resume next_epoch=1 → 只跑 epoch 1，共 30463 步）
# LR 方案: const 恒 5e-5（用户确认方案 1；docs/RESUME_AUDIT.md）
set -e
cd /root
mkdir -p log result_webnovel_v2
nohup /root/miniconda3/bin/python -u train/train.py \
  --train-txt /root/autodl-tmp/data/train_webnovel_v2.txt \
  --val-txt /root/autodl-tmp/data/val_webnovel_v2.txt \
  --sample-mode pack --batch-size 64 --block-size 512 \
  --n-layer 10 --n-head 8 --n-embd 512 \
  --vocab-size 6144 --tie-embeddings --tokens-sample 4000000 \
  --bpe-trainer fast --cache-format shards \
  --lr 8e-4 --min-lr-ratio 0.0625 --warmup-steps 1000 --weight-decay 0.05 \
  --epochs 2 --val-every 5000 --patience 2 --eval-batches 100 \
  --encode-workers 16 --lr-scheme const \
  --out-dir /root/result_webnovel_v2 \
  --cache-dir /root/autodl-tmp/data \
  --log-dir /root/log \
  --resume /root/result_webnovel_v2/checkpoint_latest.pt \
  > /root/log/train_webnovel_v2_e2.log 2>&1 &
echo "Epoch2 已后台启动 PID $!"
echo "日志: tail -f /root/log/train_webnovel_v2_e2.log"
