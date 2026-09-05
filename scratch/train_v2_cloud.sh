#!/bin/bash
# v2 训练：35M (10L/512d) + webnovel ~1B token + fast tokenizer + shards 编码
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
  --epochs 1 --val-every 5000 --patience 2 --eval-batches 100 \
  --encode-workers 16 \
  --out-dir /root/result_webnovel_v2 \
  --cache-dir /root/autodl-tmp/data \
  --log-dir /root/log \
  > /root/log/train_webnovel_v2.log 2>&1 &
echo "训练已后台启动 PID $!"
echo "日志: tail -f /root/log/train_webnovel_v2.log"
