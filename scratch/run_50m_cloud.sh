#!/bin/bash
# v3 50M (12L/576d/9H = 51.4M) 从头训练：shard0 1B token，单 epoch
# 与 v2 35M E1 严格可比（同 tokenizer/语料/val/batch64/lr 调度/epochs=1）
set -e
cd /root
mkdir -p log result_50m
# tokenizer 缓存：复用 v2 的（vocab6144/s4000000，shard0 训练）→ 免 ~30min 重训
if [ ! -f /root/result_50m/tokenizer_v6144_s4000000.pkl ]; then
  cp /root/result_webnovel_v2/tokenizer_v6144_s4000000.pkl /root/result_50m/
  echo "已复用 v2 tokenizer 缓存"
fi
# 冒烟优先：batch64 若 OOM（预估 ~20GB），降到 48 并如实记录（对照带 batch 变量）
nohup /root/miniconda3/bin/python -u train/train.py \
  --train-txt /root/autodl-tmp/data/train_webnovel_v2.txt \
  --val-txt /root/autodl-tmp/data/val_webnovel_v2.txt \
  --sample-mode pack --batch-size 64 --block-size 512 \
  --n-layer 12 --n-head 9 --n-embd 576 \
  --vocab-size 6144 --tie-embeddings --tokens-sample 4000000 \
  --bpe-trainer fast --cache-format shards \
  --lr 8e-4 --min-lr-ratio 0.0625 --warmup-steps 1000 --weight-decay 0.05 \
  --epochs 1 --val-every 5000 --patience 2 --eval-batches 100 \
  --encode-workers 16 --lr-scheme cosine \
  --out-dir /root/result_50m \
  --cache-dir /root/autodl-tmp/data \
  --log-dir /root/log \
  > /root/log/train_50m.log 2>&1 &
echo "50M 训练已后台启动 PID $!"
echo "日志: tail -f /root/log/train_50m.log"
