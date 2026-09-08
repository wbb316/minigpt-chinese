#!/bin/bash
# v2 50M RoPE 正式训练（从零，完整 1B，单 epoch）
# 栈：50M(12L/576d/9H) + 新 GPT-2 init + rope + fp16 + compile + batch64
# 对比基线：旧 50M E1（sinusoidal+旧init, val 3.6154）—— 本 run 验证新底层是否更低
set -e
cd /root
mkdir -p log result_50m_rope
if [ ! -f /root/result_50m_rope/tokenizer_v6144_s4000000.pkl ]; then
  cp /root/result_50m/tokenizer_v6144_s4000000.pkl /root/result_50m_rope/
fi
nohup /root/miniconda3/bin/python -u train/train.py \
  --train-txt /root/autodl-tmp/data/train_webnovel_v2.txt \
  --val-txt /root/autodl-tmp/data/val_webnovel_v2.txt \
  --sample-mode pack --batch-size 64 --block-size 512 \
  --n-layer 12 --n-head 9 --n-embd 576 \
  --vocab-size 6144 --tie-embeddings --tokens-sample 4000000 \
  --bpe-trainer fast --cache-format shards \
  --lr 8e-4 --min-lr-ratio 0.0625 --warmup-steps 1000 --weight-decay 0.05 \
  --epochs 1 --val-every 5000 --patience 2 --eval-batches 100 \
  --encode-workers 16 --position-encoding rope --seed 42 \
  --compile \
  --out-dir /root/result_50m_rope \
  --cache-dir /root/autodl-tmp/data \
  --log-dir /root/log \
  > /root/log/train_50m_rope.log 2>&1 &
echo "50M RoPE 正式训练已启动 PID $!"
echo "日志: tail -f /root/log/train_50m_rope.log"
