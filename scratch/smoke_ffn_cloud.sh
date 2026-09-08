#!/bin/bash
# FFN 变体 30 步冒烟：验证 --ff-type relu/gelu/swiglu + --init-from v3_alpha 加载无误
set -u
cd /root
COMMON="--train-txt /root/autodl-tmp/data/train_webnovel_v2.txt \
  --val-txt /root/autodl-tmp/data/val_webnovel_v2.txt \
  --sample-mode pack --batch-size 64 --block-size 512 \
  --n-layer 12 --n-head 9 --n-embd 576 \
  --vocab-size 6144 --tie-embeddings --tokens-sample 4000000 \
  --bpe-trainer fast --cache-format shards \
  --lr 8e-4 --min-lr-ratio 0.0625 --warmup-steps 0 --weight-decay 0.05 \
  --lr-scheme const --dropout 0.1 --precision fp16 \
  --epochs 1 --max-steps 30 --val-every 1000 --patience 99 --eval-batches 10 \
  --encode-workers 16 --position-encoding rope --seed 42 \
  --cache-dir /root/autodl-tmp/data --log-dir /root/log \
  --init-from /root/v3_alpha_ckpt/checkpoint_best.pt"

for ff in relu gelu swiglu; do
  echo "===== SMOKE $ff ====="
  /root/miniconda3/bin/python -u train/train.py $COMMON \
    --ff-type "$ff" --out-dir /root/scratch/_ffn_smoke_$ff > /root/scratch/_ffn_smoke_$ff.log 2>&1
  echo "exit=$?"
  grep -E 'FFN|参数量|适配|加载|init|起始 loss|step 30|错误|Error|Traceback|missing|unexpected' /root/scratch/_ffn_smoke_$ff.log | head -12
done
echo "SMOKE DONE"
