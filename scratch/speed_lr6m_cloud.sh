#!/bin/bash
# 6M 测速 v2：batch16 vs batch64（block256），各 150 步测稳态 step/s
# ★ out-dir 预置 tokenizer 缓存，避免触发 400 万字符 BPE 重训（会卡几分钟）
set -u
cd /root
for BS in 16 64; do
  echo "===== batch=$BS ====="
  OUT=/root/result_lr6m/_speed_bs$BS
  mkdir -p $OUT
  cp /root/data_baihe/tokenizer_v6144_s4000000.pkl $OUT/ 2>/dev/null
  /root/miniconda3/bin/python -u train/train.py \
    --train-txt /root/data_baihe/train.txt \
    --val-txt /root/data_baihe/val.txt \
    --sample-mode pack --batch-size $BS --block-size 256 \
    --n-layer 6 --n-head 8 --n-embd 256 \
    --vocab-size 6144 --tie-embeddings --dropout 0.1 \
    --tokens-sample 4000000 --bpe-trainer fast \
    --lr 8e-4 --warmup-steps 100 --weight-decay 0.05 \
    --epochs 1 --max-steps 150 --val-every 1000 --patience 99 --eval-batches 10 \
    --encode-workers 16 --position-encoding rope --seed 42 \
    --lr-scheme cosine --min-lr-ratio 0.0625 --compile \
    --cache-dir /root/data_baihe --log-dir /root/log_lr6m \
    --out-dir $OUT \
    > /root/log_lr6m/speed_bs$BS.log 2>&1
  echo "  exit=$?"
  grep -E '吞吐|tokens/s' /root/log_lr6m/speed_bs$BS.log | tail -1
done
echo SPEED_DONE
