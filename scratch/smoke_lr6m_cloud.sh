#!/bin/bash
# 云端 6M + wsd schedule 冒烟：验证新代码路径 + 实测 step/s
set -u
cd /root
/root/miniconda3/bin/python -u train/train.py \
  --train-txt /root/data_baihe/train.txt \
  --val-txt /root/data_baihe/val.txt \
  --sample-mode pack --batch-size 16 --block-size 256 \
  --n-layer 6 --n-head 8 --n-embd 256 \
  --vocab-size 6144 --tie-embeddings --dropout 0.1 \
  --tokens-sample 4000000 --bpe-trainer fast \
  --lr 8e-4 --warmup-steps 250 --weight-decay 0.05 \
  --epochs 1 --max-steps 30 --val-every 1000 --patience 99 --eval-batches 10 \
  --encode-workers 16 --position-encoding rope --seed 42 \
  --lr-scheme wsd --min-lr-ratio 0.0625 \
  --cache-dir /root/data_baihe --log-dir /root/log_lr6m \
  --out-dir /root/result_lr6m/_smoke \
  > /root/log_lr6m/smoke_wsd.log 2>&1
echo "exit=$?"
grep -E 'LR 方案|参数量|适配|加载|step 3|tokens/s|Error|Traceback|错误' /root/log_lr6m/smoke_wsd.log | tail -12
