#!/bin/bash
# 6M LR schedule 筛选实验（from-scratch，B 版 5 组）2026-09-08
#
# 目的：用 6M(6L/8H/256d/vocab6144/tie) + 百合 90/10（train ≈20.5M token 单遍）
# 筛选 LR schedule。唯一变量 = lr-scheme（含 min-lr-ratio），其余完全固定。
# ⚠️ 评估空间 = v1 百合（与 webnovel_v2 的 50M 不可比），只做组内相对比较；
#    LR 结论后续须在 50M v3 上短程验证，不能直接外推。
#
# 配置（5 组完全一致，仅 schedule 不同）：
#   6L/8H/256d/vocab6144/tie/dropout0.1/block256/batch16(=4096 tok/step)
#   peak lr 8e-4，warmup 250 步（≈总步 5%），from-scratch，seed 42
#   总步 = ceil(train_tokens/4096) ≈ 5010（编码后精确值，用 --max-steps 5010 兜底）
#   val_every 250（≈20 个验证点，看 curve 细节），eval_batches 100
#   不早停（patience 99），让每组完整跑完
#
# S1 基线(当前式):  cosine,            min_ratio 0.0625   （快速衰减到 5e-5）
# S2 慢 cosine:     cosine,            min_ratio 0.3      （衰减温和，尾部 lr 2.4e-4）
# S3 WSD:           wsd,               min_ratio 0.0625   （stable 至 75% 再衰减）
# S4 hold-decay:    hold-decay,        min_ratio 0.0625   （保持至 60% 再衰减）
# S5 const-tail:    const-tail,        min_ratio 0.0625   （前 50% 衰减后恒 min）
set -u
cd /root

mkdir -p /root/result_lr6m
COMMON="--train-txt /root/data_baihe/train.txt \
  --val-txt /root/data_baihe/val.txt \
  --sample-mode pack --batch-size 16 --block-size 256 \
  --n-layer 6 --n-head 8 --n-embd 256 \
  --vocab-size 6144 --tie-embeddings --dropout 0.1 \
  --tokens-sample 4000000 --bpe-trainer fast \
  --lr 8e-4 --warmup-steps 250 --weight-decay 0.05 \
  --epochs 1 --max-steps 5150 --val-every 250 --patience 99 --eval-batches 100 \
  --encode-workers 16 --position-encoding rope --seed 42 --compile \
  --cache-dir /root/data_baihe --log-dir /root/log_lr6m"

# 预拷贝 tokenizer（避免各 run 重训；out-dir 名含 scheme）
for name in S1_cosine_fast S2_cosine_slow S3_wsd S4_hold_decay S5_const_tail; do
  mkdir -p /root/result_lr6m/$name
  cp /root/data_baihe/tokenizer_v6144_s4000000.pkl /root/result_lr6m/$name/ 2>/dev/null || true
done

run_one () {
  local name=$1; shift
  local out=/root/result_lr6m/$name
  local log=/root/log_lr6m/train_${name}.log
  mkdir -p /root/log_lr6m
  echo "=================== $name  start $(date +%F\ %T) ==================="
  /root/miniconda3/bin/python -u train/train.py $COMMON "$@" \
    --out-dir "$out" > "$log" 2>&1
  echo "  $name exit=$? $(date +%F\ %T)"
  grep -E 'LR 方案|参数量|训练结束|最佳模型|tokens/s|吞吐' "$log" | tail -5
}

# 顺序跑 5 组
run_one S1_cosine_fast   --lr-scheme cosine     --min-lr-ratio 0.0625
run_one S2_cosine_slow   --lr-scheme cosine     --min-lr-ratio 0.3
run_one S3_wsd           --lr-scheme wsd        --min-lr-ratio 0.0625
run_one S4_hold_decay    --lr-scheme hold-decay --min-lr-ratio 0.0625
run_one S5_const_tail    --lr-scheme const-tail --min-lr-ratio 0.0625

echo
echo "=================== 全部完成 $(date +%F\ %T) ==================="
echo "汇总："
for name in S1_cosine_fast S2_cosine_slow S3_wsd S4_hold_decay S5_const_tail; do
  echo "--- $name ---"
  grep '===== step' /root/log_lr6m/train_${name}.log | tail -3
done
