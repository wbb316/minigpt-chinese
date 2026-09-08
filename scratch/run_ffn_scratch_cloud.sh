#!/bin/bash
# v3 FFN 变体 FROM-SCRATCH 公平对照（2026-09-08）
#
# 背景：续训式对照（从 v3_alpha relu 权重出发）里 relu 天然占优——激活切换要花步数
# 从扰动中恢复，5000 步只是恢复期。本脚本改为 from-scratch：三个激活同 seed 从头训练，
# 唯一变量 = ff-type，公平回答「哪个激活更好」。
#
# 配置完全复刻 rope vs sinusoidal 短训（2026-09-08 10:55/11:05 run）：
#   50M(12L/576d/9H) / vocab6144 / tie / pack / batch64 / ctx512
#   lr 8e-4 cosine（warmup 1000 → min 5e-5）/ weight_decay 0.05 / fp16 / compile
#   seed 42（数据流一致）/ max-steps 5000 / val_every 1000
#
# 产出：/root/result_ffn_scratch_{relu,gelu,swiglu}/ + 日志
# 对照参考：rope 短训（同配置 rope 激活）val@3000 ≈ 3.78
set -u
cd /root

STEPS=5000
VAL_EVERY=1000
SEED=42
TOK_SRC=/root/result_50m_rope/tokenizer_v6144_s4000000.pkl
COMMON="--train-txt /root/autodl-tmp/data/train_webnovel_v2.txt \
  --val-txt /root/autodl-tmp/data/val_webnovel_v2.txt \
  --sample-mode pack --batch-size 64 --block-size 512 \
  --n-layer 12 --n-head 9 --n-embd 576 \
  --vocab-size 6144 --tie-embeddings --tokens-sample 4000000 \
  --bpe-trainer fast --cache-format shards \
  --lr 8e-4 --min-lr-ratio 0.0625 --warmup-steps 1000 --weight-decay 0.05 \
  --lr-scheme cosine --dropout 0.1 --precision fp16 \
  --epochs 1 --max-steps ${STEPS} --val-every ${VAL_EVERY} --patience 99 --eval-batches 100 \
  --encode-workers 16 --position-encoding rope --seed ${SEED} --compile \
  --cache-dir /root/autodl-tmp/data --log-dir /root/log"

run_one () {
  local ff=$1
  local out=/root/result_ffn_scratch_${ff}
  local log=/root/log/train_ffn_scratch_${ff}.log
  mkdir -p "$out"
  if [ ! -f "$out/tokenizer_v6144_s4000000.pkl" ] && [ -f "$TOK_SRC" ]; then
    cp "$TOK_SRC" "$out/"
  fi
  echo "=================== SCRATCH FFN=$ff  start $(date +%F\ %T) ==================="
  /root/miniconda3/bin/python -u train/train.py $COMMON \
    --ff-type "$ff" \
    --out-dir "$out" > "$log" 2>&1
  echo "  exit=$? $(date +%F\ %T)"
  grep -E 'FFN:|参数量|tokens/s|显存峰值|训练结束|最佳模型' "$log" | tail -6
}

for ff in relu gelu swiglu; do
  run_one "$ff"
done

echo
echo "=================== 汇总（各 run 的 val 轨迹） ==================="
for ff in relu gelu swiglu; do
  echo "--- $ff ---"
  grep '===== step' /root/log/train_ffn_scratch_${ff}.log | grep -oE 'step [0-9]+.*val [0-9.]+ \| train_eval\(无dropout\) [0-9.]+ \| gap [+-][0-9.]+'
  grep -E '训练结束' /root/log/train_ffn_scratch_${ff}.log | tail -1
done
