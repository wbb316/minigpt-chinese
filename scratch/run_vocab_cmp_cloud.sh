#!/bin/bash
# vocab 对照：6144 vs 8192，各 5000 步（2026-09-10）
#
# 目的：tokenizer 第二阶段（vocab 消融）已证明 4096 淘汰、6144 vs 8192 只差
#       3.44% token；intrinsic 指标无法裁定 → 用一次短程 LM 训练给出定论。
#
# 设计（唯一变量 = vocab）
#   - 结构/优化/数据/schedule 全部照抄 v3_alpha 生产配方：
#       12L / 576d / 9H / swiglu / tie / rope / dropout0.1
#       batch64 × block512 = 32768 token/step，lr 8e-4，warmup 1000，
#       min-lr-ratio 0.0625，wd 0.05，seed 42，fp16 + compile
#   - total_steps 仍按整 epoch(=30463) 计算，--max-steps 5050 只是提前停
#     → 本实验 = 生产配方的**前 5000 步**，LR 轨迹与生产完全一致
#     ⚠️ 用 5050 而非 5000：train.py 的 max-steps break 在 val 检查**之前**，
#        设成 5000 会把 step 5000 那次验证吞掉（且 epoch 末验证也会被跳过）→
#        就没有 step 5000 的 val 点、checkpoint_best 会停在 step 4500。
#        5050 让 step 5000 的验证正常执行，之后 50 步不影响对照（两臂一致）。
#   - val_every 500（10 个点看 curve），eval_batches 100（快速子集）
#
# ⚠️ per-token val_loss **不可跨 vocab 比较**（词表越大单 token 信息越少）。
#    定论看 scratch/eval_val_bits.py 的全量验证集 bits/char、bits/byte。
#    本脚本里的 val 只用于看 curve 形状与选 best。
#
# 数据：/root/autodl-tmp/data/{train,val}_webnovel_v2.txt（与 v3_alpha 同一份）
set -u
cd /root

DATA=/root/autodl-tmp/data
OUT=/root/result_vcmp
LOG=/root/log_vcmp
mkdir -p "$OUT" "$LOG" "$LOG/v6144" "$LOG/v8192"

COMMON="--train-txt $DATA/train_webnovel_v2.txt \
  --val-txt $DATA/val_webnovel_v2.txt \
  --sample-mode pack --batch-size 64 --block-size 512 \
  --n-layer 12 --n-head 9 --n-embd 576 \
  --tie-embeddings --tokens-sample 4000000 \
  --bpe-trainer fast --cache-format shards \
  --lr 8e-4 --min-lr-ratio 0.0625 --warmup-steps 1000 --weight-decay 0.05 \
  --epochs 1 --max-steps 5050 --val-every 500 --patience 99 --eval-batches 100 \
  --encode-workers 16 --position-encoding rope --seed 42 \
  --compile \
  --cache-dir $DATA"

run_one () {
  local V=$1
  local out=$OUT/v$V
  local log=$LOG/train_v$V.log
  # 每臂独立 log-dir：val_history/step_history 的文件名只按语料 basename 取，
  # 共用 log-dir 会让两次 run 的行混进同一个 CSV（step 还会重置），难以分离。
  local tlog=$LOG/v$V
  mkdir -p "$out" "$tlog"
  echo "=================== vocab=$V  start $(date +%F\ %T) ==================="

  # 预置 tokenizer：文件名必须是 tokenizer_v{V}_s{4000000}.pkl，train.py 才会直接加载
  if [ ! -f "$out/tokenizer_v${V}_s4000000.pkl" ]; then
    echo "❌ 缺少预置 tokenizer: $out/tokenizer_v${V}_s4000000.pkl"
    return 1
  fi

  /root/miniconda3/bin/python -u train/train.py $COMMON \
    --vocab-size "$V" \
    --log-dir "$tlog" \
    --out-dir "$out" > "$log" 2>&1
  local rc=$?
  echo "  vocab=$V exit=$rc  $(date +%F\ %T)"

  # 关键信息回显（含「是否真的用了预置 tokenizer」）
  grep -E '从缓存加载分词器|tokenizer词表大小|参数量|位置编码|FFN:|训练集|验证集' "$log" | head -8
  grep -E '训练结束|最佳模型|吞吐' "$log" | tail -3
  echo "--- 最后 3 个 val 点 ---"
  grep '===== step' "$log" | tail -3
  echo
  return $rc
}

# ---- 前置检查 ----
for V in 6144 8192; do
  if [ ! -f "$OUT/v$V/tokenizer_v${V}_s4000000.pkl" ]; then
    echo "❌ $OUT/v$V/tokenizer_v${V}_s4000000.pkl 缺失，先上传 tokenizer"
    exit 1
  fi
done
for f in "$DATA/train_webnovel_v2.txt" "$DATA/val_webnovel_v2.txt"; do
  if [ ! -f "$f" ]; then echo "❌ 数据缺失: $f"; exit 1; fi
done
echo "✅ 前置检查通过（tokenizer + 数据就位）"
echo

run_one 6144
run_one 8192

echo "=================== 两次训练完成 $(date +%F\ %T) ==================="
echo
echo "下一步：全量验证集可比评测（bits/char、bits/byte）"
echo "  python -u scratch/eval_val_bits.py --label v6144 \\"
echo "    --ckpt $OUT/v6144/checkpoint_best.pt \\"
echo "    --tokenizer $OUT/v6144/tokenizer_v6144_s4000000.pkl \\"
echo "    --val-txt $DATA/val_webnovel_v2.txt \\"
echo "    --n-layer 12 --n-head 9 --n-embd 576 --block-size 512 \\"
echo "    --out $OUT/eval_v6144.json"
echo "  （v8192 同理换路径与 --tokenizer）"
