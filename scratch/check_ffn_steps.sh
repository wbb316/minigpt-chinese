#!/bin/bash
# 确认每个变体实际跑到的步数
for f in relu gelu swiglu; do
  echo "===== $f ====="
  # 日志里最后一次 step 进度
  grep -oE '[0-9]+/30463' /root/log/train_ffn_${f}.log | tail -1
  # run_config 或结束信息
  grep -E '已完成|max_steps|5000 步' /root/log/train_ffn_${f}.log | tail -2
  # 验证 CSV 里该 run 的最后几行（按时间）
  echo "--- val CSV 尾部(该run) ---"
  tail -4 /root/log/val_history_train_webnovel_v2.csv
done
