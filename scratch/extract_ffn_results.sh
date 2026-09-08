#!/bin/bash
# 提取三个 FFN 变体的验证轨迹 + best
for f in relu gelu swiglu; do
  echo "===== $f ====="
  grep '===== step' /root/log/train_ffn_${f}.log | grep -oE 'step [0-9]+.*val [0-9.]+ \| train_eval\(无dropout\) [0-9.]+ \| gap [+-][0-9.]+'
  grep -E '训练结束|最佳模型' /root/log/train_ffn_${f}.log | tail -2
done
