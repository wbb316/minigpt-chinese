#!/bin/bash
# 提取 from-scratch 三个 FFN 变体的验证轨迹 + best + 吞吐
for f in relu gelu swiglu; do
  echo "===== $f ====="
  grep '===== step' /root/log/train_ffn_scratch_${f}.log | grep -oE 'step [0-9]+.*val [0-9.]+ \| train_eval\(无dropout\) [0-9.]+ \| gap [+-][0-9.]+'
  grep -E '训练结束|吞吐' /root/log/train_ffn_scratch_${f}.log | tail -2
done
