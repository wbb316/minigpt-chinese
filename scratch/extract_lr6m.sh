#!/bin/bash
# 提取 5 组 LR6M 的完整 val 轨迹
cd /root
for name in S1_cosine_fast S2_cosine_slow S3_wsd S4_hold_decay S5_const_tail; do
  echo "===== $name ====="
  grep '===== step' /root/log_lr6m/train_${name}.log | grep -oE 'step [0-9]+.*val [0-9.]+ \| train_eval\(无dropout\) [0-9.]+ \| gap [+-][0-9.]+'
  grep -E '训练结束|最佳模型|吞吐' /root/log_lr6m/train_${name}.log | tail -2
done
