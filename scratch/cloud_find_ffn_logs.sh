#!/bin/bash
# 查 FFN 实验的曲线原始数据还在不在（result_ffn_* 目录里只有 .pt 和 tokenizer，
# val/step CSV 应该在别的 log 目录）
echo "=== /root 下的 log 目录 ==="
ls -d /root/log* 2>/dev/null || echo "(无)"

echo
echo "=== 全盘找 ffn 相关文件 ==="
find /root -iname '*ffn*' 2>/dev/null | head -40 || echo "(无)"

echo
echo "=== 各 log 目录内容 ==="
for d in /root/log /root/log_lr6m /root/log_vcmp /root/log_100m /root/log_100m_smoke /root/log_prep /root/log_ffn; do
  if [ -d "$d" ]; then
    echo "--- $d ---"
    ls -1 "$d" 2>/dev/null | head -15
  fi
done

echo
echo "=== 找所有 val_history / step_history 文件 ==="
find /root -name 'val_history_*.csv' -o -name 'step_history_*.csv' 2>/dev/null | head -30
