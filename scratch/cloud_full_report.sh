#!/bin/bash
# 100M [full] 正式 run 的最终报告
LOG=/root/log_100m/driver.log
STEP=/root/log_100m/step_history_train_webnovel_shard12.csv
VAL=/root/log_100m/val_history_train_webnovel_shard12.csv

echo "=== 规模确认 ==="
grep -a -E '从缓存加载|训练集|位置编码|FFN:|GPT 参数量|LR 方案' "$LOG" 2>/dev/null | head -8

echo
echo "=== step 历史（最后 3 行）==="
tail -3 "$STEP" 2>/dev/null || echo '(无)'

echo
echo "=== 全部验证点 ==="
cat "$VAL" 2>/dev/null || echo '(无)'

echo
echo "=== 收尾信息 ==="
grep -a -E '训练结束|最佳模型|吞吐|显存峰值|AMP skipped|tokens_seen 合计' "$LOG" 2>/dev/null | tail -10

echo
echo "=== 结束标记 ==="
grep -a -c '100M \[full\] 结束' "$LOG" 2>/dev/null || echo 0
grep -a '100M \[full\] 结束' "$LOG" 2>/dev/null | tail -1
grep -a 'AUTOSHUTDOWN' "$LOG" 2>/dev/null | tail -2

echo
echo "=== 产物 ==="
ls -lh /root/result_100m/ 2>/dev/null

echo
echo "=== 进程 / GPU ==="
echo -n 'python 进程数: '; pgrep -c python || echo 0
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo '(GPU 已随关机消失)'
