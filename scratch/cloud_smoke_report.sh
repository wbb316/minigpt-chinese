#!/bin/bash
# 100M smoke test 体检报告（从 driver.log 抽关键行，避开 tqdm 的 \r 刷屏）
LOG=/root/log_100m_smoke/driver.log
STEP=/root/log_100m_smoke/step_history_train_webnovel_shard12.csv
VAL=/root/log_100m_smoke/val_history_train_webnovel_shard12.csv

echo "=== 配置与规模 ==="
grep -a -E '从缓存加载分词器|tokenizer词表大小|位置编码|FFN:|GPT 参数量|训练集|验证集' "$LOG" 2>/dev/null | head -10

echo
echo "=== 训练进度（最后 3 个 step 行）==="
tail -3 "$STEP" 2>/dev/null || echo '(无 step csv)'

echo
echo "=== 验证点 ==="
cat "$VAL" 2>/dev/null || echo '(还没验证)'

echo
echo "=== 收尾信息 ==="
grep -a -E '训练结束|最佳模型|吞吐|显存峰值|AMP skipped' "$LOG" 2>/dev/null | tail -8

echo
echo "=== 是否结束 ==="
grep -a -c '100M \[smoke\] 结束' "$LOG" 2>/dev/null || echo 0

echo
echo "=== 进程 / GPU ==="
echo -n 'python 进程数: '; pgrep -c python || echo 0
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
