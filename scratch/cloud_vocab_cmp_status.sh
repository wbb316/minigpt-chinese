#!/bin/bash
# vocab 对照实验：云端状态速查（一条命令拿全所有关键信息）
echo "===== 时间 ====="; date '+%F %T'
echo "===== 进程 ====="
pgrep -af 'train/train\.py' | head -1 | cut -c1-120 || echo "(无训练进程)"
echo "worker 数: $(pgrep -cf 'train/train\.py')"
echo "===== GPU ====="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi 不可用)"
echo "===== driver.log ====="
tail -30 /root/log_vcmp/driver.log 2>/dev/null || echo "(无 driver.log)"
echo "===== v6144 进度 ====="
tail -c 500 /root/log_vcmp/train_v6144.log 2>/dev/null | tr '\r' '\n' | tail -4 || echo "(无日志)"
echo "===== v8192 进度 ====="
tail -c 500 /root/log_vcmp/train_v8192.log 2>/dev/null | tr '\r' '\n' | tail -4 || echo "(无日志)"
echo "===== 产物 ====="
ls -l /root/result_vcmp/v6144/ /root/result_vcmp/v8192/ 2>/dev/null | grep -E "checkpoint|tokenizer|:" || true
echo "===== val 曲线（每臂独立 log-dir）====="
for V in 6144 8192; do
  f=/root/log_vcmp/v$V/val_history_train_webnovel_v2.csv
  if [ -f "$f" ]; then echo "--- v$V ---"; cat "$f"; fi
done
echo "===== 验证点（训练日志）====="
grep -h '===== step' /root/log_vcmp/train_v6144.log /root/log_vcmp/train_v8192.log 2>/dev/null | tail -8
