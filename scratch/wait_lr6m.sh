#!/bin/bash
# 等待 5 组 LR6M 全部跑完（driver 出现 5 次 exit）
cd /root
for i in $(seq 1 240); do
  n=$(grep -c 'exit=' /root/log_lr6m/lr6m_driver.log 2>/dev/null || echo 0)
  if [ "$n" -ge 5 ]; then
    echo "ALL_DONE after ~$((i*30))s"
    tail -3 /root/log_lr6m/lr6m_driver.log
    exit 0
  fi
  sleep 30
done
echo "TIMEOUT 120min"
exit 1
