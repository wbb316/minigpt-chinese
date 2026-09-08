#!/bin/bash
# 云端等待器：阻塞直到三个 FFN from-scratch 变体全部跑完（driver 出现 3 次 "exit="）
# 每 30 秒查一次，完成后打印最终汇总
cd /root
for i in $(seq 1 200); do
  n=$(grep -c 'exit=' /root/log/ffn_scratch_driver.log 2>/dev/null || echo 0)
  if [ "$n" -ge 3 ]; then
    echo "ALL_DONE after ~$((i*30))s wait"
    tail -5 /root/log/ffn_scratch_driver.log
    exit 0
  fi
  sleep 30
done
echo "TIMEOUT after 100 min"
exit 1
