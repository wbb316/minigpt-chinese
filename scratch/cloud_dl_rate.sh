#!/bin/bash
# 测 hf-mirror 的**持续**下载速率（60 秒采样）
# 短时 range 测速是突发值（1.6-2.3 MB/s），实际持续可能被限速得多。
f=/root/autodl-tmp/raw/webnovel_1.jsonl
if [ ! -f "$f" ]; then echo "❌ $f 不存在"; exit 1; fi
s1=$(stat -c %s "$f")
t1=$(date +%s)
sleep 60
s2=$(stat -c %s "$f")
t2=$(date +%s)
dt=$((t2 - t1))
db=$((s2 - s1))
echo "采样 ${dt}s，增加 ${db} 字节"
if [ "$dt" -gt 0 ] && [ "$db" -gt 0 ]; then
  rate=$((db / dt))
  echo "持续速率 = ${rate} B/s = $((rate / 1024)) KB/s = $(awk -v r=$rate 'BEGIN{printf "%.2f", r/1048576}') MB/s"
  echo "当前大小 = ${s2} 字节"
  echo "按此速率下完 8.39GB 需要 $(awk -v r=$rate 'BEGIN{printf "%.1f", 8389545077/r/3600}') 小时"
else
  echo "⚠️ 采样期内没有增长（下载可能已停或卡住）"
fi
echo "--- curl 进程 ---"
pgrep -af 'curl.*webnovel' | grep -v pgrep | head -3 || echo "(无)"
