#!/bin/bash
# 等待 4 组 tokenizer 训练完成，然后自动生成提醒（写 DONE 标记 + 打印汇总）
# ★ 修掉之前 grep -c 的 bug：用 tr -d 清空空白，或直接判断文件存在
cd /root
mkdir -p /root/tok_exp/log

for i in $(seq 1 360); do            # 最多等 6 小时（360 × 60s）
  n=0
  for s in 4M 8M 16M 32M; do
    [ -f "/root/tok_exp/tok/tok_${s}.pkl" ] && n=$((n + 1))
  done
  if [ "$n" -ge 4 ]; then
    echo "===== ALL_DONE $(date +%F\ %T) ====="
    echo "4 组 tokenizer 全部训练完成："
    for s in 4M 8M 16M 32M; do
      f=/root/tok_exp/log/train_${s}.log
      if [ -f "$f" ]; then
        printf "  %-4s " "$s"
        grep -oE 'RESULT.*' "$f" | tail -1
      fi
    done
    touch /root/tok_exp/ALL_DONE
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT: 只完成 $n/4 组"
exit 1
