#!/bin/bash
# 100M+2B 单屏进度视图 —— 配 watch 用:
#   watch -n 15 -t bash /root/scratch/progress_bar.sh
STEP=/root/log_100m/step_history_train_webnovel_shard12.csv
VAL=/root/log_100m/val_history_train_webnovel_shard12.csv
TOTAL=59886
PER_STEP=32768

last=$(tail -1 "$STEP" 2>/dev/null)
if [ -z "$last" ]; then
  echo "还没有 step 记录（可能在 torch.compile 编译阶段）"
  exit 0
fi

cur=$(echo "$last" | cut -d, -f1)
loss=$(echo "$last" | cut -d, -f4)
lr=$(echo "$last" | cut -d, -f6)
tps=$(echo "$last" | cut -d, -f7)
ts=$(echo "$last" | cut -d, -f8)

pct=$(awk -v a="$cur" -v b="$TOTAL" 'BEGIN{printf "%.1f", 100*a/b}')
width=44
filled=$(awk -v p="$pct" -v w="$width" 'BEGIN{printf "%d", p*w/100}')
empty=$((width - filled))
bar=$(printf "%${filled}s" | tr ' ' '#')
pad=$(printf "%${empty}s" | tr ' ' '.')

eta=$(awk -v c="$cur" -v t="$TOTAL" -v s="$PER_STEP" -v r="$tps" \
      'BEGIN{ if (r>0) printf "%.2f", (t-c)*s/r/3600; else print "?" }')
gone=$(awk -v c="$cur" -v s="$PER_STEP" 'BEGIN{printf "%.2f", c*s/1e9}')

echo "=============== 100M (16L/704d/11H) + shard1+2 ==============="
echo
echo "  [$bar$pad]  $pct%"
echo
echo "  step      : $cur / $TOTAL"
echo "  token     : ${gone}B / 1.96B"
echo "  train loss: $loss"
echo "  lr        : $lr"
echo "  吞吐      : $tps tok/s"
echo "  剩余      : ${eta} 小时"
echo "  更新于    : $ts"
echo
echo "---------------------- 验证点 ----------------------"
# 注意：云端没有 column 命令，直接 cat（CSV 逗号分隔，够看）
tail -6 "$VAL" 2>/dev/null
echo
echo "（每 5000 步一个验证点；跑完会自动关机）"
