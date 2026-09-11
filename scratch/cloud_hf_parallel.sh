#!/bin/bash
# 测 hf-mirror 并发分段下载的**聚合**带宽
# 单连接实测 1.6-2.3 MB/s；若 8 并发能叠到 6+ MB/s，8.4GB 就只要 ~20 分钟。
# 注意：云端**没有** python3，只有 /root/miniconda3/bin/python（上一版因此报 command not found）。
PY=/root/miniconda3/bin/python
BASE=https://hf-mirror.com/datasets/qqceqqq/webnovel-chinese/resolve/main/data/webnovel_1.jsonl
rm -rf /tmp/dltest; mkdir -p /tmp/dltest
SEG=$((32*1024*1024))          # 每段 32MB
N=8
start=$($PY -c 'import time;print(time.time())')
for i in $(seq 0 $((N-1))); do
  a=$((i*SEG)); b=$((a+SEG-1))
  curl -sL -o /tmp/dltest/p$i -r $a-$b "$BASE" &
done
wait
end=$($PY -c 'import time;print(time.time())')
$PY - "$start" "$end" <<'PYEOF'
import os, sys
start, end = float(sys.argv[1]), float(sys.argv[2])
d = '/tmp/dltest'
files = os.listdir(d)
tot = sum(os.path.getsize(os.path.join(d, f)) for f in files)
dt = end - start
print(f'并发 {len(files)} 段 x 32MB')
print(f'总字节 {tot:,}  用时 {dt:.1f}s  聚合速率 {tot/dt/1048576:.2f} MB/s')
if tot > 0 and dt > 0:
    print(f'推算 8.39GB 需要 {8389545077/(tot/dt)/60:.1f} 分钟')
PYEOF
rm -rf /tmp/dltest
