#!/bin/bash
# 测 hf-mirror 上真实文件的下载速度（拉 100MB range）
BASE=https://hf-mirror.com/datasets/qqceqqq/webnovel-chinese/resolve/main/data
for n in 1 2; do
  echo "--- webnovel_${n}.jsonl ---"
  timeout 90 curl -sL -o /dev/null \
    -w '  http=%{http_code} speed=%{speed_download} B/s got=%{size_download} bytes time=%{time_total}s\n' \
    -r 0-104857600 "$BASE/webnovel_${n}.jsonl" || echo "  (失败/超时)"
done
echo
echo "--- 与本地分片大小核对（应一致）---"
echo "  webnovel_1.jsonl 期望 4198277512"
echo "  webnovel_2.jsonl 期望 4191247565"
timeout 30 curl -sIL "$BASE/webnovel_1.jsonl" | grep -iE '^HTTP/|content-length' | head -6
