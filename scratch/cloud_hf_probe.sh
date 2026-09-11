#!/bin/bash
# 查 hf-mirror 上 qqceqqq/webnovel-chinese 的真实文件结构 + 测真实下载速度
echo "===== 1) 数据集 API（文件清单）====="
timeout 40 curl -sL https://hf-mirror.com/api/datasets/qqceqqq/webnovel-chinese | head -c 3000
echo
echo
echo "===== 2) 404 的响应体到底是什么 ====="
timeout 30 curl -sL https://hf-mirror.com/datasets/qqceqqq/webnovel-chinese/resolve/main/webnovel_1.jsonl
echo
echo
echo "===== 3) 试 head 一下几个候选路径 ====="
for u in \
  "https://hf-mirror.com/datasets/qqceqqq/webnovel-chinese/resolve/main/README.md" \
  "https://hf-mirror.com/api/datasets/qqceqqq/webnovel-chinese/tree/main" \
  "https://hf-mirror.com/api/datasets/qqceqqq/webnovel-chinese/tree/main?recursive=true" ; do
  echo "--- $u"
  timeout 30 curl -sL -o /dev/null -w '  http=%{http_code} size=%{size_download}\n' "$u"
done
echo
echo "===== 4) 若 tree 可列，打印前 40 行 ====="
timeout 40 curl -sL "https://hf-mirror.com/api/datasets/qqceqqq/webnovel-chinese/tree/main?recursive=true" | head -c 4000
echo
