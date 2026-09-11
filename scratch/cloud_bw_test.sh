#!/bin/bash
# 测试云端到 HF / hf-mirror 的连通性与下载速度
# 目的：7.8GB 语料从本地上传只有 0.51 MB/s（≈2 小时，gzip 后），
#       若云端能自己下 webnovel_1/2.jsonl（8GB）会快得多。
echo "===== 网络出口 ====="
curl -s --max-time 10 https://api.ipify.org; echo
echo "===== DNS ====="
getent hosts huggingface.co hf-mirror.com 2>/dev/null || echo "(getent 不可用)"
echo
echo "===== huggingface.co 直连（拉 50MB 测速）====="
timeout 60 curl -sL -o /dev/null -w 'http=%{http_code} speed=%{speed_download} B/s got=%{size_download} bytes\n' \
  -r 0-50000000 \
  https://huggingface.co/datasets/qqceqqq/webnovel-chinese/resolve/main/webnovel_1.jsonl \
  || echo "(失败/超时)"
echo
echo "===== hf-mirror.com（拉 50MB 测速）====="
timeout 60 curl -sL -o /dev/null -w 'http=%{http_code} speed=%{speed_download} B/s got=%{size_download} bytes\n' \
  -r 0-50000000 \
  https://hf-mirror.com/datasets/qqceqqq/webnovel-chinese/resolve/main/webnovel_1.jsonl \
  || echo "(失败/超时)"
echo
echo "===== 文件头信息（确认能拿到真文件）====="
timeout 30 curl -sIL https://hf-mirror.com/datasets/qqceqqq/webnovel-chinese/resolve/main/webnovel_1.jsonl \
  | grep -iE 'HTTP/|content-length|content-type|location' | head -12
echo
echo "===== 云端是否已有 python 环境可用 ====="
ls -l /root/miniconda3/bin/python
python3 -c "import sys; print('python3', sys.version)" 2>/dev/null || echo "(系统 python3 不可用)"
