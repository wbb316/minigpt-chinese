#!/bin/bash
# 试 AutoDL 官方「学术资源加速」通道（/etc/network_turbo）
# 若可用，可从 huggingface.co 直连，速度通常远超 hf-mirror 的 ~2MB/s 限速。
echo "===== 1) network_turbo 是否存在 ====="
if [ -f /etc/network_turbo ]; then
  echo "存在 /etc/network_turbo"; cat /etc/network_turbo
else
  echo "不存在 /etc/network_turbo"
fi
echo
echo "===== 2) source 后的代理环境变量 ====="
source /etc/network_turbo 2>/dev/null
env | grep -iE 'proxy' || echo "(无代理变量)"
echo
echo "===== 3) 直连 huggingface.co（拉 100MB 测速）====="
timeout 90 curl -sL -o /dev/null \
  -w '  http=%{http_code} speed=%{speed_download} B/s got=%{size_download} bytes time=%{time_total}s\n' \
  -r 0-104857600 \
  https://huggingface.co/datasets/qqceqqq/webnovel-chinese/resolve/main/data/webnovel_1.jsonl \
  || echo "  (失败/超时)"
echo
echo "===== 4) 加速通道下再测 hf-mirror ====="
timeout 90 curl -sL -o /dev/null \
  -w '  http=%{http_code} speed=%{speed_download} B/s got=%{size_download} bytes time=%{time_total}s\n' \
  -r 0-104857600 \
  https://hf-mirror.com/datasets/qqceqqq/webnovel-chinese/resolve/main/data/webnovel_1.jsonl \
  || echo "  (失败/超时)"
echo
echo "===== 5) 测 GitHub（判断加速是否生效）====="
timeout 40 curl -sL -o /dev/null -w '  github http=%{http_code} speed=%{speed_download} B/s\n' \
  https://github.com  || echo "  (失败)"
