#!/bin/bash
# 探测容器内关机手段（**只查看，不执行任何关机命令**）
echo "=== 可用的关机类命令 ==="
for c in shutdown poweroff halt reboot; do
  p=$(command -v $c 2>/dev/null)
  if [ -n "$p" ]; then
    echo "  $c -> $p  ($(file -b "$p" 2>/dev/null | cut -c1-70))"
  else
    echo "  $c -> 无"
  fi
done
echo
echo "=== 是否为脚本（若是脚本，说明是平台包装）==="
for p in /usr/sbin/shutdown /sbin/shutdown /usr/bin/shutdown /usr/local/bin/shutdown; do
  if [ -e "$p" ]; then
    echo "--- $p ---"
    head -c 300 "$p" 2>/dev/null | head -12
    echo
  fi
done
echo "=== AutoDL 相关工具 ==="
ls /usr/local/bin/ 2>/dev/null | head -20
echo
echo "=== 提示文件 ==="
ls -la /etc/motd /root/README* /etc/autodl* 2>/dev/null | head
cat /etc/motd 2>/dev/null | head -20
