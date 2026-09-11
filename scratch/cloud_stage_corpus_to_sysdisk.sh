#!/bin/bash
# 【在 westb 老机上跑】把 shard12 语料放到**系统盘** /root/data，好让下一次克隆把它带走
#
# 为什么：AutoDL 克隆只复制系统盘（/），不复制数据盘（/root/autodl-tmp）。
# 语料原本在数据盘 → 克隆后就没有；放到 /root/data 后，克隆出来的新机直接可用，
# 省掉 50 分钟的跨机/跨网上传。
#
# 空间：系统盘 30G 已用 19G（可用 12G），语料 8.22GB 塞进去只剩 2.8G，太紧
#       → 先删 /root/data 里 4.6GB 的旧 .npy token 缓存（Sep 1-3 轻小说时代，
#         可由同目录的 train_lightnovel*.txt 重建）→ 腾到 8.4GB 可用
set -u

SRC=/root/autodl-tmp/data
DST=/root/data
mkdir -p "$DST"

echo "=================== 开始 $(date +%F\ %T) ==================="
echo "--- 清理前 ---"
df -h / | tail -1
du -sh "$DST"

echo "--- 待删的旧 token 缓存（可重建）---"
ls -lh "$DST"/*.npy 2>/dev/null || echo "(无)"
rm -f "$DST"/*.npy
echo "--- 删除后 ---"
df -h / | tail -1

for f in train_webnovel_shard12.txt val_webnovel_shard12.txt; do
  echo "--- 拷贝 $f  $(date +%T) ---"
  cp "$SRC/$f" "$DST/$f" || { echo "❌ $f 拷贝失败"; exit 1; }
  echo "  完成 $(date +%T)"
done

echo "--- 校验 ---"
for f in train_webnovel_shard12.txt val_webnovel_shard12.txt; do
  a=$(stat -c %s "$SRC/$f")
  b=$(stat -c %s "$DST/$f")
  if [ "$a" = "$b" ]; then echo "✅ $f  $b 字节"; else echo "❌ $f 大小不符 src=$a dst=$b"; fi
done

echo "--- /root/data 最终内容 ---"
ls -lh "$DST"
echo "--- 磁盘 ---"
df -h / | tail -1
echo "=================== 完成 $(date +%F\ %T) ===================="
echo "STAGE_DONE"
