#!/bin/bash
# 云端：拼装分片上传的语料 → 校验 md5 → 解压 → 校验大小
# 用法: bash cloud_assemble.sh <name> <expected_md5_of_gz> <expected_bytes_of_text>
set -u
UP=/root/autodl-tmp/upload
NAME=$1
EXP_MD5=$2
EXP_TEXT=$3
GZ=$UP/${NAME}.txt.gz
OUT=/root/autodl-tmp/data/${NAME}.txt

mkdir -p /root/autodl-tmp/data

if [ ! -f "$GZ" ]; then
  echo "--- 拼装 ${NAME} ---"
  ls -1 $UP/${NAME}.txt.gz.part* 2>/dev/null | sort > /tmp/parts_$NAME.txt
  n=$(wc -l < /tmp/parts_$NAME.txt)
  echo "分片数: $n"
  if [ "$n" = "0" ]; then echo "❌ 没有分片"; exit 1; fi
  cat $(cat /tmp/parts_$NAME.txt) > "$GZ"
fi
echo "gz 大小: $(stat -c %s "$GZ")"

echo "--- 校验 gz md5 ---"
got=$(md5sum "$GZ" | cut -d' ' -f1)
if [ "$got" != "$EXP_MD5" ]; then
  echo "❌ md5 不符"
  echo "   实得 $got"
  echo "   期望 $EXP_MD5"
  exit 1
fi
echo "✅ md5 一致"

echo "--- 解压 ---"
gunzip -c "$GZ" > "$OUT" || { echo "❌ 解压失败"; exit 1; }
sz=$(stat -c %s "$OUT")
if [ "$sz" != "$EXP_TEXT" ]; then
  echo "❌ 解压后大小不符: 实得 $sz 期望 $EXP_TEXT"
  exit 1
fi
echo "✅ $OUT  $sz 字节"
rm -f "$GZ" $(cat /tmp/parts_$NAME.txt 2>/dev/null) 2>/dev/null
echo "已清理分片"
