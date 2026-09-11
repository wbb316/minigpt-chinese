#!/bin/bash
# 云端下载 shard1+shard2 原始 jsonl 并清洗成训练语料（全自动，无人值守）
#
# 背景（2026-09-10）：
#   - 本地上传到云端实测只有 0.51 MB/s → 7.84GB 要 4.3 小时（gzip 后仍 ~2 小时）
#   - AutoDL 学术加速(/etc/network_turbo)对 HF 无效；且会把 hf-mirror 代理坏
#     （hf-mirror.com 不在 no_proxy 里）
#   - 不加代理直连 hf-mirror 单连接 ~2 MB/s，8.39GB ≈ 70 分钟
#   - 并发分段实测**更慢**（8 并发聚合 0.99 MB/s）→ hf-mirror 按客户端限速，加并发无用
#   - 文件真实路径是 data/webnovel_N.jsonl（漏 data/ 会 404）
#
# 产出：/root/autodl-tmp/data/train_webnovel_shard12.txt
#       /root/autodl-tmp/data/val_webnovel_shard12.txt
set -u
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # 必须走直连

RAW=/root/autodl-tmp/raw
PREP=/root/autodl-tmp/prep        # prepare_webnovel.py 的输出落在 <其父目录>/data
DATA=/root/autodl-tmp/data
LOG=/root/log_prep
BASE=https://hf-mirror.com/datasets/qqceqqq/webnovel-chinese/resolve/main/data
PY=/root/miniconda3/bin/python
mkdir -p "$RAW" "$PREP/data" "$DATA" "$LOG"

# 官方分片大小（用于完整性校验）
SIZE_1=4198277512
SIZE_2=4191247565

echo "=================== 开始 $(date +%F\ %T) ==================="

# ---------- 1) 下载 ----------
for n in 1 2; do
  f="$RAW/webnovel_${n}.jsonl"
  echo "--- 下载 webnovel_${n}.jsonl（断点续传）---"
  curl -L -C - -o "$f" "$BASE/webnovel_${n}.jsonl" 2>&1 | tail -2
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  echo "  实得 $sz 字节"
done

# ---------- 2) 校验大小 ----------
ok=1
for n in 1 2; do
  eval "exp=\$SIZE_$n"
  f="$RAW/webnovel_${n}.jsonl"
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$sz" = "$exp" ]; then
    echo "✅ webnovel_${n}.jsonl 大小正确 ($sz)"
  else
    echo "❌ webnovel_${n}.jsonl 大小不符: 实得 $sz 期望 $exp"
    ok=0
  fi
done
if [ "$ok" != "1" ]; then
  echo "❌ 下载不完整，终止（可重跑本脚本断点续传）"
  exit 1
fi

# ---------- 3) 清洗（输出到 autodl-tmp，避免写满 30G 的系统盘）----------
if [ ! -f "$PREP/data/prepare_webnovel.py" ]; then
  cp /root/data/prepare_webnovel.py "$PREP/data/" || { echo "❌ 找不到 prepare_webnovel.py"; exit 1; }
fi
echo "--- 清洗 --shards 1 2 ---"
$PY "$PREP/data/prepare_webnovel.py" --src "$RAW" --shards 1 2 2>&1 | tail -8

# ---------- 4) 改名落到 data/ ----------
if [ -f "$PREP/data/train_webnovel_v2.txt" ]; then
  mv "$PREP/data/train_webnovel_v2.txt" "$DATA/train_webnovel_shard12.txt"
  echo "✅ train: $(stat -c %s "$DATA/train_webnovel_shard12.txt") 字节"
fi
if [ -f "$PREP/data/val_webnovel_v2.txt" ]; then
  mv "$PREP/data/val_webnovel_v2.txt" "$DATA/val_webnovel_shard12.txt"
  echo "✅ val:   $(stat -c %s "$DATA/val_webnovel_shard12.txt") 字节"
fi

# ---------- 5) 清掉原始 jsonl 省空间（要重下就重跑本脚本）----------
rm -f "$RAW"/webnovel_1.jsonl "$RAW"/webnovel_2.jsonl
rmdir "$RAW" 2>/dev/null || true

echo "ALL_DONE $(date +%F\ %T)" > "$LOG/ALL_DONE"
echo "=================== 完成 $(date +%F\ %T) ==================="
df -h /root/autodl-tmp | tail -1
