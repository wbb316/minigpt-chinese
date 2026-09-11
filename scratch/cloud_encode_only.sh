#!/bin/bash
# 【只建 token 缓存，不训练】
#
# 为什么单独一个脚本：编码是 CPU 活（~5 分钟），训练是 GPU 活（~5 小时）。
# 用户要求先把编码做掉，训练等他回宿舍再开 —— 所以这里只调 encode_to_cache，
# 完全不碰 train.py。
#
# 缓存命名与 train.py 内部的调用完全一致（encode_to_cache(text, tok, cache_dir, tag)），
# 所以之后 train.py 启动时会**直接命中**这两个缓存，跳过编码。
#
# 用 n_procs=32（机器有 128 核）：缓存命中校验只认 tokenizer hash/vocab/len(text)/version，
# 与分片数无关，所以之后 train.py 用 n_procs=16 也照样命中。
set -u
cd /root
DATA=/root/autodl-tmp/data
TOK=/root/result_vcmp/v8192/tokenizer_v8192_s4000000.pkl

if [ ! -f "$TOK" ]; then echo "❌ 缺 tokenizer: $TOK"; exit 1; fi
for f in train_webnovel_shard12.txt val_webnovel_shard12.txt; do
  [ -f "$DATA/$f" ] || { echo "❌ 缺语料: $DATA/$f"; exit 1; }
done

echo "=================== 仅编码 开始 $(date +%F\ %T) ==================="
df -h /root/autodl-tmp | tail -1

/root/miniconda3/bin/python -u - <<'PYEOF'
import os, pickle, sys, time
sys.path.insert(0, '/root')
from data.token_cache import encode_to_cache

DATA = '/root/autodl-tmp/data'
TOK = '/root/result_vcmp/v8192/tokenizer_v8192_s4000000.pkl'
tok = pickle.load(open(TOK, 'rb'))
print(f'tokenizer: vocab={len(tok.vocab)} merges={len(tok.merges)}', flush=True)

for tag, fn in (('train', 'train_webnovel_shard12.txt'),
                ('val',   'val_webnovel_shard12.txt')):
    path = os.path.join(DATA, fn)
    t0 = time.time()
    print(f'--- {tag}: 读取 {path} ---', flush=True)
    text = open(path, encoding='utf-8').read()
    print(f'{tag}: {len(text):,} 字符，开始编码 (n_procs=32) ...', flush=True)
    p = encode_to_cache(text, tok, DATA, tag, n_procs=32)
    print(f'{tag}: ✅ → {p}  ({time.time()-t0:.0f}s)', flush=True)
    del text

print('ENCODE_ALL_DONE', flush=True)
PYEOF

echo "=================== 仅编码 结束 $(date +%F\ %T) ==================="
echo "缓存目录:"
ls -d /root/autodl-tmp/data/train_v8192_* /root/autodl-tmp/data/val_v8192_* 2>/dev/null
du -sh /root/autodl-tmp/data/*/ 2>/dev/null | tail -6
df -h /root/autodl-tmp | tail -1
echo ">>> 编码完成，按用户要求**不启动训练**。"
echo ">>> 缓存写在 /root/autodl-tmp（数据盘），**关机不会丢**，回来开机即可继续。"

# ---- 可选自动关机（默认关：用户可能马上回来训练）----
if [ "${AUTOSHUTDOWN:-0}" = "1" ]; then
  echo "AUTOSHUTDOWN=1 → 10 分钟宽限期后关机（留窗口给远端核验结果）"
  echo "取消方式: pkill -f 'sleep 600'"
  sync
  sleep 600
  echo "宽限期结束，关机 $(date +%F\ %T)"
  shutdown
fi
