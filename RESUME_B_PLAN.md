# B 组恢复清单（过几天再训）

> 最后更新: 2026-09-04 22:43。训练推迟几天，回来按此清单继续。
> 核心目标: 数据扩到 ~1B token + 模型 30-50M + 上下文 512（SDPA）。

## 1️⃣ 语料下载（应该已完成——先确认）

```powershell
# 确认 3 个分片都下完（每个应 ~3.9GB = 4,188,xx4,xxx 字节附近）
Get-ChildItem 'D:\小说\webnovel' -Filter '*.jsonl' | Select Name, Length
```

预期（全部完成后）:
- `D:\小说\webnovel\webnovel_0.jsonl` / `_1` / `_2`，各 ~3.9GB，合计 ~11.7GB

如果没下完/被中断（断点续传，重跑即可）:
```powershell
$proxy = 'http://127.0.0.1:7897'   # 梯子端口，变了就改
foreach ($i in 0..2) {
  $url = "https://huggingface.co/datasets/qqceqqq/webnovel-chinese/resolve/main/data/webnovel_${i}.jsonl?download=true"
  $out = "D:\小说\webnovel\webnovel_${i}.jsonl"
  curl.exe -s -L --proxy $proxy -A 'python-requests' -C - -o $out $url
}
```

## 2️⃣ 清洗切分（本地，轻量）

```bash
# 先只处理 1 个分片评估 token 量（v2 后缀, 不碰旧语料）
python data/prepare_webnovel.py --shards 0
# 产物: data/train_webnovel_v2.txt + data/val_webnovel_v2.txt
```

注意: 全量 3 分片 ≈ 30 亿 token, 对 30-50M 模型严重过量。
**先用 1 分片 (~10亿 token), 不够再加**。短书(<30000字符)已被脚本丢弃。

## 3️⃣ 云端 4090（AutoDL）

1. 开机
2. 上传（在本地 D:\WBB_Python\pytorch 执行）:
```bash
# SDPA 版注意力(训练省显存, block 512 的前提) + min_lr 版训练脚本
scp -i .autodl_key\id_ed25519 -P 24932 model/attention.py train/train.py data/prepare_webnovel.py \
    root@connect.westb.seetacloud.com:/root/minigpt/
# 语料 txt（若在本地处理完）
scp -i .autodl_key\id_ed25519 -P 24932 data/train_webnovel_v2.txt data/val_webnovel_v2.txt \
    root@connect.westb.seetacloud.com:/root/minigpt/data/
```
> 云端实际路径以 `find /root -name train.py` 为准; 端口/域名可能变化, 以 AutoDL 控制台为准。

3. 训练命令（草案, 30-50M 模型）:
```bash
python train/train.py \
  --train-txt data/train_webnovel_v2.txt --val-txt data/val_webnovel_v2.txt \
  --sample-mode pack --batch-size 256 --block-size 512 \
  --n-layer 10 --n-head 8 --n-embd 512 \
  --vocab-size 6144 --tie-embeddings --tokens-sample 4000000 \
  --lr 8e-4 --warmup-steps 500 --weight-decay 0.05 \
  --out-dir result_webnovel --cache-dir data \
  --epochs 3
```
> n_embd 512 × 10 层 ≈ 40M 参数。block 512 必须 SDPA(attention.py 已含)。
> 首次运行会训练新 tokenizer + 编码 ~10亿 token, 需较长时间(多进程已并行)。

## 4️⃣ 本地验证 demo（可选）

```bash
python app/server.py --ckpt result_webnovel/checkpoint_best.pt \
                     --tokenizer result_webnovel/tokenizer_best.pkl
```

## 关键文件状态（都已 commit + push 到 GitHub wbb316/minigpt-chinese）

- `model/sampling.py` + `model/generation.py` — 采样三参数 + KV cache 超窗引擎 ✅
- `model/attention.py` — SDPA 训练路径 + KV cache 手写路径 ✅
- `train/train.py` — `--min-lr-ratio 0.1`(cosine 不到 0) ✅
- `data/prepare_webnovel.py` — jsonl → 按真书 90/10 清洗切分 ✅
- `.web_readme.md` 说明该数据集 = HF qqceqqq/webnovel-chinese (apache-2.0, 9000本, 36GB)

## 数据量背景速查

| 语料 | token | 状态 |
|---|---|---|
| v0+v1 (旧, train_lightnovel_all) | 4.16 亿 | 20M 模型已用, val 3.636 |
| webnovel 1 分片 | ~10 亿(估) | 待清洗 |
| webnovel 3 分片 | ~30 亿(估) | 留作长期语料库 |
| Chinchilla 建议 (30-50M 模型) | 0.6-1B | 目标 |
