# -*- coding: utf-8 -*-
"""跨词表可比的验证集评测：val_loss / bits-per-char / bits-per-byte。

为什么需要它
------------
不同 vocab 的 per-token val_loss **不可直接比较**：词表越大，单个 token 承载的
信息越少，per-token loss 天然更低。要让 6144 与 8192 可比，必须把 loss 归一化到
与词表无关的单位上：

    bits/char = (val_loss_nats / ln2) × (n_val_tokens / n_val_chars)
    bits/byte = (val_loss_nats / ln2) × (n_val_tokens / n_val_bytes)

两者都与词表大小无关，可直接跨 vocab 比较（越小越好）。
val_loss 本身照常输出，但只作为「各自词表空间内」的参考值。

评测协议与 train.py 的验证完全一致
----------------------------------
- 同一份 held-out 文本（val-txt）
- PackedDataset(block_size, offset=block_size//2)，不重叠
- fp16 autocast，token 级交叉熵
- 遍历**全部**验证块（不是 100 批的子集）→ 无采样噪声
- 模型结构：显式传参，与训练 run 保持一致

用法
----
  python scratch/eval_val_bits.py \\
    --ckpt result_vcmp/v6144/checkpoint_best.pt \\
    --tokenizer result_vcmp/v6144/tokenizer_v6144_s4000000.pkl \\
    --val-txt /root/autodl-tmp/data/val_webnovel_v2.txt \\
    --n-layer 12 --n-head 9 --n-embd 576 --block-size 512 \\
    --batch-size 32 --out result_vcmp/eval_v6144.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import PackedDataset          # noqa: E402
from model.gpt import GPT                       # noqa: E402
from model.layers import swiglu_hidden, DEFAULT_FF_TYPE  # noqa: E402

LN2 = math.log(2.0)

_ENC_TOK = None


def _enc_init(tok):
    global _ENC_TOK
    _ENC_TOK = tok


def _enc_slice(s):
    return np.asarray(_ENC_TOK.encode(s), dtype=np.int64)


def encode_text(text, tokenizer, n_procs=16):
    """多进程切块编码（按换行对齐，避免切断字符/词）。"""
    if n_procs <= 1:
        return np.asarray(tokenizer.encode(text), dtype=np.int64)
    import multiprocessing as mp
    n = max(1, min(n_procs, len(text) // 2_000_000))
    step = len(text) // n
    slices = []
    for i in range(n):
        a = i * step
        b = len(text) if i == n - 1 else (i + 1) * step
        if i > 0:
            nl = text.find('\n', a)
            if 0 <= nl < b:
                a = nl + 1
        if i < n - 1:
            nl = text.rfind('\n', a, b)
            if nl > a:
                b = nl + 1
        slices.append(text[a:b])
    with mp.Pool(n, initializer=_enc_init, initargs=(tokenizer,)) as pool:
        parts = pool.map(_enc_slice, slices)
    return np.concatenate(parts)


def strip_compile_prefix(sd):
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--val-txt', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--label', default='')
    # 模型结构（必须与训练 run 完全一致）
    ap.add_argument('--n-layer', type=int, default=12)
    ap.add_argument('--n-head', type=int, default=9)
    ap.add_argument('--n-embd', type=int, default=576)
    ap.add_argument('--block-size', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--position-encoding', default='rope')
    ap.add_argument('--tie-embeddings', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--ff-type', default=DEFAULT_FF_TYPE)
    ap.add_argument('--ff-hidden', type=int, default=0)
    ap.add_argument('--ff-hidden-round', type=int, default=0)
    # 评测
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-blocks', type=int, default=0, help='0 = 全部验证块')
    ap.add_argument('--encode-workers', type=int, default=16)
    ap.add_argument('--precision', default='fp16', choices=['fp16', 'bf16'])
    ap.add_argument('--passes', type=int, default=1, help='大于 1 可检查确定性')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.float16 if args.precision == 'fp16' else torch.bfloat16

    # ---------- tokenizer + 语料 ----------
    import pickle
    with open(args.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)
    V = len(tokenizer.vocab)

    with open(args.val_txt, encoding='utf-8') as f:
        text = f.read()
    n_chars = len(text)
    n_bytes = len(text.encode('utf-8'))

    t0 = time.perf_counter()
    ids = encode_text(text, tokenizer, args.encode_workers)
    enc_dt = time.perf_counter() - t0
    n_tok = len(ids)
    print(f'验证文本: {n_chars:,} 字符 / {n_bytes:,} 字节 → {n_tok:,} token '
          f'(vocab={V}, chars/token={n_chars/n_tok:.4f}, 编码 {enc_dt:.1f}s)')

    # ---------- 模型 ----------
    ff_hidden = args.ff_hidden or (swiglu_hidden(args.n_embd, args.ff_hidden_round)
                                   if args.ff_type == 'swiglu' else 0)
    gpt = GPT(vocab_size=V, block_size=args.block_size, n_layer=args.n_layer,
              n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
              tie_embeddings=args.tie_embeddings,
              position_encoding=args.position_encoding,
              ff_type=args.ff_type, ff_hidden=ff_hidden or None).to(device)

    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    sd = ck.get('model', ck) if isinstance(ck, dict) else ck
    sd = strip_compile_prefix(sd)
    missing, unexpected = gpt.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith('cos_cached')]
    assert not missing, f'❌ 权重缺失（会静默随机初始化）: {missing[:5]}'
    assert not unexpected, f'❌ 多余的权重键: {unexpected[:5]}'
    print(f'✅ 权重加载完成: {args.ckpt}  (参数量 {gpt.get_num_params():,})')
    gpt.eval()

    # ---------- 全量验证 ----------
    ds = PackedDataset(ids, args.block_size, offset=args.block_size // 2)
    n_blocks = len(ds) if args.max_blocks <= 0 else min(args.max_blocks, len(ds))
    print(f'验证块: 用 {n_blocks:,} / 共 {len(ds):,} （block={args.block_size}, '
          f'offset={args.block_size//2}）')

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, list(range(n_blocks))),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    runs = []
    for p in range(max(1, args.passes)):
        total_nll = 0.0
        total_cnt = 0
        t0 = time.perf_counter()
        with torch.no_grad():
            for bi, (x, y) in enumerate(loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast('cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
                    logits = gpt(x)
                s = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                    y.reshape(-1), reduction='sum')
                total_nll += float(s.double())
                total_cnt += y.numel()
                if bi % 200 == 0:
                    el = time.perf_counter() - t0
                    print(f'  pass{p+1} {bi}/{n_blocks} blocks  '
                          f'({total_cnt/1e6:.1f}M token, {el:.0f}s, '
                          f'{total_cnt/max(el,1e-9)/1e3:.1f}k tok/s)', flush=True)
        dt = time.perf_counter() - t0
        mean_nll = total_nll / total_cnt
        runs.append((mean_nll, total_cnt, dt))
        print(f'pass{p+1}: val_loss={mean_nll:.6f} nats/token  '
              f'({total_cnt:,} token, {dt:.0f}s, '
              f'{total_cnt/dt/1e3:.1f}k tok/s)')

    mean_nll, total_cnt, dt = runs[0]
    res = {
        'label': args.label or os.path.basename(os.path.dirname(args.ckpt)),
        'ckpt': args.ckpt,
        'tokenizer': args.tokenizer,
        'vocab_size': V,
        # 原始（各自词表空间内，跨 vocab 不可比）
        'val_loss_nats_per_token': mean_nll,
        'val_loss_bits_per_token': mean_nll / LN2,
        # 跨 vocab 可比
        'bits_per_char': mean_nll / LN2 * (n_tok / n_chars),
        'bits_per_byte': mean_nll / LN2 * (n_tok / n_bytes),
        # 上下文
        'n_val_chars': n_chars,
        'n_val_bytes': n_bytes,
        'n_val_tokens': n_tok,
        'chars_per_token': n_chars / n_tok,
        'bytes_per_token': n_bytes / n_tok,
        'n_pred_tokens': total_cnt,
        'n_blocks_used': n_blocks,
        'block_size': args.block_size,
        'eval_seconds': dt,
        'model': {'n_layer': args.n_layer, 'n_head': args.n_head,
                  'n_embd': args.n_embd, 'ff_type': args.ff_type,
                  'position_encoding': args.position_encoding,
                  'tie_embeddings': args.tie_embeddings},
        'determinism': [r[0] for r in runs],
    }
    print('\n' + '=' * 78)
    print(f'{"label":<12} {"vocab":>6} {"val_loss":>10} {"bits/tok":>9} '
          f'{"bits/char":>10} {"bits/byte":>10} {"char/tok":>9}')
    print('-' * 78)
    print(f'{res["label"]:<12} {V:>6} {mean_nll:>10.4f} '
          f'{res["val_loss_bits_per_token"]:>9.4f} {res["bits_per_char"]:>10.4f} '
          f'{res["bits_per_byte"]:>10.4f} {res["chars_per_token"]:>9.4f}')
    print('=' * 78)
    print('※ 跨 vocab 比较只看 bits/char 与 bits/byte；val_loss 是各自词表空间的值。')

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f'结果已保存: {args.out}')


if __name__ == '__main__':
    main()
