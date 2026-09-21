"""生成性能基准：量化"满窗后每步重建"的开销，验证 `max_overrun` 修复。

用法：
    python benchmark/benchmark_generation.py                    # 用 20M 存档（快）
    python benchmark/benchmark_generation.py --max-overrun 0    # ★ 修复前行为（对照）
    python benchmark/benchmark_generation.py --max-overrun 256  # 修复后（默认）
    python benchmark/benchmark_generation.py --ckpt <path> --tokenizer <path>

为什么需要它：`model/generation.py` 里只要"缓存一满就每步整段重建"，之后每一步都用
最近 block_size 个 token **整段重新 prefill**（而非增量）。这是"接着写"在长文本上
极慢的根因。修复后允许缓存超窗 max_overrun 步再重建，把单次重建成本摊薄
max_overrun 倍；`--max-overrun 0` 复现修复前行为（逐位一致），故本脚本可同时充当
"修复前/后"的同一把尺子。
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from model.generation import generate_ids  # noqa: E402
from model.gpt import GPT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CKPT = os.path.join(ROOT, 'result', '20M参数+416Mtokens', 'checkpoint_best.pt')
DEFAULT_TOK = os.path.join(ROOT, 'result', '20M参数+416Mtokens', 'tokenizer_best.pkl')


def load(ckpt, tok_path):
    import pickle

    from model.layers import swiglu_hidden
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k[len('_orig_mod.'):]: v for k, v in sd.items()}
    n_layer = max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.')) + 1
    n_embd = sd['token_emb.weight'].shape[1]
    vocab = sd['token_emb.weight'].shape[0]
    block = sd['pos_emb.pe'].shape[1]
    pe = 'rope' if 'rope.cos_cached' in sd else 'sinusoidal'
    if pe == 'rope':
        n_head = n_embd // int(sd['rope.cos_cached'].shape[-1])
    else:
        n_head = 8
    ff_type = 'swiglu' if any(k.endswith('gate_proj.weight') for k in sd) else 'relu'
    ff_hidden = swiglu_hidden(n_embd) if ff_type == 'swiglu' else None
    g = GPT(vocab_size=vocab, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            block_size=block, tie_embeddings=True, position_encoding=pe,
            ff_type=ff_type, ff_hidden=ff_hidden)
    g.load_state_dict(sd, strict=False)
    g.to('cpu').eval()
    with open(tok_path, 'rb') as f:
        tk = pickle.load(f)
    return g, tk


def timed_generate(g, prompt_ids, n_new, max_overrun=256):
    """返回 (总耗时秒, 每步耗时列表)。每步含 1 次前向（或重建）+ 采样。"""
    times = []
    import model.generation as gen
    orig = gen.sample

    def timed_sample(*a, **kw):
        t0 = time.perf_counter()
        r = orig(*a, **kw)
        times.append(time.perf_counter() - t0)
        return r

    gen.sample = timed_sample
    try:
        t0 = time.perf_counter()
        generate_ids(g, prompt_ids, max_new_tokens=n_new, temperature=0.8, top_p=0.9,
                     max_overrun=max_overrun)
        total = time.perf_counter() - t0
    finally:
        gen.sample = orig
    return total, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--tokenizer', default=DEFAULT_TOK)
    ap.add_argument('--new-tokens', type=int, default=120)
    ap.add_argument('--max-overrun', type=int, default=256,
                    help='KV cache 允许超出 block_size 的步数；0 = 修复前行为（对照用）')
    args = ap.parse_args()

    g, tk = load(args.ckpt, args.tokenizer)
    bs = g.block_size
    n_layer = len(g.blocks)
    n_embd = g.token_emb.weight.shape[1]
    print(f'模型: {n_layer}L/{n_embd}d, block_size={bs}, '
          f'位置编码={g.position_encoding}, 参数量={g.get_num_params():,}')
    print(f'每档生成 {args.new_tokens} token, max_overrun={args.max_overrun} '
          f'(缓存上限 {bs + args.max_overrun})\n')

    base = tk.encode('那天下着细雨，她推开旧书店的门。')
    print(f'{"prompt token":>12s} {"总耗时":>9s} {"ms/token":>9s} {"中位步":>8s} {"最慢步":>8s} {"是否满窗":>9s}')
    print('-' * 66)
    rows = []
    for plen_target in (16, 64, 128, 256, 384, 448, 480, 512):
        # 用重复文本凑到目标 token 数
        p = list(base)
        while len(p) < plen_target:
            p += base
        p = p[:plen_target]
        total, times = timed_generate(g, p, args.new_tokens,
                                      max_overrun=args.max_overrun)
        per = total / args.new_tokens * 1000
        med = statistics.median(times) * 1000 if times else 0
        mx = max(times) * 1000 if times else 0
        over = '是' if len(p) >= bs else '否'
        print(f'{len(p):>12d} {total:>8.2f}s {per:>9.1f} {med:>8.1f} {mx:>8.1f} {over:>9s}')
        rows.append((len(p), per, over))

    print()
    print('解读：max_overrun=0 时 prompt 达到 block_size 后每步都要整段重建 → ms/token 陡增；')
    print('      max_overrun>0 时重建降频为每 max_overrun 步一次 → 满窗档应回落到接近未满窗。')
    short = [r for r in rows if r[2] == '否']
    full = [r for r in rows if r[2] == '是']
    if short and full:
        a = statistics.median([r[1] for r in short])
        b = statistics.median([r[1] for r in full])
        print(f'  未满窗中位 ≈ {a:.1f} ms/token；满窗中位 ≈ {b:.1f} ms/token → '
              f'满窗比未满窗慢 {b / a:.1f}×')
        if args.max_overrun > 0:
            print(f'  （max_overrun=0 时该比值即"修复前"的性能悬崖；'
                  f'本档 max_overrun={args.max_overrun}）')


if __name__ == '__main__':
    main()
