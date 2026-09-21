"""生成测试 + KV cache 推理加速验证

加载训练好的最佳模型 (result/checkpoint_best.pt + result/tokenizer_best.pkl)，
提供两种生成方式并验证它们的结果完全一致、KV cache 更快：

- 原版 (use_cache=False): 每步把整个序列重新前向一遍（旧 generate 的方式）
- KV cache 版 (use_cache=True): 每步只喂最后一个 token，复用之前算好的 K/V

用法:
    python generate.py               # 验证一致性 + 计时对比 + 示例生成
    python generate.py --demo        # 只跑示例生成（用 KV cache）

⚠️ max_overrun（2026-09-20 性能修复，本文件第二份生成循环同步修复）：
    旧实现的 cache 版一满窗就**每步**整段重建（20M/block256 实测 8.0 → 41.5 ms/token）。
    `max_overrun` 允许缓存超窗若干步再重建，把重建成本摊薄。默认 **64**
（实测甜点：50M/block512/prompt512 下 64 → 35.7 ms/token 最快，128/256 反而略慢且文本质量退化）。
    · `--max-overrun 0` = 修复前行为（逐位一致），可与默认值对比加速比；
    · 超窗期间位置索引超过 block_size，位置表由 ensure_pos_capacity() 就地扩长
      （确定性三角函数，无新参数）；
    · 代价：重锚时刻变化 → **同一 seed 下 cache 版文本可能与"原版"分支不同**
      （当总长度超过 block_size 时；默认小 prompt 跑不到窗口上限，仍然一致）。
      这是已确认接受的取舍，不是 bug。
"""
import argparse
import os
import pickle
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.gpt import GPT  # noqa: E402
from model.sampling import sample  # noqa: E402
from model.generation import generate_ids, ensure_pos_capacity  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------- 模型加载
def load_model(ckpt_path='result/35M参数+998Mtokens/checkpoint_best.pt',
               tok_path='result/35M参数+998Mtokens/tokenizer_best.pkl', n_head=8):
    """加载模型，架构从 state_dict 自动推断（头数默认 8，可用 --n-head 覆盖）。

    默认加载 v2 35M（webnovel_v2 语料，2B token，val 3.6372）；其它模型传 --ckpt。
    """
    ckpt = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(ROOT, ckpt_path)
    tok = tok_path if os.path.isabs(tok_path) else os.path.join(ROOT, tok_path)

    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    # torch.compile 训练的 checkpoint 带 '_orig_mod.' 前缀（OptimizedModule 痕迹），
    # 剥离后所有解析逻辑按无前缀 key 统一处理（旧存档本来无前缀，兼容）
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k[len('_orig_mod.'):]: v for k, v in sd.items()}
    n_layer = max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.')) + 1
    n_embd = sd['token_emb.weight'].shape[1]
    vocab_size = sd['token_emb.weight'].shape[0]
    block_size = sd['pos_emb.pe'].shape[1]
    # 自动检测位置编码：state_dict 含 rope buffer → rope；否则 sinusoidal（旧模型）
    pe = 'rope' if 'rope.cos_cached' in sd else 'sinusoidal'
    # 自动检测 FFN：含 gate_proj → swiglu（v3 FFN 变体）；否则 relu/gelu 的 fc1/fc2
    # （relu 与 gelu 参数名相同、无法从权重区分 → 默认按 relu 建，输出等价）
    ff_type = 'swiglu' if any(k.endswith('gate_proj.weight') for k in sd) else 'relu'
    ff_hidden = sd['blocks.0.ff.gate_proj.weight'].shape[0] if ff_type == 'swiglu' else None

    gpt = GPT(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
              n_embd=n_embd, block_size=block_size,
              position_encoding=pe, ff_type=ff_type, ff_hidden=ff_hidden)
    gpt.load_state_dict(sd)
    gpt.to(device).eval()

    with open(tok, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f'模型: {n_layer}层/{n_head}头/{n_embd}维, 词表{vocab_size}, '
          f'block_size={block_size}, 位置编码: {pe}, FFN: {ff_type}'
          + (f'(h={ff_hidden})' if ff_hidden else ''))
    return gpt, tokenizer


# ---------------------------------------------------------------- 生成
def _sample(logits, temperature, rng, top_p=1.0, repetition_penalty=1.0,
            prev_ids=None):
    """从 logits 采样一个 token；temperature<=0 时贪心(argmax)。返回 (B,1) 张量。"""
    return sample(logits, temperature=temperature, top_p=top_p,
                  repetition_penalty=repetition_penalty,
                  prev_ids=prev_ids, rng=rng)


def generate(gpt, tokenizer, prompt, max_new_tokens=50, temperature=1.0,
             top_p=1.0, repetition_penalty=1.0,
             use_cache=False, seed=0, max_overrun=64):
    """续写。use_cache=True 用 KV cache 加速（预填充后每步只算新 token）。

    max_overrun: cache 版允许超窗的步数（见模块 docstring）；0 = 修复前行为。
    """
    ids = tokenizer.encode(prompt)
    idx = torch.tensor(ids[-gpt.block_size:], device=device).unsqueeze(0)
    rng = torch.Generator(device=device).manual_seed(seed)
    ctx = ids[-gpt.block_size:]          # 完整可见上下文（重复惩罚用）
    all_ids = list(ctx)

    if use_cache:
        # 超窗重锚前先把位置表建长（越界是静默的，必须提前扩）
        if max_overrun < 0:
            raise ValueError(f'max_overrun 不能为负，得到 {max_overrun}')
        limit = gpt.block_size + max_overrun
        if max_overrun > 0:
            ensure_pos_capacity(gpt, limit)
        # 预填充：完整前向一次，拿到所有层的 KV 缓存 + 全部位置的 logits
        logits, kvs = gpt(idx, return_kv=True)
    else:
        logits = gpt(idx)

    cur = idx
    for _ in range(max_new_tokens):
        next_token = _sample(logits, temperature, rng,
                             top_p=top_p, repetition_penalty=repetition_penalty,
                             prev_ids=all_ids)   # 取最后一个位置采样
        nid = int(next_token.item())
        all_ids.append(nid)

        if use_cache:
            # 下步：只喂刚生成的 token，复用缓存；缓存将超过 block_size+max_overrun 则重建
            if kvs[0][0].size(2) + 1 > limit:
                wx = torch.tensor([all_ids[-gpt.block_size:]],
                                  device=device)
                logits, kvs = gpt(wx, return_kv=True)
            else:
                logits, kvs = gpt(next_token, past_kvs=kvs)
        else:
            cur = torch.cat([cur, next_token], dim=1)
            cur = cur[:, -gpt.block_size:]          # 超窗截断（与训练一致）
            logits = gpt(cur)

    return tokenizer.decode(idx[0].tolist() + all_ids[len(ctx):])


# ---------------------------------------------------------------- 验证
def verify_identical(gpt, tokenizer, prompt, n_steps=30):
    """验证 KV cache 版与原版：逐步对比同一全局位置的 logits 是否一致。"""
    ids = tokenizer.encode(prompt)
    idx = torch.tensor(ids[-gpt.block_size:], device=device).unsqueeze(0)

    max_diff = 0.0
    with torch.no_grad():
        # 原版：完整序列前向（取最后一个位置）
        lg_full = gpt(idx)[:, -1, :]
        # cache 版：预填充，同样取最后一个位置
        lg_prefill, kvs = gpt(idx, return_kv=True)
        lg_cache = lg_prefill[:, -1, :]

        cur = idx
        for step in range(n_steps):
            diff = (lg_full - lg_cache).abs().max().item()
            max_diff = max(max_diff, diff)
            assert diff < 1e-4, \
                f'step {step}: logits 不一致, max diff={diff:.3e}'

            # 用贪心同步推进两种方式到下一个位置
            nxt = lg_full.argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
            lg_full = gpt(cur)[:, -1, :]
            lg_cache, kvs = gpt(nxt, past_kvs=kvs)
            lg_cache = lg_cache[:, -1, :]

    print(f'✓ 逐位置 logits 最大差异: {max_diff:.3e}（<1e-4，KV cache 与原版一致）')


def benchmark(gpt, tokenizer, prompt, max_new_tokens=100, seed=42,
              max_overrun=64):
    """计时对比：同一 seed 下原版 vs KV cache，各生成 max_new_tokens 个 token。

    max_overrun 只影响 cache 版（原版每步重算整段，与它无关）。传 0 即修复前行为，
    可用于跑出"修复前后"对比表。
    """
    # 预跑一次热身（排除加载/缓存开销；顺带把位置表扩好，不计入计时）
    generate(gpt, tokenizer, prompt, max_new_tokens=5, use_cache=True, seed=seed,
             max_overrun=max_overrun)
    generate(gpt, tokenizer, prompt, max_new_tokens=5, use_cache=False, seed=seed)

    t0 = time.perf_counter()
    text_old = generate(gpt, tokenizer, prompt, max_new_tokens=max_new_tokens,
                        use_cache=False, seed=seed)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    text_new = generate(gpt, tokenizer, prompt, max_new_tokens=max_new_tokens,
                        use_cache=True, seed=seed, max_overrun=max_overrun)
    t_new = time.perf_counter() - t0

    print(f'\n===== 计时对比 (生成 {max_new_tokens} 个 token, '
          f'max_overrun={max_overrun}) =====')
    print(f'原版(整序列重算):  {t_old:.3f}s')
    print(f'KV cache 版:       {t_new:.3f}s')
    print(f'加速比:            {t_old / t_new:.2f}x')
    if text_old == text_new:
        print('采样结果一致: ✓ 相同文本')
    else:
        # 只有总长超过 block_size 时才会走到这里：max_overrun>0 改变了重锚时刻，
        # 上下文锚定不同 → 轨迹分叉，属预期（用户已接受的取舍）
        print(f'采样结果一致: ✗ 文本不同（max_overrun={max_overrun}；'
              '总长超过 block_size 时重锚时刻变化导致轨迹分叉，属预期）'
              if max_overrun > 0 else '采样结果一致: ✗ 文本不同（不应发生！）')
    return text_old, text_new


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description='生成 + KV cache 加速验证')
    parser.add_argument('--demo', action='store_true', help='只跑示例生成（KV cache）')
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='采样温度（<=0 贪心）')
    parser.add_argument('--top-p', type=float, default=1.0,
                        help='nucleus 采样阈值（1.0 = 不截断）')
    parser.add_argument('--repetition-penalty', type=float, default=1.0,
                        help='重复惩罚（1.0 = 不惩罚；1.1~1.2 抑制复读）')
    parser.add_argument('--ckpt', default='result/35M参数+998Mtokens/checkpoint_best.pt',
                        help='模型权重路径')
    parser.add_argument('--tokenizer', default='result/35M参数+998Mtokens/tokenizer_best.pkl',
                        help='分词器 pkl 路径')
    parser.add_argument('--max-overrun', type=int, default=64,
                        help='KV cache 允许超出 block_size 的步数（超窗重建降频）；'
                             '0 = 修复前行为，用于跑前后对比')
    args = parser.parse_args()

    gpt, tokenizer = load_model(ckpt_path=args.ckpt, tok_path=args.tokenizer,
                                n_head=args.n_head)

    kw = dict(top_p=args.top_p, repetition_penalty=args.repetition_penalty)

    prompts = ["真唯同学", "我喜欢你"]
    if args.demo:
        for p in prompts:
            out = generate(gpt, tokenizer, p, max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, use_cache=True,
                           max_overrun=args.max_overrun, **kw)
            print(f'\n【{p}】→\n{out}')
        return

    # 1) 一致性验证
    print('\n---- 一致性验证 ----')
    verify_identical(gpt, tokenizer, prompts[0], n_steps=30)

    # 2) 计时对比（同时再次验证同 seed 文本一致）
    print('\n---- 计时对比 ----')
    text_old, text_new = benchmark(gpt, tokenizer, prompts[0],
                                   max_new_tokens=args.max_new_tokens,
                                   max_overrun=args.max_overrun)

    # 3) 示例生成
    print('\n---- 示例生成 (KV cache) ----')
    for p in prompts:
        out = generate(gpt, tokenizer, p, max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, use_cache=True,
                       max_overrun=args.max_overrun, **kw)
        print(f'\n【{p}】→\n{out}')


if __name__ == '__main__':
    main()
