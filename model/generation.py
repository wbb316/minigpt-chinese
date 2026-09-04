"""共享自回归生成引擎：KV cache + 超窗重建，server 与 generate.py 共用。

核心思路：
- 首轮对整个 prompt 预填充一次，拿到每层的 (k, v) 缓存；
- 之后每步只把上一个新 token 喂给模型，增量解码（快 ~2x）；
- 但 PositionalEncoding 只训练到 block_size：当缓存长度将超过 block_size 时，
  用最近 block_size 个 token（含刚生成的那个）重新预填充，丢弃更早的历史。
  这等价于"滑动窗口 + 重锚位置"，不会越界，也与训练时"永远只看 ≤block_size"
  的分布一致。
"""
from typing import List, Optional

import torch

from model.sampling import sample


def generate_ids(gpt, prompt_ids: List[int],
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 repetition_penalty: float = 1.0,
                 rng: Optional[torch.Generator] = None,
                 device: Optional[torch.device] = None):
    """自回归续写，返回 (ctx_ids, new_ids)。

    - ctx_ids: prompt（截断到 block_size 后） + 新生成的全部 token id
    - new_ids: 仅新生成的 max_new_tokens 个 id
    - prev_ids 用于重复惩罚时始终给完整 ctx（含 prompt），与 HF 惯例一致
    """
    if device is None:
        device = next(gpt.parameters()).device
    prompt_ids = list(prompt_ids[-gpt.block_size:])
    ctx: List[int] = list(prompt_ids)

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_ids: List[int] = []
    with torch.no_grad():
        logits, kvs = gpt(x, return_kv=True)
        for _ in range(max_new_tokens):
            nid = sample(logits, temperature=temperature, top_p=top_p,
                         repetition_penalty=repetition_penalty,
                         prev_ids=ctx, rng=rng)
            new_ids.append(int(nid.item()))
            ctx.append(new_ids[-1])

            cache_len = kvs[0][0].size(2)
            if cache_len + 1 > gpt.block_size:
                # 缓存将超窗：用最近 block_size 个 token 重建
                wx = torch.tensor([ctx[-gpt.block_size:]],
                                  dtype=torch.long, device=device)
                logits, kvs = gpt(wx, return_kv=True)
            else:
                logits, kvs = gpt(nid, past_kvs=kvs)
    return ctx, new_ids
