"""生成采样工具：temperature + top_p (nucleus) + repetition_penalty。

server.py 与 generate.py 共用，避免两处逻辑漂移。
所有函数纯函数式、无状态、可单测。
"""
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F


def apply_repetition_penalty(logits: torch.Tensor,
                             penalty: float,
                             prev_ids: Sequence[int]) -> torch.Tensor:
    """对 prev_ids 中出现过的 token 施加重复惩罚（就地改 logits）。

    - penalty = 1.0 → 不惩罚
    - penalty > 1.0 → 削弱重复 token（logit>0 时除以 penalty，logit<0 时乘 penalty，
      与 transformers 的 RepetitionPenaltyLogitsProcessor 一致）
    - penalty < 1.0 → 反而鼓励重复（很少用）
    """
    if penalty == 1.0 or not prev_ids:
        return logits
    for tid in set(prev_ids):          # 去重：同一 token 只罚一次
        if tid >= logits.shape[-1]:    # 越界防御（理论上不会发生）
            continue
        lg = logits[..., tid]
        logits[..., tid] = torch.where(lg > 0, lg / penalty, lg * penalty)
    return logits


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus 截断：只保留累积概率达到 top_p 的最小 token 集（其余置 -inf）。

    与 transformers 的 TopPLogitsWarper 一致：按概率降序累计，累积超过 top_p
    之后的 token 全部移除；但越过阈值的那一个也保留（保证至少一个 token 可用）。
    """
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = probs.cumsum(dim=-1)
    # 先标出"累积概率已超过 top_p"的位置，再整体右移一位 → 越过阈值的那个不删
    remove = cumsum > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    # 把 remove 还原成原索引顺序，在原 logits 上置 -inf
    unsort = sorted_idx.argsort(dim=-1)
    removed_mask = remove.gather(-1, unsort)
    return logits.masked_fill(removed_mask, float('-inf'))


def sample(logits: torch.Tensor,
           temperature: float = 1.0,
           top_p: float = 1.0,
           repetition_penalty: float = 1.0,
           prev_ids: Optional[Sequence[int]] = None,
           rng: Optional[torch.Generator] = None) -> torch.Tensor:
    """从最后位置 logits (B, V) 采样下一个 token，返回 (B, 1)。

    - temperature <= 0 → 贪心 argmax（忽略 top_p / 重复惩罚）
    - 组合顺序：先重复惩罚 → 再 top_p 截断 → 温度缩放 → softmax → 采样
    """
    lg = logits[:, -1, :].clone()        # 只取最后位置；不污染调用方
    if temperature <= 0:
        return lg.argmax(dim=-1, keepdim=True)
    if repetition_penalty != 1.0 and prev_ids is not None:
        apply_repetition_penalty(lg, repetition_penalty, prev_ids)
    apply_top_p(lg, top_p)
    lg = lg / temperature
    probs = F.softmax(lg, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)
