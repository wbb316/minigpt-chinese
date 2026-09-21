"""共享自回归生成引擎：KV cache + 超窗重建，server 与 generate.py 共用。

核心思路：
- 首轮对整个 prompt 预填充一次，拿到每层的 (k, v) 缓存；
- 之后每步只把上一个新 token 喂给模型，增量解码（快 ~2x）；
- 位置编码只训练到 block_size：当缓存长度将超过 block_size + max_overrun 时，
  用最近 block_size 个 token（含刚生成的那个）重新预填充，丢弃更早的历史。
  这等价于"滑动窗口 + 重锚位置"，与训练时"永远只看 ≤block_size"的分布大致一致
  （窗口语义不变，只是**重锚频率**从每步一次降到每 max_overrun 步一次）。

★ 为什么要 max_overrun（2026-09-20 性能修复）：
  原实现的判据是 `cache_len + 1 > block_size` —— 缓存一满，**之后每一步**都要用
  block_size 个 token 整段重新 prefill。等于把增量解码退化回全量前向：
  20M/block_size=256 实测 未满窗 8.0 ms/token → 满窗 41.5 ms/token（5.2× 慢），
  「接着写」（上轮全文接回 prompt，必然 ≥block_size）因此不可用。
  现在允许缓存超出 block_size 至多 max_overrun 步再重建，单次重建成本被摊薄
  max_overrun 倍；`max_overrun=0` 时判据与旧实现**逐位一致**（回归保护）。

★ 扩表（必须配套，否则静默算错）：
  允许超窗后，增量步的 RoPE/sinusoidal 位置索引会到 block_size + max_overrun - 1，
  而 cos/sin（或 pe）表原本只到 block_size。**越界切片不会报错** —— T=1 时切片
  静默变空、位置编码被整个跳过（见 model/rope.py 顶部警告）。所以本模块在真正
  用超窗之前调用 ensure_pos_capacity() **就地把表建长**：
    · RoPE 的 cos/sin 是无参数确定性三角函数 → 扩表不引入任何新参数；
    · GPT.__init__ 把**同一个** RotaryEmbedding 对象共享给所有 block，
      所以就地改 gpt.rope 这一张表，所有层同时可见（下面有显式断言守住这点）。
  （generate_ids 拿到的是已建好的 gpt，无法改其构造参数 —— 就地扩展是唯一
   既不破坏 checkpoint 兼容、也不需要调用方改动的方式。）

⚠️ 已知取舍（2026-09-20/21 实测，选 max_overrun 时要看）：
  超窗期间注意力的**相对位置距离**会超过训练上限（训练只见 ≤ block_size-1）——
  到 block_size + max_overrun - 1，窗口里的 token 数也从 block_size 涨到
  block_size + max_overrun。这是 RoPE 的外推区间，质量会下降。

  150M/block=512、prompt 848 字、生成 150 token 实测（0.8/0.9 采样）的**质量**：
    · max_overrun=0   → 相对距离 ≤ 511（分布内），文本全程连贯；
    · max_overrun=32  → ≤ 543，文本仍连贯；
    · max_overrun=64  → ≤ 575，文本仍连贯；
    · max_overrun=128 → ≤ 639，约 60 token 后开始出现词沙拉；
    · max_overrun=256 → ≤ 767，约 60 token 后明显退化。

  50M/block=512、prompt 恰好 512、生成 40 token 实测的**速度**（3 次取中位）：
    · max_overrun=0   → 269.3 ms/token（悬崖）
    · max_overrun=16  →  50.4
    · max_overrun=32  →  40.0
    · max_overrun=64  →  35.7   ← 最快
    · max_overrun=128 →  36.9
    · max_overrun=256 →  39.1   ← 比 64 更慢！

  ★ 所以默认取 **64**：速度已到顶（再大反而略慢，因为单步增量才是主导成本，
    超窗只是摊薄偶发重建），而 128/256 会明显牺牲文本质量。
    「64 = block_size//8」——这是当前 block_size=512 下的甜点值。
"""
from typing import List, Optional

import torch

from model.sampling import sample


def ensure_pos_capacity(gpt, min_len: int) -> int:
    """**就地**确保 gpt 的位置编码表覆盖位置索引 [0, min_len)（幂等，只推理用）。

    - 不新建模块、不改任何参数、不动旧行数值（只在表尾续算若干行确定性三角函数），
      因此对已有输出逐位无影响；
    - 表长已 >= min_len 时什么都不做（`max_overrun=0` 路径根本不调用本函数）；
    - 返回扩后的表长。

    ⚠️ 副作用（有意，且是本方案的核心）：这是**就地修改调用方的模型**。
      cos/sin（或 pe）是**持久化 buffer** → 扩表后 `gpt.state_dict()` 里这两/一张表
      会变长，此后不要再对该 gpt 调 `load_state_dict(ckpt, strict=True)`
      （shape 不匹配会报错；报错是响亮的，不会静默）。同一进程内重复调用是幂等的，
      所以 server 每个请求都调也只会扩一次。

    为什么强调"就地"：GPT.__init__ 把同一个 RoPE 对象传给**所有** block
    （`rope=rope` → 每个 attn.rope 都是同一个引用），于是扩展 gpt.rope 一处即可
    全层生效。若哪天改成每层各持一份副本，下面的一致性断言会立刻报错而不是
    静默只扩一层（那会退化成"部分层位置编码越界"，正是最坏的一类静默失效）。
    """
    if getattr(gpt, 'position_encoding', 'rope') == 'rope':
        rope = getattr(gpt, 'rope', None)
        if rope is not None:
            for i, blk in enumerate(gpt.blocks):
                if blk.attn.rope is not rope:
                    raise RuntimeError(
                        f'第 {i} 层的 attn.rope 不是 gpt.rope 同一个对象，'
                        '就地扩表只会扩到部分层 → 拒绝继续（否则会静默算出垃圾）。')
            return rope.extend_to(min_len)
    return gpt.pos_emb.extend_to(min_len)


def generate_ids(gpt, prompt_ids: List[int],
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 repetition_penalty: float = 1.0,
                 rng: Optional[torch.Generator] = None,
                 device: Optional[torch.device] = None,
                 max_overrun: int = 64):
    """自回归续写，返回 (ctx_ids, new_ids)。

    - ctx_ids: prompt（截断到 block_size 后） + 新生成的全部 token id
    - new_ids: 仅新生成的 max_new_tokens 个 id
    - prev_ids 用于重复惩罚时始终给完整 ctx（含 prompt），与 HF 惯例一致

    max_overrun: KV cache 允许**超出 block_size** 的步数（默认 **64**，
        经实测选定的甜点值；必须是可配置的，不要依赖默认值本身）。重建判据是
        `cache_len + 1 > block_size + max_overrun`，重建窗口仍是最近 block_size 个
        token（窗口语义不变）。`max_overrun=0` → 判据退化为旧的
        `cache_len + 1 > block_size`，输出与修复前**逐位一致**。
        超窗期间位置索引会超过 block_size（最多到 block_size+max_overrun-1），
        故进入超窗前会 ensure_pos_capacity() 就地扩表；扩表只用确定性三角函数，
        不引入新参数。副作用：重锚时刻变化 → 同一 seed 的输出与修复前不同（已确认接受）。

        为什么默认 64 而不是更大：实测 50M/block512/prompt512/40token，
        64 → 35.7 ms/token 已是**最快**，128/256 反而略慢（36.9/39.1），
        因为主导成本是单步增量、超窗只摊薄偶发重建；而 128 起文本质量明显退化。
        详见本模块顶部「已知取舍」。
    """
    if max_overrun < 0:
        raise ValueError(f'max_overrun 不能为负，得到 {max_overrun}')
    if device is None:
        device = next(gpt.parameters()).device
    block = gpt.block_size
    limit = block + max_overrun          # 缓存有效长度上限（含）；位置索引 < limit
    if max_overrun > 0:
        # 扩表必须发生在任何超窗位置被用到之前（越界是静默的！）
        ensure_pos_capacity(gpt, limit)
    prompt_ids = list(prompt_ids[-block:])
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
            if cache_len + 1 > limit:
                # 缓存将超过 block_size + max_overrun：用最近 block_size 个 token 重建
                wx = torch.tensor([ctx[-block:]],
                                  dtype=torch.long, device=device)
                logits, kvs = gpt(wx, return_kv=True)
            else:
                logits, kvs = gpt(nid, past_kvs=kvs)
    return ctx, new_ids
