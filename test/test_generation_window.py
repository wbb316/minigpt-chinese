# -*- coding: utf-8 -*-
"""自回归生成的"滑动窗口 + 超窗重建"语义不变量测试（2026-09-20 性能修复）。

背景：修复前 cache 一满窗就**每步**整段重建（20M/block256 实测 8.0 → 41.5 ms/token，
5.2× 慢）。修复引入 `max_overrun`：缓存允许超出 block_size 至多 max_overrun 步再重建。
允许超窗的前提是**位置编码表必须够长** —— 而 model/rope.py 的切片越界是**静默**的
（T=1 增量解码时切片变空 → 位置编码被整个跳过，输出"看着正常"却是垃圾）。
所以本文件的核心是三条护栏：

1. 缓存有效长度 <= block_size + max_overrun（不越界）
2. ★ 位置索引 start + T <= 位置表长（防上面那个静默失效）—— 这条最重要
3. max_overrun=0 与修复前实现**逐位一致**（回归保护，证明默认语义没被改坏）

外加：prompt 恰好 = block_size-1 / block_size / block_size+1 的边界、token 计数、
RoPE 表"只能变长、旧行不变"、以及把"越界静默失效"这个坑本身钉成测试。
"""
import inspect

import pytest
import torch

from model.gpt import GPT
from model.generation import ensure_pos_capacity, generate_ids
from model.rope import RotaryEmbedding
from model.sampling import sample

VOCAB, N_LAYER, N_HEAD, N_EMBD, BLOCK = 64, 2, 2, 32, 8


def make_gpt(pe='rope', **kw):
    """极小模型（block_size=8）→ 几十 token 就能完整走完"超窗→重建"多轮循环。"""
    torch.manual_seed(0)
    return GPT(vocab_size=VOCAB, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
               block_size=BLOCK, position_encoding=pe, ff_type='relu',
               **kw).eval()


def fresh_rng(seed=1234):
    return torch.Generator(device='cpu').manual_seed(seed)


def pos_table_len(gpt):
    """当前实际生效的位置编码表长（rope → cos/sin；sinusoidal → pe）。"""
    if gpt.position_encoding == 'rope':
        return gpt.rope.cos_cached.size(2)
    return gpt.pos_emb.pe.size(1)


class GenProbe:
    """记录 (a) 每次前向后 KV 缓存的有效长度；(b) 每次位置编码调用的 (start, T, 表长)。

    rope 模式挂 gpt.rope.forward；sinusoidal 模式挂 gpt.pos_emb.forward。
    GPT.__init__ 把同一个 rope 对象共享给所有 block，所以挂在 gpt.rope 上能捕获全部层。
    """

    def __init__(self, gpt):
        self.gpt = gpt
        self.cache_lens = []
        self.pos_calls = []
        self._forward = gpt.forward
        self._rope = getattr(gpt, 'rope', None)
        self._rope_forward = self._rope.forward if self._rope is not None else None
        self._pe = gpt.pos_emb
        self._pe_forward = self._pe.forward

    def __enter__(self):
        orig = self._forward

        def fwd(x, past_kvs=None, return_kv=False):
            out = orig(x, past_kvs=past_kvs, return_kv=return_kv)
            kvs = out[1] if isinstance(out, tuple) else None
            if kvs is not None:
                self.cache_lens.append(kvs[0][0].size(2))
            return out
        self.gpt.forward = fwd

        if self._rope is not None:
            rope, orf = self._rope, self._rope_forward

            def rf(q, k, start=0):
                self.pos_calls.append((start, q.size(2), rope.cos_cached.size(2)))
                return orf(q, k, start=start)
            rope.forward = rf
        else:
            pe, opf = self._pe, self._pe_forward

            def pf(x, start=0):
                self.pos_calls.append((start, x.size(1), pe.pe.size(1)))
                return opf(x, start=start)
            pe.forward = pf
        return self

    def __exit__(self, *exc):
        del self.gpt.forward
        if self._rope is not None:
            del self._rope.forward
        else:
            del self._pe.forward
        return False


def assert_invariants(probe, block, max_overrun):
    """★ 三条核心不变量（越界在这里必须变成断言失败，而不是静默算错）。"""
    limit = block + max_overrun
    assert probe.cache_lens, '探针没记录到任何前向'
    assert max(probe.cache_lens) <= limit, (
        f'缓存有效长度 {max(probe.cache_lens)} 超过上限 block_size+max_overrun={limit}')
    assert min(probe.cache_lens) >= 1
    assert probe.pos_calls, '探针没记录到任何位置编码调用'
    for start, T, table_len in probe.pos_calls:
        assert start >= 0
        assert start + T <= table_len, (
            f'位置越界：start={start}, T={T} 需要表长 {start + T}，实际 {table_len} '
            '—— 这正是 rope.py 会静默失效的场景')


# --------------------------------------------------------------- 核心不变量
@pytest.mark.parametrize('pe', ['rope', 'sinusoidal'])
def test_window_and_pos_index_invariants(pe):
    """超窗路径真的被走到：缓存长到 block+max_overrun 后重建，且位置索引全程 < 表长。"""
    gpt = make_gpt(pe)
    overrun = 4
    prompt = [1, 2, 3]
    with GenProbe(gpt) as probe:
        ctx, new_ids = generate_ids(gpt, prompt, max_new_tokens=20,
                                    max_overrun=overrun, rng=fresh_rng())

    assert_invariants(probe, BLOCK, overrun)
    limit = BLOCK + overrun
    # 修复确实生效：缓存用满了超窗额度、位置索引真的越过了 block_size
    assert max(probe.cache_lens) == limit, probe.cache_lens
    assert max(s for s, _, _ in probe.pos_calls) >= BLOCK, \
        f'位置索引从未越过 block_size → 本用例没走到被修复的路径: {probe.pos_calls}'
    # 至少发生过一次重建（缓存长度回落）
    assert any(b < a for a, b in zip(probe.cache_lens, probe.cache_lens[1:])), \
        f'没观测到重建: {probe.cache_lens}'
    # 表长必须覆盖到 limit（末位合法索引 = limit-1）
    assert pos_table_len(gpt) == limit
    assert len(new_ids) == 20


@pytest.mark.parametrize('pe', ['rope', 'sinusoidal'])
def test_max_overrun_zero_keeps_old_window(pe):
    """max_overrun=0：缓存严格不超 block_size，表长原样不动（旧行为 + 旧内存占用）。"""
    gpt = make_gpt(pe)
    with GenProbe(gpt) as probe:
        generate_ids(gpt, [1, 2, 3], max_new_tokens=20, max_overrun=0,
                     rng=fresh_rng())
    assert_invariants(probe, BLOCK, 0)
    assert max(probe.cache_lens) == BLOCK, probe.cache_lens
    assert pos_table_len(gpt) == BLOCK, 'max_overrun=0 不应扩表（必须与修复前逐位一致）'


def test_default_max_overrun_is_64():
    """默认值 = 64，是**实测选定的甜点**，别被悄悄改掉。

    为什么是 64 而不是更大（50M/block512/prompt512/40token，3 次取中位实测）：
        0 → 269.3 ms/token（悬崖）｜16 → 50.4｜32 → 40.0
        64 → **35.7（最快）**｜128 → 36.9｜256 → 39.1（反而更慢）
    且 128 起文本质量明显退化（60 token 后出现词沙拉）。
    主导成本是单步增量，超窗只摊薄偶发重建 → 超过 64 无收益却损质量。
    """
    sig = inspect.signature(generate_ids)
    assert sig.parameters['max_overrun'].default == 64


def test_negative_max_overrun_raises():
    gpt = make_gpt('rope')
    with pytest.raises(ValueError, match='max_overrun'):
        generate_ids(gpt, [1, 2], max_new_tokens=2, max_overrun=-1)


# --------------------------------------------------------------- 边界长度
@pytest.mark.parametrize('plen', [1, BLOCK - 1, BLOCK, BLOCK + 1, BLOCK + 5])
def test_prompt_boundaries(plen):
    """prompt 恰好在窗口边界（block-1 / block / block+1）时：不崩、计数正确、上下文语义不变。"""
    gpt = make_gpt('rope')
    prompt = [(i % VOCAB) for i in range(plen)]
    n = 15
    overrun = 4
    with GenProbe(gpt) as probe:
        ctx, new_ids = generate_ids(gpt, prompt, max_new_tokens=n,
                                    max_overrun=overrun, rng=fresh_rng())
    assert_invariants(probe, BLOCK, overrun)
    assert pos_table_len(gpt) == BLOCK + overrun
    # token 计数与上下文拼接语义（prompt 截断到最近 block_size 个，与修复前一致）
    assert len(new_ids) == n
    assert ctx == list(prompt[-BLOCK:]) + new_ids
    assert all(0 <= i < VOCAB for i in new_ids)


def test_generated_token_count_matches_request():
    gpt = make_gpt('rope')
    for n in (1, 3, 40):
        _, new_ids = generate_ids(gpt, [5, 6, 7], max_new_tokens=n,
                                  max_overrun=4, rng=fresh_rng(n))
        assert len(new_ids) == n


def test_same_seed_same_output():
    gpt = make_gpt('rope')
    a = generate_ids(gpt, [1, 2, 3], max_new_tokens=12, max_overrun=4,
                     rng=fresh_rng(7))
    b = generate_ids(gpt, [1, 2, 3], max_new_tokens=12, max_overrun=4,
                     rng=fresh_rng(7))
    assert a[1] == b[1]


# --------------------------------------------------------------- 回归保护：逐位一致
def legacy_generate_ids(gpt, prompt_ids, max_new_tokens, temperature=1.0,
                        top_p=1.0, repetition_penalty=1.0, rng=None, device=None):
    """修复前 model/generation.py（git HEAD 版本）的逐字复刻，仅用于回归对照。

    判据 `cache_len + 1 > gpt.block_size`，重建窗口 = 最近 block_size 个 token。
    """
    if device is None:
        device = next(gpt.parameters()).device
    prompt_ids = list(prompt_ids[-gpt.block_size:])
    ctx = list(prompt_ids)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_ids = []
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
                wx = torch.tensor([ctx[-gpt.block_size:]], dtype=torch.long,
                                  device=device)
                logits, kvs = gpt(wx, return_kv=True)
            else:
                logits, kvs = gpt(nid, past_kvs=kvs)
    return ctx, new_ids


@pytest.mark.parametrize('plen', [BLOCK - 2, BLOCK - 1, BLOCK, BLOCK + 1, BLOCK + 3])
@pytest.mark.parametrize('pe', ['rope', 'sinusoidal'])
def test_max_overrun_zero_bit_identical_to_legacy(pe, plen):
    """★ 回归保护：max_overrun=0 与修复前实现**逐位一致**（同权重 / 同 seed）。"""
    gpt = make_gpt(pe)
    prompt = [(i * 3 + 1) % VOCAB for i in range(plen)]
    n = 14
    ctx_old, ids_old = legacy_generate_ids(gpt, prompt, n, temperature=0.9,
                                           top_p=0.95, repetition_penalty=1.1,
                                           rng=fresh_rng(99))
    ctx_new, ids_new = generate_ids(gpt, prompt, n, temperature=0.9, top_p=0.95,
                                    repetition_penalty=1.1, max_overrun=0,
                                    rng=fresh_rng(99))
    assert ids_new == ids_old, f'(pe={pe}, plen={plen}) 新 token 序列不一致'
    assert ctx_new == ctx_old, f'(pe={pe}, plen={plen}) 上下文不一致'
    assert pos_table_len(gpt) == BLOCK, 'max_overrun=0 路径不该动位置表'


# --------------------------------------------------------------- 扩表机制
def test_gpt_rope_extra_len_default_unchanged():
    """GPT 默认 rope_extra_len=0 → 表长与今天完全一致（训练侧 train.py 不传即不变）。"""
    g0 = make_gpt('rope')
    g1 = make_gpt('rope', rope_extra_len=4)
    assert g0.rope.cos_cached.size(2) == BLOCK
    assert g0.pos_emb.pe.size(1) == BLOCK
    assert g1.rope.cos_cached.size(2) == BLOCK + 4
    # 加长只是"追加"：已存在位置的数值逐位不变（不重算、无新参数）
    assert torch.equal(g1.rope.cos_cached[:, :, :BLOCK], g0.rope.cos_cached)
    assert torch.equal(g1.rope.sin_cached[:, :, :BLOCK], g0.rope.sin_cached)
    # 参数量不变（cos/sin 是 buffer，不是 Parameter）
    assert g1.get_num_params() == g0.get_num_params()
    assert g1.block_size == g0.block_size


def test_gpt_sinusoidal_extra_len_and_extend_equivalence():
    """sinusoidal 表同样可扩；就地扩表结果 == 一次性建同样长的表。"""
    g_short = make_gpt('sinusoidal')
    g_long = make_gpt('sinusoidal', rope_extra_len=4)
    g_short.pos_emb.extend_to(BLOCK + 4)
    assert torch.equal(g_short.pos_emb.pe, g_long.pos_emb.pe)


def test_extend_to_is_idempotent_and_prefix_preserving():
    rope = RotaryEmbedding(8, max_seq_len=16)
    before = rope.cos_cached.clone()
    assert rope.extend_to(16) == 16          # 不大于当前表长 → 什么都不做
    assert torch.equal(rope.cos_cached, before)
    assert rope.extend_to(24) == 24
    assert rope.cos_cached.size(2) == 24
    assert torch.equal(rope.cos_cached[:, :, :16], before)   # 旧行原样保留
    # 续算行 == 直接按 24 建表（同一公式、同 dtype）——用 equal 卡死，允许 0 误差
    ref = RotaryEmbedding(8, max_seq_len=24)
    assert torch.equal(rope.cos_cached, ref.cos_cached)
    assert torch.equal(rope.sin_cached, ref.sin_cached)
    assert rope.extra_len == 24 - 16


def test_prebuilt_long_table_equals_inplace_extension():
    """★ 生成时"就地扩表" 与 "生成前就建好长表" 结果完全一致（扩表数值无差异）。

    这是"扩表方案没有引入任何计算差异"的直接证据：两份模型逐位相同，唯一区别是
    位置表在**何时**被加长（生成循环内 vs 生成之前），同 seed 生成必须逐 token 相同。
    """
    g_short = make_gpt('rope')                      # 表长 8，由 generate_ids 就地扩到 12
    g_long = make_gpt('rope')
    g_long.load_state_dict(g_short.state_dict())    # 先完全对齐（含 8 长的表）
    g_long.rope.extend_to(BLOCK + 4)                # 生成前就把表建到 12
    assert g_short.rope.cos_cached.size(2) == BLOCK
    assert g_long.rope.cos_cached.size(2) == BLOCK + 4
    a = generate_ids(g_short, [1, 2, 3], 20, max_overrun=4, rng=fresh_rng(5))
    b = generate_ids(g_long, [1, 2, 3], 20, max_overrun=4, rng=fresh_rng(5))
    assert g_short.rope.cos_cached.size(2) == g_long.rope.cos_cached.size(2)
    assert torch.equal(g_short.rope.cos_cached, g_long.rope.cos_cached)
    assert a[1] == b[1], '就地扩表与预建长表的生成结果不一致'
    assert a[0] == b[0]


def test_rope_extra_len_is_checkpoint_incompatible():
    """rope_extra_len>0 建的模型**不能** strict load 旧 checkpoint（size mismatch）。

    钉住这个已知边界（响亮报错、不静默），也解释为什么默认值必须是 0：
    训练侧/加载侧要保兼容，推理需要长表时用 ensure_pos_capacity() 在 load 之后扩。
    """
    g0 = make_gpt('rope')
    g1 = make_gpt('rope', rope_extra_len=4)
    with pytest.raises(RuntimeError, match='size mismatch'):
        g1.load_state_dict(g0.state_dict())


def test_extend_to_updates_param_count():
    rope = RotaryEmbedding(8, max_seq_len=8)
    assert list(rope.parameters()) == []
    rope.extend_to(32)
    assert list(rope.parameters()) == []      # ★ 扩表不引入任何参数
    assert rope.cos_cached.size(2) == 32


def test_rope_object_shared_across_blocks_after_extend():
    """GPT 把所有 block 指向同一个 rope 对象 → 就地扩表全层同时生效。"""
    gpt = make_gpt('rope')
    assert all(blk.attn.rope is gpt.rope for blk in gpt.blocks)
    ensure_pos_capacity(gpt, BLOCK + 6)
    assert all(blk.attn.rope.cos_cached.size(2) == BLOCK + 6 for blk in gpt.blocks)
    assert torch.equal(gpt.blocks[-1].attn.rope.cos_cached, gpt.rope.cos_cached)


def test_ensure_pos_capacity_on_non_rope_uses_pos_emb():
    gpt = make_gpt('sinusoidal')
    assert ensure_pos_capacity(gpt, BLOCK + 3) == BLOCK + 3
    assert gpt.pos_emb.pe.size(1) == BLOCK + 3


# --------------------------------------------------------------- 钉住"静默失效"这个坑
def test_rope_slice_beyond_table_is_silently_empty():
    """越界切片**不报错**、T=1 时长度为 0（位置编码被整个跳过）—— 这就是必须扩表的原因。

    数值上：q * cos 里 cos 在位置维长度 0，广播后结果位置维也是 0，随后
    「旋转」实际等于什么都没做；修复前这条路完全没有响声。
    """
    rope = RotaryEmbedding(8, max_seq_len=16)
    assert rope.cos_cached[:, :, 16:17].numel() == 0     # 越界 → 静默变空
    assert rope.cos_cached[:, :, 15:16].numel() == 8     # 边界内正常


def test_rope_beyond_table_raises_loudly():
    """加了显式范围检查：越界必须报错（带 start/T/表长），而不是静默跳过位置编码。"""
    rope = RotaryEmbedding(8, max_seq_len=16)
    q = torch.randn(1, 1, 1, 8)
    rope(q, q, start=15)                     # 边界内（start+T = 16 == 表长）不报错
    with pytest.raises(IndexError, match='RoPE 位置越界'):
        rope(q, q, start=16)
    with pytest.raises(IndexError, match='RoPE 位置越界'):
        rope(torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8), start=14)


def test_sinusoidal_beyond_table_raises_loudly():
    gpt = make_gpt('sinusoidal')
    x = torch.randn(1, 1, N_EMBD)
    gpt.pos_emb(x, start=BLOCK - 1)           # 边界内
    with pytest.raises(IndexError, match='位置越界'):
        gpt.pos_emb(x, start=BLOCK)
