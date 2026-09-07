# -*- coding: utf-8 -*-
"""RoPE 位置编码（可切换）测试。

验证：
1. 旋转数学：||q_rot|| == ||q||（正交）、不同位置旋转不同、公式 = q*cos+rotate_half(q)*sin
2. rope 模式 GPT：forward shape 正确、无 NaN、initial CE ≈ ln(vocab)
3. sinusoidal 模式：attn.rope is None（旧行为路径不变）、输出与 rope 模式不同
4. KV cache 增量（rope）：分步 past_kvs logits == 整段前向（start 偏移正确性）
5. 短训练冒烟：rope 模式 loss 下降、无 NaN
"""
import math

import pytest
import torch
import torch.nn.functional as F

from model.gpt import GPT
from model.rope import RotaryEmbedding, rotate_half

VOCAB, N_LAYER, N_HEAD, N_EMBD, BLOCK = 512, 2, 4, 64, 32
B = 2


def make_gpt(pe):
    return GPT(vocab_size=VOCAB, block_size=BLOCK, n_layer=N_LAYER,
               n_head=N_HEAD, n_embd=N_EMBD, dropout=0.0,
               tie_embeddings=True, position_encoding=pe)


def test_rotary_math():
    """旋转保持 norm；不同位置旋转不同；公式正确。"""
    hd = 64
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, hd)
    rope = RotaryEmbedding(hd, max_seq_len=16)
    q0, _ = rope(q, q, start=0)
    q1, _ = rope(q, q, start=1)
    q_at0, _ = rope(q, q, start=0)
    assert torch.allclose(q0[0].norm(), q[0].norm(), atol=1e-5), '旋转应保持 norm'
    assert not torch.allclose(q0, q1, atol=1e-5), '不同位置应旋转不同'
    # 公式对照：手工计算（位置 0：cos 全 1 / sin 全 0 → 恒等）
    cos = rope.cos_cached[:, :, 0]
    sin = rope.sin_cached[:, :, 0]
    manual = q * cos + rotate_half(q) * sin
    assert torch.allclose(q0, manual, atol=1e-6), '实现应与公式一致'
    # 位置 0 旋转应恒等（cos=1, sin=0）
    assert torch.allclose(q_at0, q, atol=1e-6), '位置 0（cos=1,sin=0）应恒等'


def test_rope_gpt_forward_backward():
    """rope 模式：shape、无 NaN、initial CE ≈ ln(vocab)。"""
    gpt = make_gpt('rope')
    x = torch.randint(0, VOCAB, (B, BLOCK))
    y = torch.randint(0, VOCAB, (B, BLOCK))
    logits = gpt(x)
    assert logits.shape == (B, BLOCK, VOCAB)
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
    assert torch.isfinite(loss)
    assert abs(loss.item() - math.log(VOCAB)) < 1.5, \
        f'rope 模式 initial CE 应接近 ln(vocab)={math.log(VOCAB):.2f}，得到 {loss.item():.2f}'
    loss.backward()
    for n, p in gpt.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_sinusoidal_default_unchanged():
    """sinusoidal（默认）行为路径不变：attn.rope is None；输出与 rope 不同。"""
    g_sin = make_gpt('sinusoidal')
    g_rope = make_gpt('rope')
    for blk in g_sin.blocks:
        assert blk.attn.rope is None, 'sinusoidal 模式不应注入 rope'
    for blk in g_rope.blocks:
        assert blk.attn.rope is not None
    # 同权重下两种位置编码输出应不同（位置信息注入方式不同）
    g_rope.load_state_dict({k: v for k, v in g_sin.state_dict().items()
                            if 'rope' not in k and k != 'pos_emb.pe'},
                           strict=False)   # 允许 rope buffer 缺失（用 rope 默认）
    torch.manual_seed(1)
    x = torch.randint(0, VOCAB, (B, BLOCK))
    with torch.no_grad():
        o_sin = g_sin(x)
        o_rope = g_rope(x)
    assert not torch.allclose(o_sin, o_rope, atol=1e-3), \
        'sinusoidal 与 rope 输出应不同（位置注入机制不同）'


def test_rope_kv_cache_consistency():
    """rope 模式 KV cache 增量 == 整段前向（start 偏移正确性）。"""
    gpt = make_gpt('rope').eval()
    torch.manual_seed(2)
    ids = torch.randint(0, VOCAB, (1, BLOCK))
    with torch.no_grad():
        logits_full, _ = gpt(ids, return_kv=True)
        # 分两步：前 T1 个 + 后 T2 个（past_kvs 增量）
        T1 = 12
        x1, x2 = ids[:, :T1], ids[:, T1:]
        _, kvs = gpt(x1, return_kv=True)
        logits2, _ = gpt(x2, past_kvs=kvs)
        full_last = logits_full[:, T1:, :]
        assert torch.allclose(full_last, logits2, atol=1e-4), \
            'rope KV cache 增量 logits 与整段不一致'


def test_rope_short_training():
    """rope 模式短训练冒烟：loss 下降、无 NaN。"""
    gpt = make_gpt('rope')
    opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    torch.manual_seed(3)
    x = torch.randint(0, VOCAB, (B, BLOCK))
    first = None
    for _ in range(30):
        y = torch.randint(0, VOCAB, (B, BLOCK))
        opt.zero_grad()
        loss = F.cross_entropy(gpt(x).view(-1, VOCAB), y.view(-1))
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    assert loss.item() < first, f'loss 应下降: {first:.3f} → {loss.item():.3f}'
