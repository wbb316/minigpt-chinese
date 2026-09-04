"""model/sampling.py 单测：temperature / top_p / repetition_penalty 三个采样器。"""
import pytest
import torch

from model.sampling import sample, apply_top_p, apply_repetition_penalty


def logits_like(t):
    """t 任意形状张量 → 模拟 (B, T, V) logits，最后一维是 V。"""
    return t.float()


# ---------------------------------------------------------------- temperature
def test_greedy_when_temp_zero():
    lg = torch.tensor([[[0.1, 5.0, 0.3]]])
    out = sample(lg, temperature=0.0)
    assert out.shape == (1, 1)
    assert int(out) == 1


def test_temperature_scales_logits():
    lg = torch.tensor([[[0.0, 1.0]]])
    # temp=1: softmax([0,1])；temp 大 → 更均匀
    p_low = torch.softmax(lg / 1.0, dim=-1)[0, 0]
    p_high = torch.softmax(lg / 10.0, dim=-1)[0, 0]
    assert p_low[0] < p_high[0]           # 低温更尖锐


def test_sample_distribution_shape_and_range():
    lg = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
    rng = torch.Generator().manual_seed(0)
    out = sample(lg, temperature=1.0, rng=rng)
    assert out.shape == (1, 1)
    assert 0 <= int(out) < 4


# ---------------------------------------------------------------- top_p
def test_top_p_keeps_only_top_token():
    """概率悬殊时，top_p 很小 → 只剩最高概率 token（确定性）。"""
    lg = torch.tensor([[[0.0, 0.0, 0.0, 10.0]]])   # token 3 压倒性
    lg = apply_top_p(lg, top_p=0.1)
    probs = torch.softmax(lg, dim=-1)
    assert probs[0, 0, 3] == pytest.approx(1.0)
    assert probs[0, 0, 0] == 0.0


def test_top_p_identity_when_1():
    lg = torch.tensor([[[1.0, 2.0, 3.0]]])
    assert torch.equal(apply_top_p(lg.clone(), top_p=1.0), lg)


def test_top_p_never_empty():
    """极端 top_p 也不能清空分布（至少保留最高概率 token）。"""
    lg = torch.tensor([[[0.0, 1.0, 2.0]]])
    lg = apply_top_p(lg, top_p=1e-9)
    probs = torch.softmax(lg, dim=-1)
    assert torch.isfinite(probs).all()
    assert probs.sum().item() == pytest.approx(1.0)
    assert probs[0, 0].argmax().item() == 2   # 最高概率 token 被保留


# ---------------------------------------------------------------- repetition penalty
def test_repetition_penalty_reduces_past_token():
    """出现过 token 1 后，其 logit 按 penalty 下降。"""
    lg = torch.tensor([[[0.0, 5.0, 0.0]]])
    apply_repetition_penalty(lg, penalty=2.0, prev_ids=[1])
    # logit>0 → 除以 penalty：5/2 = 2.5
    assert lg[0, 0, 1].item() == pytest.approx(2.5)


def test_repetition_penalty_negative_logit_multiplies():
    """logit<0 → 乘 penalty（更负，进一步抑制）。"""
    lg = torch.tensor([[[0.0, -4.0, 0.0]]])
    apply_repetition_penalty(lg, penalty=2.0, prev_ids=[1])
    assert lg[0, 0, 1].item() == pytest.approx(-8.0)


def test_repetition_penalty_noop_when_1():
    lg = torch.tensor([[[0.0, 5.0, 0.0]]])
    lg2 = lg.clone()
    apply_repetition_penalty(lg2, penalty=1.0, prev_ids=[1])
    assert torch.equal(lg, lg2)


def test_repetition_penalty_dedup():
    """同一 token 重复出现多次只罚一次（set 去重）。"""
    lg = torch.tensor([[[0.0, 5.0, 0.0]]])
    apply_repetition_penalty(lg, penalty=2.0, prev_ids=[1, 1, 1])
    assert lg[0, 0, 1].item() == pytest.approx(2.5)


def test_repetition_penalty_many_tokens():
    lg = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    apply_repetition_penalty(lg, penalty=1.5, prev_ids=[0, 2, 4])
    expect = torch.tensor([[1.0 / 1.5, 2.0, 3.0 / 1.5, 4.0, 5.0 / 1.5]])
    assert torch.allclose(lg, expect)
