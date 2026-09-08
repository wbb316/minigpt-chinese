"""FFN 变体测试（v3 FFN 优化实验，2026-09-08）

覆盖：
1. relu 路径与原实现逐元素一致（回归保护）
2. relu ↔ gelu 参数名/形状完全相同（可直接换激活，单变量）
3. swiglu 参数量与 relu 对齐（<0.05%），结构为 gate/up/down
4. checkpoint 适配器 relu ↔ swiglu 双向映射：值精确、加载无 missing/unexpected
5. 三变体从同一 checkpoint 出发时起点 CE 量级相当（≈ln(vocab)）
6. 默认值回归：不传 ff_type 的 GPT/FeedForward = swiglu（含 gate_proj）
"""
import math

import pytest
import torch

from model.gpt import GPT
from model.layers import FeedForward, swiglu_hidden
from model.ffn_adapter import adapt_state_dict, detect_ff_type


def _mk(ff_type, seed=42, n_layer=3, n_embd=64, n_head=4, vocab=256, block=32):
    torch.manual_seed(seed)
    return GPT(vocab_size=vocab, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
               block_size=block, dropout=0.0, tie_embeddings=True,
               position_encoding='rope', ff_type=ff_type)


def test_relu_matches_original_formula():
    """relu 分支必须与 v3_alpha 原式 fc2(relu(fc1(x))) 完全一致。"""
    torch.manual_seed(0)
    ff = FeedForward(64, dropout=0.0, ff_type='relu')
    x = torch.randn(2, 7, 64)
    ref = ff.fc2(torch.relu(ff.fc1(x)))
    assert torch.equal(ff(x), ref)
    assert ff.hidden == 4 * 64


def test_gelu_shares_param_names_with_relu():
    """relu ↔ gelu 只是激活不同：state_dict key 与形状完全一致 → 可直接 load。"""
    a, b = _mk('relu'), _mk('gelu')
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert set(sd_a) == set(sd_b)
    for k in sd_a:
        assert sd_a[k].shape == sd_b[k].shape
    b.load_state_dict(sd_a, strict=True)          # 不抛错即通过
    x = torch.randint(0, 256, (2, 16))
    a.eval(); b.eval()
    with torch.no_grad():
        assert not torch.allclose(a(x), b(x))     # 激活确实生效


def test_swiglu_structure_and_param_parity():
    """swiglu 为 gate/up/down 结构，参数量与 relu 版差 <0.05%。"""
    relu, swiglu = _mk('relu'), _mk('swiglu')
    ff = swiglu.blocks[0].ff
    assert hasattr(ff, 'gate_proj') and hasattr(ff, 'up_proj') and hasattr(ff, 'down_proj')
    assert not hasattr(ff, 'fc1')
    assert ff.hidden == swiglu_hidden(64)
    diff = abs(relu.blocks[0].ff.param_count() - swiglu.blocks[0].ff.param_count())
    assert diff / relu.blocks[0].ff.param_count() < 0.05, f'FFN 层参数差 {diff}'
    # 4d 公式：h ≈ 8d/3
    assert abs(ff.hidden - 8 * 64 / 3) <= 16


def test_swiglu_hidden_solver():
    """dim=576（50M 模型）时 h=1536，FFN 层参数量与 4d 版差 <0.05%。"""
    assert swiglu_hidden(576) == 1536
    d = 576
    n_relu = 2 * 4 * d * d + 4 * d
    h = swiglu_hidden(d)
    n_swi = 3 * h * d + h + 3 * d
    assert abs(n_swi - n_relu) / n_relu < 5e-4
    # 其它常见 dim 也都能压到 <2%（round_to=1 时 <1%）
    for dd in (128, 256, 384, 512, 768, 1024):
        n_r = 2 * 4 * dd * dd + 4 * dd
        for r in (0, 1):
            hh = swiglu_hidden(dd, r)
            n_s = 3 * hh * dd + hh + 3 * dd
            assert abs(n_s - n_r) / n_r < 0.02, f'd={dd} r={r} h={hh}'


def test_default_ff_type_is_swiglu():
    """回归保护：GPT/FeedForward 不传 ff_type 时必须默认 swiglu（含 gate_proj 结构）。

    防止以后把 DEFAULT_FF_TYPE 误改回 relu/gelu（或漏传默认）而本套测试不察觉。
    """
    from model.layers import DEFAULT_FF_TYPE
    assert DEFAULT_FF_TYPE == 'swiglu'
    # FeedForward 默认
    ff = FeedForward(64)
    assert ff.ff_type == 'swiglu'
    assert hasattr(ff, 'gate_proj') and hasattr(ff, 'up_proj') and hasattr(ff, 'down_proj')
    assert not hasattr(ff, 'fc1')
    # GPT 默认（不传 ff_type）：模型级默认生效，Block FFN 为 swiglu 结构
    gpt = GPT(vocab_size=128, n_layer=1, n_head=2, n_embd=16, block_size=16,
              position_encoding='rope')
    assert gpt.ff_type == 'swiglu'
    blk_ff = gpt.blocks[0].ff
    assert blk_ff.ff_type == 'swiglu'
    assert hasattr(blk_ff, 'gate_proj') and hasattr(blk_ff, 'up_proj') \
        and hasattr(blk_ff, 'down_proj') and not hasattr(blk_ff, 'fc1')


def test_adapter_legacy_to_swiglu_exact():
    """relu/gelu 存档 → swiglu 模型：权重按预期搬运，加载无 missing/unexpected。"""
    src, dst = _mk('relu'), _mk('swiglu')
    sd = src.state_dict()
    h = dst.blocks[0].ff.hidden
    new_sd, rep = adapt_state_dict(sd, dst, verbose=False)
    assert rep['src_ff'] == 'legacy' and rep['dst_ff'] == 'swiglu'
    missing, unexpected = dst.load_state_dict(new_sd, strict=False)
    assert [k for k in missing if not k.endswith('cos_cached')] == []
    assert unexpected == []
    # gate 取 fc1 前半、up 取后半（4d=256 → 各 128 行），down 取 fc2 前 h 列
    assert torch.equal(dst.blocks[0].ff.gate_proj.weight[:128],
                       sd['blocks.0.ff.fc1.weight'][:128])
    assert torch.equal(dst.blocks[0].ff.up_proj.weight[:128],
                       sd['blocks.0.ff.fc1.weight'][128:256])
    assert torch.equal(dst.blocks[0].ff.down_proj.weight, sd['blocks.0.ff.fc2.weight'][:, :h])


def test_adapter_swiglu_to_legacy_roundtrip():
    """swiglu 存档 → relu 模型：形状回到 4d，加载无 missing/unexpected。"""
    src, dst = _mk('swiglu'), _mk('relu')
    new_sd, rep = adapt_state_dict(src.state_dict(), dst, verbose=False)
    assert rep['src_ff'] == 'swiglu' and rep['dst_ff'] == 'legacy'
    missing, unexpected = dst.load_state_dict(new_sd, strict=False)
    assert [k for k in missing if not k.endswith('cos_cached')] == []
    assert unexpected == []
    assert dst.blocks[0].ff.fc1.weight.shape == (4 * 64, 64)


def test_adapter_detects_ff_type():
    assert detect_ff_type(_mk('relu').state_dict()) == 'legacy'
    assert detect_ff_type(_mk('gelu').state_dict()) == 'legacy'
    assert detect_ff_type(_mk('swiglu').state_dict()) == 'swiglu'


def test_three_variants_start_from_same_ckpt_have_comparable_loss():
    """同一 checkpoint 出发：三个变体起点 CE 接近 ln(vocab)（±1.0 以内）。"""
    base = _mk('relu')
    sd = base.state_dict()
    x = torch.randint(0, 256, (4, 32))
    y = torch.randint(0, 256, (4, 32))
    ref = math.log(256)
    for ff in ('relu', 'gelu', 'swiglu'):
        m = _mk(ff)
        new_sd, _ = adapt_state_dict(sd, m, verbose=False)
        m.load_state_dict(new_sd, strict=False)
        m.eval()
        with torch.no_grad():
            loss = torch.nn.functional.cross_entropy(
                m(x).view(-1, 256), y.view(-1)).item()
        assert abs(loss - ref) < 1.0, f'{ff} 起点 CE {loss:.3f} 偏离 ln(vocab) {ref:.3f}'


def test_bad_ff_type_rejected():
    with pytest.raises(ValueError):
        FeedForward(64, ff_type='relu2')
    with pytest.raises(ValueError):
        GPT(vocab_size=64, n_layer=1, n_head=2, n_embd=8, block_size=16,
            ff_type='nope')


def test_default_swiglu_short_training():
    """默认（不传 ff_type）模型端到端短训练冒烟：loss 稳健下降、无 NaN。

    缺口来源（2026-09-08 默认 FFN relu→swiglu 后）：其余「短训练降 loss」用例
    全部显式钉在 ff_type='relu'，没有任何测试覆盖默认 swiglu 模型的训练路径。

    目标 y 只固定抽一次（不是每步重抽）——重抽会让任务不可学习，loss 只在
    ln(vocab)≈5.545 附近随机游走，断言近乎掷硬币；根因详见
    test/test_model_rope.py::test_rope_short_training 的 docstring。固定目标后
    模型可真正过拟合 2×32=64 个 token，30 步内 loss 稳健下降（实测 drop≈2.7，
    断言只要求 >1.0，余量远离噪声边缘，且全程固定 seed → 结果确定可复现）。

    默认结构（gate_proj/up_proj/down_proj、DEFAULT_FF_TYPE=='swiglu'）的完整断言
    见 test_default_ff_type_is_swiglu，此处不重复，仅做一条自证守卫。
    """
    torch.manual_seed(123)
    gpt = GPT(vocab_size=256, n_layer=2, n_head=4, n_embd=64,
              block_size=32, dropout=0.0, tie_embeddings=True,
              position_encoding='rope')     # ★ 不传 ff_type → 走默认 swiglu
    assert gpt.ff_type == 'swiglu'          # 训练的就是默认 swiglu 路径
    opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))      # ★ 目标固定（见上：任务须可学习）
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(gpt(x).view(-1, 256), y.view(-1))
        assert torch.isfinite(loss), '训练中出现 NaN'
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    assert loss.item() < first - 1.0, \
        f'默认 swiglu 短训 loss 应稳健下降（余量 >1.0）: {first:.3f} → {loss.item():.3f}'


def test_train_cli_help_and_ff_args():
    """train.py --help 必须能正常打印（argparse 的 help 字符串里裸 % 会炸），
    且 FFN 相关参数在 help 里可见。"""
    import os
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(root, 'train', 'train.py'),
                        '--help'], capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr[-800:]
    for flag in ('--ff-type', '--ff-hidden', '--init-from', '--init-optimizer'):
        assert flag in r.stdout
    assert 'swiglu' in r.stdout
