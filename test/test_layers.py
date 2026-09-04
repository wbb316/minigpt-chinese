import torch
from model.layers import LayerNorm

def test_layernorm_normalize():
    torch.manual_seed(42)
    ln = LayerNorm(dim=64)
    x = torch.randn(2, 10, 64) * 5 + 3
    y = ln(x)
    assert y.shape == x.shape
    assert torch.allclose(y.mean(dim=-1), torch.zeros(2, 10), atol=1e-4)
    # 标准 LayerNorm 用有偏方差归一化，所以 torch.var（默认无偏, 除以 N-1）
    # 的输出方差 = N/(N-1) = 64/63 ≈ 1.0159
    assert torch.allclose(y.var(dim=-1), torch.full((2, 10), 64.0 / 63.0), atol=1e-3)