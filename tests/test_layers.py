import torch
from model.layers import LayerNorm

def test_layernorm_normalize():
    torch.manual_seed(42)
    ln=LayerNorm(dim=64)
    x=torch.randn(2,10,64)*5+3
    y=ln(x)
    assert y.shape == x.shape
    assert torch.allclose(y.mean(dim=-1), torch.zeros(2, 10), atol=1e-4)
    assert torch.allclose(y.var(dim=-1), torch.ones(2, 10), atol=1e-3)