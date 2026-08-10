import torch
from model.attention import MultiHeadAttention

def test_attention_output_shape():
    torch.manual_seed(0)
    attn=MultiHeadAttention(dim=64,n_heads=4)
    x=torch.randn(2,10,64)
    y=attn(x)
    assert y.shape == (2,10,64)

def test_attention_causal_mask_blocks_future():
    """因果 mask 生效：位置 t 的输出不应依赖 t 之后的位置。

    验证方法：改输入在位置 3 之后的 token，位置 0..2 的输出应不变。
    """
    torch.manual_seed(0)
    attn=MultiHeadAttention(dim=32,n_heads=2)
    x1=torch.randn(1,6,32)
    x2=x1.clone()
    x2[0,3:]=torch.randn(1,3,32)
    with torch.no_grad():
        y1=attn(x1)
        y2=attn(x2)
    assert torch.allclose(y1[0,:3], y2[0,:3], rtol=1e-3)
