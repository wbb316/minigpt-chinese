import torch
from model.gpt import GPT

def test_gpt_out_shape():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    gpt=GPT(vocab_size=100,n_layer=2,n_head=4,n_embd=32,block_size=64)
    idx=torch.randint(0,100,(2,10))
    logist=gpt(idx)
    assert logist.shape == (2,10,100)
    print(f"参数量: {gpt.get_num_params() if hasattr(gpt, 'get_num_params') else '未实现'}")

def test_gpt_different_positions_differ():
    """sinusoidal 的绝对位置特性：相同 token 在不同位置输出不同。
    （rope 模式不带绝对位置信号，此特性不适用——该测试显式用 sinusoidal）"""
    torch.manual_seed(0)
    gpt = GPT(vocab_size=50, n_layer=2, n_head=2, n_embd=32, block_size=32,
              position_encoding='sinusoidal')
    gpt.eval()
    idx = torch.tensor([[5, 5, 5, 5]])
    with torch.no_grad():
        logist = gpt(idx)
    assert not torch.allclose(logist[0, 3], logist[0, 1], atol=1e-3)