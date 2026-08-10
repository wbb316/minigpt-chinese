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
    torch.manual_seed(0)
    gpt=GPT(vocab_size=50,n_layer=2,n_head=2,n_embd=32,block_size=32)
    gpt.eval()
    idx=torch.tensor([[5,5,5,5]])
    with torch.no_grad():
        logist=gpt(idx)
    assert not torch.allclose(logist[0,3], logist[0,1], atol=1e-3)