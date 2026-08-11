import torch
import torch.nn.functional as F
from model.gpt import GPT

def train_step(model, optimizer, inputs, targets):
    logits = model(inputs)
    loss=F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def test_overfit_small_data():
    torch.manual_seed(42)
    gpt=GPT(vocab_size=16,n_layer=2,n_head=2,n_embd=32,block_size=32)
    optimizer=torch.optim.Adam(gpt.parameters(),lr=1e-3)
    seq=torch.randint(0,16,(8,32))
    input,target=seq[:,:-1],seq[:,1:]
    with torch.no_grad():
        init_loss = F.cross_entropy(gpt(input).view(-1,16),target.reshape(-1)).item()
    final_loss=None
    for _ in range(200):
        final_loss=train_step(gpt, optimizer, input, target)
    print(f"初始 loss={init_loss:.3f} (随机猜≈ln(16)≈2.77) -> 最终 loss={final_loss:.3f}")
    assert init_loss > 2.5, f"初始loss应约等于ln(16)≈2.77，实际{init_loss}"
    assert final_loss < 1.0, f"过拟合失败：final={final_loss}，模型可能没在学"
