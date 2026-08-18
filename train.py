import torch
import torch.nn.functional as F
from data.tokenizer import BPETokenizer
from model.gpt import GPT
from data.dataset import TextDataset
import pickle
import os

# 自动选择设备：有 GPU 用 GPU，没有用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 确保保存目录存在（重要！上次就是这里崩的）
os.makedirs('result', exist_ok=True)

with open('data/corpus.txt',encoding='utf-8') as f:
    text=f.read()
print(f'语料长度：{len(text)}')
tokenizer = BPETokenizer(vocab_size=256+1200)
# 从语料不同位置均匀采样30万字训练分词器（比只取开头更均衡）
sample_parts = [text[s:s+75000] for s in range(0, len(text), len(text)//4)]
tokenizer.train(''.join(sample_parts))
print(f'tokenizer词表大小：{len(tokenizer.vocab)}')
tokens=tokenizer.encode(text)
print(f"编码完成: {len(text)} 字符 → {len(tokens)} token")

block_size=64
ds=TextDataset(tokens=tokens,block_size=block_size)
loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
print(f"数据集样本数: {len(ds)}，每批 256 个")

gpt=GPT(vocab_size=len(tokenizer.vocab),block_size=block_size,n_layer=6,n_head=8,n_embd=256)
gpt = gpt.to(device)
optimizer=torch.optim.AdamW(gpt.parameters(),lr=1e-3)
print(f"GPT 参数量: {gpt.get_num_params()}")

for epoch in range(15):
    total_loss=0
    for i,(x,y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits=gpt(x)
        loss=F.cross_entropy(logits.view(-1,logits.size(-1)),y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        if i%200==0:
            print(f"epoch {epoch}, step {i}, loss {loss.item():.3f}")
    avg_loss = total_loss / len(loader)
    print(f"epoch {epoch} 平均 loss: {avg_loss:.3f}")

    # ★ 每轮保存一次（防止中途崩白跑，上次就是只最后保存才丢的）
    torch.save(gpt.state_dict(), 'result/checkpoint.pt')
    with open('result/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"  ✓ epoch {epoch} 已保存 (result/checkpoint.pt)")

print("训练完成，模型已保存: result/checkpoint.pt + result/tokenizer.pkl")
