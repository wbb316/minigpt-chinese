import torch
import torch.nn.functional as F
from data.tokenizer import BPETokenizer
from model.gpt import GPT
from data.dataset import TextDataset
import pickle

with open('data/corpus.txt',encoding='utf-8') as f:
    text=f.read()
print(f'语料长度：{len(text)}')
tokenizer = BPETokenizer(vocab_size=256+1200)
tokenizer.train(text)
print(f'tokenizer词表大小：{len(tokenizer.vocab)}')
tokens=tokenizer.encode(text)
print(f"编码完成: {len(text)} 字符 → {len(tokens)} token")

block_size=64
ds=TextDataset(tokens=tokens[:100000],block_size=block_size)
loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
print(f"数据集样本数: {len(ds)}，每批 16 个")

gpt=GPT(vocab_size=len(tokenizer.vocab),block_size=block_size,n_layer=2,n_head=4,n_embd=64)
optimizer=torch.optim.AdamW(gpt.parameters(),lr=1e-3)
print(f"GPT 参数量: {gpt.get_num_params()}")

for epoch in range(2):
    total_loss=0
    for i,(x,y) in enumerate(loader):
        logits=gpt(x)
        loss=F.cross_entropy(logits.view(-1,logits.size(-1)),y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        if i%200==0:
            print(f"epoch {epoch}, step {i}, loss {loss.item():.3f}")
    print(f"epoch {epoch} 平均 loss: {total_loss / len(loader):.3f}")
torch.save(gpt.state_dict(), 'checkpoint.pt')           # 存模型权重
with open('tokenizer.pkl', 'wb') as f:                  # 存分词器（生成时要解码）
    pickle.dump(tokenizer, f)
print("模型已保存: checkpoint.pt + tokenizer.pkl")


