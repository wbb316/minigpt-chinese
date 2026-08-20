"""实验B：对比模型大小 —— 小模型（1.2M参数）
数据：200万字符（均匀采样，与实验C一致，避免encode全量慢）
模型：2层/4头/128维（约1.2M参数）
"""
import torch
import torch.nn.functional as F
from data.tokenizer import BPETokenizer
from model.gpt import GPT
from data.dataset import TextDataset
import pickle
import os
import csv

os.makedirs('result_b', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

with open('data/corpus.txt',encoding='utf-8') as f:
    text=f.read()
print(f'语料长度：{len(text)}')

# 均匀采样200万字符（与实验C一致，encode快）
target = 2000000
step = len(text) // 8
sample_parts = [text[i:i+target//8] for i in range(0, len(text), step)]
text = ''.join(sample_parts)[:target]
print(f'实验数据量（均匀采样）: {len(text)} 字符')

tokenizer = BPETokenizer(vocab_size=256+1200)
sample_parts = [text[s:s+75000] for s in range(0, len(text), len(text)//4)]
tokenizer.train(''.join(sample_parts))
print(f'tokenizer词表大小：{len(tokenizer.vocab)}')
tokens=tokenizer.encode(text)
print(f"编码完成: {len(text)} 字符 → {len(tokens)} token")

block_size=64
ds=TextDataset(tokens=tokens,block_size=block_size)
loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
print(f"数据集样本数: {len(ds)}，每批 256 个")

# ★ 小模型：2层/4头/128维（约1.2M参数）
gpt=GPT(vocab_size=len(tokenizer.vocab),block_size=block_size,n_layer=2,n_head=4,n_embd=128)
gpt = gpt.to(device)
optimizer=torch.optim.AdamW(gpt.parameters(),lr=1e-3)
print(f"GPT 参数量: {gpt.get_num_params()}")

# ★ 记录 loss 到 CSV（画曲线用，防止终端关闭丢数据）
loss_csv = open('result_b/loss_history.csv', 'w', newline='')
csv_writer = csv.writer(loss_csv)
csv_writer.writerow(['epoch', 'train_loss'])

for epoch in range(10):
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

    # 写入 CSV
    csv_writer.writerow([epoch, f"{avg_loss:.4f}"])
    loss_csv.flush()

    torch.save(gpt.state_dict(), 'result_b/checkpoint.pt')
    with open('result_b/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"  ✓ epoch {epoch} 已保存 (result_b/checkpoint.pt)")

loss_csv.close()
print("训练完成，模型已保存: result_b/checkpoint.pt + result_b/tokenizer.pkl")
print("loss 已记录到: result_b/loss_history.csv")
