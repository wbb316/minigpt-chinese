import sys
import os
# ★ 让 Python 能找到上级目录的 data/model 包（train.py 在子目录运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from data.tokenizer import BPETokenizer
from model.gpt import GPT
from data.dataset import TextDataset
import pickle
from tqdm import tqdm

# 自动选择设备：有 GPU 用 GPU，没有用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 确保保存目录存在（重要！上次就是这里崩的）
os.makedirs('../result', exist_ok=True)

with open('../data/corpus.txt', encoding='utf-8') as f:
    text=f.read()
print(f'语料长度：{len(text)}')
tokenizer = BPETokenizer(vocab_size=256+3000)   # 词表 3256
# 从语料不同位置均匀采样80万字训练分词器（词表3256，需要更多数据支撑合并）
sample_parts = [text[s:s+200000] for s in range(0, len(text), len(text)//4)]
tokenizer.train(''.join(sample_parts))
print(f'tokenizer词表大小：{len(tokenizer.vocab)}')

# ★ 编码缓存：文件名带词表大小（防止不同词表的 token 混用）
import numpy as np
vocab_size = len(tokenizer.vocab)
cache_path = f'../data/tokens_cache_v{vocab_size}.npy'
if os.path.exists(cache_path):
    tokens = np.load(cache_path).tolist()
    print(f"从缓存加载 {len(tokens)} token ({cache_path})")
else:
    tokens = tokenizer.encode(text)
    np.save(cache_path, np.array(tokens))
    print(f"编码完成并缓存: {len(text)} 字符 → {len(tokens)} token ({cache_path})")

block_size=64

# ★ 数据划分：训练集 90%、验证集 10%（判断过拟合的关键）
split = int(len(tokens) * 0.9)
train_tokens = tokens[:split]
val_tokens = tokens[split:]
print(f"训练集 {len(train_tokens)} token, 验证集 {len(val_tokens)} token")

train_ds = TextDataset(tokens=train_tokens, block_size=block_size)
val_ds = TextDataset(tokens=val_tokens, block_size=block_size)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
print(f"训练集样本数: {len(train_ds)}，验证集样本数: {len(val_ds)}，每批 512 个")

gpt=GPT(vocab_size=len(tokenizer.vocab),block_size=block_size,n_layer=10,n_head=8,n_embd=256,dropout=0.1)   # 10层 = 9.6M + dropout防过拟合
gpt = gpt.to(device)
optimizer=torch.optim.AdamW(gpt.parameters(),lr=2e-3)   # batch 翻倍，lr 也翻倍
scaler = torch.cuda.amp.GradScaler()   # 混合精度的梯度缩放器
print(f"GPT 参数量: {gpt.get_num_params()}")

for epoch in range(5):   # 减到5轮，防过拟合（val loss 升就早停）
    # ========== 训练 ==========
    gpt.train()
    total_loss=0
    # 进度条：显示当前 epoch 内的训练进度 + 预计剩余时间
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="step", ncols=100)
    for i,(x,y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        # 混合精度：前向用半精度（4090 支持，快 1.5-2 倍）
        with torch.cuda.amp.autocast():
            logits=gpt(x)
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),y.view(-1))
        optimizer.zero_grad()
        scaler.scale(loss).backward()   # 用 scaler 缩放梯度
        scaler.step(optimizer)
        scaler.update()
        total_loss+=loss.item()
        # 在进度条后显示当前 loss
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    pbar.close()
    avg_loss = total_loss / len(train_loader)

    # ========== 验证（不更新参数，只算 loss）==========
    gpt.eval()   # 切到评估模式
    val_loss = 0
    with torch.no_grad():   # 不计算梯度（省内存）
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast():
                logits = gpt(x)
                loss_val = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            val_loss += loss_val.item()
    val_loss /= len(val_loader)

    print(f"epoch {epoch} 平均 loss: {avg_loss:.3f}, val loss: {val_loss:.3f}")

    # ★ 判断过拟合
    if epoch > 0:
        pass  # 过拟合判断：看 val_loss 是否回升（可在外面分析）

    # ★ 每轮保存一次（防止中途崩白跑）
    torch.save(gpt.state_dict(), '../result/checkpoint_baseline.pt')
    with open('../result/tokenizer_baseline.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"  ✓ epoch {epoch} 已保存 (result/checkpoint.pt)")

print("训练完成，模型已保存: result/checkpoint_baseline.pt + result/tokenizer_baseline.pkl")
