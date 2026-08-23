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

# ★ 分别读取 train.txt 和 val.txt（按文件划分，验证集覆盖所有书尾部）
with open('../data/train.txt', encoding='utf-8') as f:
    train_text = f.read()
with open('../data/val.txt', encoding='utf-8') as f:
    val_text = f.read()
print(f'训练文本：{len(train_text)} 字符，验证文本：{len(val_text)} 字符')

# 分词器：用训练文本采样200万训练（100段 × 20k）
tokenizer = BPETokenizer(vocab_size=256+3000)   # 词表 3256
sample_parts = [train_text[s:s+20000] for s in range(0, len(train_text), len(train_text)//100)]
tokenizer.train(''.join(sample_parts))
print(f'tokenizer词表大小：{len(tokenizer.vocab)}')

# ★ 编码缓存：训练/验证分开缓存（文件名带词表大小）
import numpy as np
vocab_size = len(tokenizer.vocab)

def get_tokens(text, name):
    """编码文本，带缓存。"""
    cache_path = f'../data/tokens_{name}_v{vocab_size}.npy'
    if os.path.exists(cache_path):
        tokens = np.load(cache_path).tolist()
        print(f"从缓存加载 {name}: {len(tokens)} token ({cache_path})")
        return tokens
    tokens = tokenizer.encode(text)
    np.save(cache_path, np.array(tokens))
    print(f"编码并缓存 {name}: {len(text)} 字符 → {len(tokens)} token")
    return tokens

train_tokens = get_tokens(train_text, 'train')
val_tokens = get_tokens(val_text, 'val')
print(f"训练集 {len(train_tokens)} token, 验证集 {len(val_tokens)} token")

block_size = 128   # ★ 补回（之前改造时丢了）

train_ds = TextDataset(tokens=train_tokens, block_size=block_size)
val_ds = TextDataset(tokens=val_tokens, block_size=block_size)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
print(f"训练集样本数: {len(train_ds)}，验证集样本数: {len(val_ds)}，每批 512 个")

gpt=GPT(vocab_size=len(tokenizer.vocab),block_size=block_size,n_layer=7,n_head=8,n_embd=256,dropout=0.2)   # 7层=7.2M，降参数防过拟合
gpt = gpt.to(device)
optimizer=torch.optim.AdamW(gpt.parameters(),lr=8e-4,weight_decay=0.01)   # lr降+weight_decay防过拟合
scaler = torch.amp.GradScaler('cuda')   # 混合精度的梯度缩放器（新API）
print(f"GPT 参数量: {gpt.get_num_params()}")

# ★ 早停参数（方案A：连续2轮val不改善就停）
best_val = float('inf')     # 历史最好 val loss
patience = 2                # 容忍连续几轮不改善
no_improve_count = 0

for epoch in range(10):   # 10轮（数据充足后不过拟合）
    # ========== 训练 ==========
    gpt.train()
    total_loss=0
    # 进度条：显示当前 epoch 内的训练进度 + 预计剩余时间
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="step", ncols=100)
    for i,(x,y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        # 混合精度：前向用半精度（4090 支持，快 1.5-2 倍）
        with torch.amp.autocast('cuda'):
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
            with torch.amp.autocast('cuda'):
                logits = gpt(x)
                loss_val = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            val_loss += loss_val.item()
    val_loss /= len(val_loader)

    tqdm.write(f"\n===== epoch {epoch} 平均 loss: {avg_loss:.3f}, val loss: {val_loss:.3f} =====")

    # ★ 早停判断：val 改善了就保存最好模型；连续 patience 轮不改善就停
    if val_loss < best_val:
        best_val = val_loss
        no_improve_count = 0
        # 保存最好的模型（早停后加载这个，不是过拟合的最后轮）
        torch.save(gpt.state_dict(), '../result/checkpoint_best.pt')
        with open('../result/tokenizer_best.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        tqdm.write(f"  ✓ 新最好模型已保存 (val {val_loss:.3f})")
    else:
        no_improve_count += 1
        tqdm.write(f"  ⚠️ val 未改善 ({no_improve_count}/{patience})")

    # 每轮也保存一个"最近" checkpoint（防中断白跑）
    torch.save(gpt.state_dict(), '../result/checkpoint_latest.pt')
    with open('../result/tokenizer_latest.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # 早停触发
    if no_improve_count >= patience:
        tqdm.write(f"🚫 早停: val loss 连续 {patience} 轮未改善，停止训练")
        break

tqdm.write("训练结束。最好模型: result/checkpoint_best.pt（val 最低）")
