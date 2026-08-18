"""生成测试：加载训练好的模型，给定开头续写中文。"""
import torch
import pickle
from model.gpt import GPT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型和分词器
gpt = GPT(vocab_size=1456, block_size=64, n_layer=6, n_head=8, n_embd=256)
gpt.load_state_dict(torch.load('result/checkpoint.pt', map_location=device))
gpt.to(device).eval()

with open('result/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def generate(prompt: str, max_new_tokens: int = 150, temperature: float = 1.0):
    """给 prompt 开头，让模型续写 max_new_tokens 个字。"""
    ids = tokenizer.encode(prompt)                    # 开头转 token
    # 只保留最后 block_size 个（模型最多看 block_size）
    idx = torch.tensor(ids[-gpt.block_size:], device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = gpt(idx)                              # 前向 → 预测分数
        logits = logits[:, -1, :] / temperature       # 取最后一个位置的分数
        probs = torch.softmax(logits, dim=-1)          # 转概率
        next_token = torch.multinomial(probs, num_samples=1)  # 采样下一个 token
        idx = torch.cat([idx, next_token], dim=1)      # 拼上继续

    new_ids = idx[0].tolist()
    return tokenizer.decode(new_ids)                   # 解码成中文

# 测试几个开头
for prompt in ["真唯同学", "今天下雨", "我喜欢你"]:
    out = generate(prompt, max_new_tokens=40)
    print(f"\n【{prompt}】→\n{out}")