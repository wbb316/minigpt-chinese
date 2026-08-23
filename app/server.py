"""FastAPI 后端：加载训练好的模型，提供中文续写接口。"""
import sys
import os

# ★ 必须先加 sys.path，才能 import 到上级目录的 model 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import pickle
import html
import re

from model.gpt import GPT


def clean_text(text: str) -> str:
    """清理生成的文本：还原 HTML 实体，过滤控制字符。

    模型可能输出 &#8943; 这类 HTML 实体（因为字节级 BPE 会把字符转义），
    这里把它们还原成正常字符，并过滤掉乱码控制字符。
    """
    text = html.unescape(text)  # &#8943; → ⋯
    # 过滤控制字符（除换行/制表符外）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的 GPT（best: 7层，256维，词表3256，序列128）
with open('../result/tokenizer_best.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

gpt = GPT(vocab_size=len(tokenizer.vocab), block_size=128, n_layer=7, n_head=8, n_embd=256)
gpt.load_state_dict(torch.load('../result/checkpoint_best.pt', map_location=device))
gpt.to(device).eval()

# FastAPI 应用
app = FastAPI()

class Generation(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0

@app.post("/generate")
def generate(request: Generation):
    ids = tokenizer.encode(request.prompt)
    # 只保留最后 block_size 个（模型最多看 block_size）
    idx = torch.tensor(ids[-gpt.block_size:], device=device).unsqueeze(0)

    for _ in range(request.max_tokens):
        idx_cond = idx[:, -gpt.block_size:]
        logits = gpt(idx_cond)
        logits = logits[:, -1, :] / request.temperature   # ★ 用 request.temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_token], dim=1)         # next_token 保持 tensor

    result = tokenizer.decode(idx[0].tolist())
    # 清理：把 HTML 实体（如 &#8943;）还原成正常字符，过滤控制字符
    result = clean_text(result)
    return {"text": result, "prompt": request.prompt}

@app.get("/", response_class=HTMLResponse)
def home():
    """返回前端页面"""
    with open('templates/index.html', encoding='utf-8') as f:
        return f.read()

@app.get("/health")
def health():
    return {"status": "ok"}
