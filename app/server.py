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
from model.generation import generate_ids


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

# ★ 从 checkpoint 自动推断架构（层数/维度/词表/block_size 随训练配置变，这里不用再改）
def load_model(ckpt_path, tok_path, n_head=8):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    n_layer = max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.')) + 1
    n_embd = sd['token_emb.weight'].shape[1]
    vocab_size = sd['token_emb.weight'].shape[0]
    block_size = sd['pos_emb.pe'].shape[1]          # PositionalEncoding 的 buffer 记录训练长度
    assert n_embd % n_head == 0, f'n_embd={n_embd} 不能被 n_head={n_head} 整除'
    gpt = GPT(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
              n_embd=n_embd, block_size=block_size)
    gpt.load_state_dict(sd)
    gpt.to(device).eval()
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f'加载模型: {n_layer}层/{n_head}头/{n_embd}维, 词表{vocab_size}, block_size={block_size}')
    return gpt, tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CKPT = os.path.join(ROOT, 'result', 'checkpoint_best.pt')
DEFAULT_TOK = os.path.join(ROOT, 'result', 'tokenizer_best.pkl')

# FastAPI 应用
app = FastAPI()

class Generation(BaseModel):
    prompt: str
    max_tokens: int = 80
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.15

@app.post("/generate")
def generate(request: Generation):
    ids = tokenizer.encode(request.prompt)
    ctx_ids, _ = generate_ids(
        gpt, ids,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )
    result = clean_text(tokenizer.decode(ctx_ids))
    return {"text": result, "prompt": request.prompt}

@app.get("/", response_class=HTMLResponse)
def home():
    """返回前端页面"""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(app_dir, 'templates', 'index.html'), encoding='utf-8') as f:
        return f.read()

@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    """命令行启动: python app/server.py [--ckpt ...] [--tokenizer ...]"""
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(description='MiniGPT 中文续写 Web 服务')
    ap.add_argument('--ckpt', default=DEFAULT_CKPT, help='模型权重 checkpoint_best.pt')
    ap.add_argument('--tokenizer', default=DEFAULT_TOK, help='分词器 tokenizer_best.pkl')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()

    global gpt, tokenizer
    gpt, tokenizer = load_model(os.path.abspath(args.ckpt), os.path.abspath(args.tokenizer))
    print(f'服务启动: http://127.0.0.1:{args.port} （Ctrl+C 停止）')
    uvicorn.run(app, host=args.host, port=args.port)


# 命令行直接跑: python app/server.py ...
if __name__ == '__main__':
    main()

# 被 uvicorn server:app 导入时: 加载默认模型
else:
    gpt, tokenizer = load_model(DEFAULT_CKPT, DEFAULT_TOK)
