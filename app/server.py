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
def load_model(ckpt_path, tok_path, n_head=None):
    """加载模型，架构从 state_dict 自动推断。

    n_head：**默认 None = 自动推断**。RoPE 的 `rope.cos_cached` 形状是
    `(1, 1, T, head_dim)`，于是 `n_head = n_embd / head_dim`。
    旧的 sinusoidal 存档没有 head_dim 信息 → 自动推断不可用，退回 8，
    需显式 `--n-head`（例：v2 50M 是 9 头）。
    """
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    # torch.compile 训练的 checkpoint 带 '_orig_mod.' 前缀（OptimizedModule 痕迹），
    # 剥离后所有解析逻辑按无前缀 key 统一处理（旧存档本来无前缀，兼容）
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k[len('_orig_mod.'):]: v for k, v in sd.items()}
    n_layer = max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.')) + 1
    n_embd = sd['token_emb.weight'].shape[1]
    vocab_size = sd['token_emb.weight'].shape[0]
    block_size = sd['pos_emb.pe'].shape[1]          # PositionalEncoding 的 buffer 记录训练长度
    pe = 'rope' if 'rope.cos_cached' in sd else 'sinusoidal'   # 自动检测
    # 头数：显式传入优先；否则由 RoPE 的 head_dim 推断；再否则回退 8
    head_dim = int(sd['rope.cos_cached'].shape[-1]) if pe == 'rope' else None
    if n_head is None:
        if head_dim and n_embd % head_dim == 0:
            n_head = n_embd // head_dim
            print(f'头数自动推断: head_dim={head_dim} → n_head={n_head}')
        else:
            n_head = 8
            print('⚠️ 该存档无 head_dim 信息（非 RoPE）→ 头数回退默认 8；'
                  '若模型不是 8 头，请用 --n-head 显式指定')
    assert n_embd % n_head == 0, f'n_embd={n_embd} 不能被 n_head={n_head} 整除'
    # ★ 防呆：显式传错头数时 n_embd 往往仍能被整除（704/8=88），
    #   旧代码会**静默加载成功但输出乱码**。这里用存档里的 head_dim 硬校验。
    if head_dim is not None:
        assert head_dim == n_embd // n_head, (
            f'--n-head={n_head} 与存档不符：存档 head_dim={head_dim}、n_embd={n_embd} '
            f'→ 应为 {n_embd // head_dim} 头（传 0/省略可自动推断）')
    # FFN 自动检测：含 gate_proj → swiglu（v3 FFN 变体）；否则 relu/gelu（fc1/fc2 同名）
    ff_type = 'swiglu' if any(k.endswith('gate_proj.weight') for k in sd) else 'relu'
    ff_hidden = sd['blocks.0.ff.gate_proj.weight'].shape[0] if ff_type == 'swiglu' else None
    gpt = GPT(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
              n_embd=n_embd, block_size=block_size,
              position_encoding=pe, ff_type=ff_type, ff_hidden=ff_hidden)
    gpt.load_state_dict(sd)
    gpt.to(device).eval()
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f'加载模型: {n_layer}层/{n_head}头/{n_embd}维, 词表{vocab_size}, '
          f'block_size={block_size}, 位置编码: {pe}, FFN: {ff_type}'
          + (f'(h={ff_hidden})' if ff_hidden else ''))
    return gpt, tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 默认模型 = **v3_beta 100M**（16L/704d/11H，vocab 8192，swiglu+rope）
#   shard1+2 新语料 1.96B token 单轮，val 3.0610（shard1+2 空间，见 docs/EXPERIMENT_LOG.md）
#   ⚠️ 头数 11 由 load_model 从 rope.cos_cached 自动推断，无需 --n-head
# 换回旧模型：改这两行即可（或命令行 --ckpt/--tokenizer 覆盖）
#   35M v2 (sinusoidal, 8 头): result/35M参数+998Mtokens/
#   50M v3_alpha (rope, 9 头): log/50M参数_v3_alpha+998Mtokens/
# 归档位置见 result/ 总文件夹结构（docs/MiniGPT_Project_Status.md）
DEFAULT_CKPT = os.path.join(ROOT, 'result', '100M参数v3+2Btokens', 'checkpoint_best.pt')
DEFAULT_TOK = os.path.join(ROOT, 'result', '100M参数v3+2Btokens', 'tokenizer_best.pkl')

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
    ap.add_argument('--n-head', type=int, default=None,
                    help='注意力头数。默认省略 = 自动推断（RoPE 存档按 '
                         'head_dim 反推；35M=8 / 50M v3_alpha=9 / 100M v3_beta=11）')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()

    global gpt, tokenizer
    gpt, tokenizer = load_model(os.path.abspath(args.ckpt),
                                os.path.abspath(args.tokenizer),
                                n_head=args.n_head)
    print(f'服务启动: http://127.0.0.1:{args.port} （Ctrl+C 停止）')
    uvicorn.run(app, host=args.host, port=args.port)


# 命令行直接跑: python app/server.py ...
if __name__ == '__main__':
    main()

# 被 uvicorn server:app 导入时: 加载默认模型
else:
    gpt, tokenizer = load_model(DEFAULT_CKPT, DEFAULT_TOK)
