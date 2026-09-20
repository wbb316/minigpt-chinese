"""FastAPI 后端：加载训练好的模型，提供中文续写接口。"""
import sys
import os
import time

# ★ 必须先加 sys.path，才能 import 到上级目录的 model 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator
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

# ★ 当前模型摘要：load_model 填充，GET /model-info 读取。
#   为什么用模块级 dict 而不是改 load_model 的返回值？
#   load_model 还被 scratch/ 下的脚本（analyze_redundancy / check_model_ce /
#   compare_generation / test_server_config）以 `gpt, tok = load_model(...)` 调用，
#   ★ 返回值签名一旦改成三元组就会把这些调用方全部弄坏。故新信息走"旁路"。
#   模块导入时（uvicorn server:app 路径）也会被填充 → /model-info 永远有数据。
MODEL_INFO = {}

# ⚠️ max_tokens 的**单一事实来源**：前端滑条默认值由 /model-info 下发，
#    pydantic 默认值也引用这里 —— 历史上前端写死 100、后端默认 80，两边不一致。
#    改这两个常量即可，不要在前端再写死一份。
MAX_TOKENS_DEFAULT = 100
MAX_TOKENS_MAX = 300


def _fmt_params(n: int) -> str:
    """参数量人类可读：148079104 → '148.1M'。"""
    return f'{n / 1e6:.1f}M' if n >= 1e6 else f'{n / 1e3:.1f}K'


def _fmt_ckpt_name(path: str) -> str:
    """展示用的权重名 = 父目录名（result/150M参数v3+1.5Btokens/xxx.pt → 150M参数v3+1.5Btokens）。"""
    return os.path.basename(os.path.dirname(os.path.abspath(path))) or os.path.basename(path)


# ★ 从 checkpoint 自动推断架构（层数/维度/词表/block_size 随训练配置变，这里不用再改）
def load_model(ckpt_path, tok_path, n_head=None, tie_embeddings=True):
    """加载模型，架构从 state_dict 自动推断。

    n_head：**默认 None = 自动推断**。RoPE 的 `rope.cos_cached` 形状是
    `(1, 1, T, head_dim)`，于是 `n_head = n_embd / head_dim`。
    旧的 sinusoidal 存档没有 head_dim 信息 → 自动推断不可用，退回 8，
    需显式 `--n-head`（例：v2 50M 是 9 头）。

    tie_embeddings：**默认 True = 与训练时一致**。所有历史存档（150M/100M/
    50M/35M/20M/6M，已核对 7 个）训练时 token_emb 与 head.weight 就是共享的
    —— 存档里两者数值逐位相同（torch.equal=True），只是保存时两张表都被写进了
    文件（冗余存储），load 进来会变成**两个独立张量**。若不显式 tie：
      · 凭空多出一张嵌入表（150M 白占 ~6.3M 参数的内存）
      · get_num_params() 按 id() 去重失效 → 报出偏大的假参数量
        （150M 会报 154.4M 而非真实的 148.1M）
      · 但**输出不受影响**（两张表数值相同，logits 逐位一致）
    所以显式 tie 是"参数量必须可信"的前提，不是可选项。

    副作用：把推断结果写进模块级 MODEL_INFO（供 GET /model-info 展示）。
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
              position_encoding=pe, ff_type=ff_type, ff_hidden=ff_hidden,
              tie_embeddings=tie_embeddings)
    gpt.load_state_dict(sd)
    gpt.to(device).eval()
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f'加载模型: {n_layer}层/{n_head}头/{n_embd}维, 词表{vocab_size}, '
          f'block_size={block_size}, 位置编码: {pe}, FFN: {ff_type}'
          + (f'(h={ff_hidden})' if ff_hidden else ''))

    # ★ n_params 用 gpt.get_num_params()（按 id() 去重）。
    #   注意：所有历史存档训练时都是 tie（token_emb.weight 与 head.weight 数值相同，
    #   已核对 150M/100M/50M/35M/20M/6M 共 7 个），但保存时两张表都写进了文件。
    #   因此**加载时必须显式 tie_embeddings=True**，否则模型会建出两张独立表、
    #   get_num_params() 去重失效 → 报出偏大的假参数量（150M 会报 154.4M 而非 148.1M）。
    #   这是"参数量必须可信"的前提，不是可选项。
    n_params = gpt.get_num_params()
    # 展示串：一眼看清"现在到底是哪个模型"
    ff_disp = {'swiglu': 'SwiGLU', 'relu': 'ReLU', 'gelu': 'GELU'}.get(ff_type, ff_type.upper())
    pe_disp = {'rope': 'RoPE', 'sinusoidal': 'Sinusoidal'}.get(pe, pe.capitalize())
    display = (f'{n_layer}L / {n_embd}d / {n_head}H · {_fmt_params(n_params)} 参数 · '
               f'{ff_disp}'
               + (f'(h={ff_hidden})' if ff_hidden else '')
               + f' · {pe_disp} · vocab {vocab_size} · block {block_size}')
    MODEL_INFO.update({
        'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd,
        'vocab_size': vocab_size, 'block_size': block_size,
        'position_encoding': pe, 'ff_type': ff_type, 'ff_hidden': ff_hidden,
        'n_params': n_params, 'params_human': _fmt_params(n_params),
        'ckpt_path': os.path.abspath(ckpt_path),
        'ckpt_name': _fmt_ckpt_name(ckpt_path),
        # 由 main() / 导入分支按"是否走过回退"覆写；load_model 本身不知道回退与否
        'is_fallback': MODEL_INFO.get('is_fallback', False),
        'device': str(device),
        'max_tokens_default': MAX_TOKENS_DEFAULT,   # 前端滑条默认值的事实来源
        'max_tokens_max': MAX_TOKENS_MAX,
        'display': display,
    })
    return gpt, tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 默认模型 = **v3_gamma 150M**（20L/768d/12H，head_dim 64，SwiGLU h=2048，vocab 8192，rope）
#   shard0+1+2 共 2.93B token 单轮，val 2.9675（shard1+2 评估空间当前 best，见 docs/EXPERIMENT_LOG.md）
#   ⚠️ 头数 12 / FFN swiglu(h=2048) / block_size 512 全部由 load_model 从 state_dict 自动推断，无需 --n-head
#   ⚠️ 目录名 "150M参数v3+1.5Btokens" 与本次实验（2.93B token）**不符**，是拉取时写的旧名。
#      文件本身已逐项核对：大小 594,288,009 B、20L/768d/12H、去重参数 148,079,104 → 确为 v3_gamma
# 换回其他模型：改这两行即可（或命令行 --ckpt/--tokenizer 覆盖）
#   100M v3_beta (rope, 11 头): result/100M参数v3+2Btokens/
#   50M v3_alpha (rope, 9 头):  log/50M参数_v3_alpha+998Mtokens/
#   35M v2 (sinusoidal, 8 头):  result/35M参数+998Mtokens/
# 归档位置见 result/ 总文件夹结构（docs/MiniGPT_Project_Status.md）
DEFAULT_CKPT = os.path.join(ROOT, 'result', '150M参数v3+1.5Btokens', 'checkpoint_best.pt')
DEFAULT_TOK = os.path.join(ROOT, 'result', '150M参数v3+1.5Btokens', 'tokenizer_best.pkl')

# 权重不入 git（*.pt / *.pkl 被 .gitignore 排除），换机器或清理归档后可能不存在。
# 显式检查 + 回退到 100M，避免 uvicorn 启动时抛一串难懂的 FileNotFoundError。
# ★ 回退事实记进 _DEFAULT_IS_FALLBACK：终端早就打印了，但前端看不见 —— 本次让
#   /model-info 带出去，页面才能显眼提示"你跑的不是默认那个模型"。
_DEFAULT_IS_FALLBACK = False
if not os.path.exists(DEFAULT_CKPT):
    _fb_dir = os.path.join(ROOT, 'result', '100M参数v3+2Btokens')
    print(f'⚠️ 默认权重不存在，回退到 100M v3_beta：\n'
          f'   缺失: {DEFAULT_CKPT}\n'
          f'   回退: {_fb_dir}')
    DEFAULT_CKPT = os.path.join(_fb_dir, 'checkpoint_best.pt')
    DEFAULT_TOK = os.path.join(_fb_dir, 'tokenizer_best.pkl')
    _DEFAULT_IS_FALLBACK = True

# FastAPI 应用（加元信息，让 /docs 的标题不再是光秃秃的 "FastAPI"）
app = FastAPI(
    title='MiniGPT-Chinese 中文续写 API',
    description='从零实现的 Transformer 中文续写服务',
    version='0.3',
)

class Generation(BaseModel):
    prompt: str
    # ★ 默认值与前端滑条同源（MAX_TOKENS_DEFAULT），别再各写一套
    max_tokens: int = MAX_TOKENS_DEFAULT
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.15

    # ★ 范围约束：字段名/语义不变，只是把明显非法的值挡在 422 之前。
    #   实测过 max_tokens=-5 → 静默返回空文本、temperature=999 → 静默走极端采样，
    #   用户看不到任何提示。前端滑条本来就在这些范围内，正常调用不受影响。
    @field_validator('max_tokens')
    @classmethod
    def _chk_max_tokens(cls, v):
        if not 1 <= v <= MAX_TOKENS_MAX:
            raise ValueError(f'max_tokens 必须在 1–{MAX_TOKENS_MAX} 之间，得到 {v}')
        return v

    @field_validator('temperature')
    @classmethod
    def _chk_temperature(cls, v):
        if not 0.0 < v <= 5.0:
            raise ValueError(f'temperature 必须在 (0, 5] 之间，得到 {v}')
        return v

    @field_validator('top_p')
    @classmethod
    def _chk_top_p(cls, v):
        if not 0.0 < v <= 1.0:
            raise ValueError(f'top_p 必须在 (0, 1] 之间，得到 {v}')
        return v

    @field_validator('repetition_penalty')
    @classmethod
    def _chk_rep(cls, v):
        if v <= 0:
            raise ValueError(f'repetition_penalty 必须 > 0，得到 {v}')
        return v

@app.post("/generate")
def generate(request: Generation):
    t0 = time.perf_counter()          # ★ 服务端实测耗时（前端那个含网络往返，仅供参考）
    ids = tokenizer.encode(request.prompt)
    ctx_ids, new_ids = generate_ids(
        gpt, ids,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )
    result = clean_text(tokenizer.decode(ctx_ids))
    # ⚠️ text / prompt 保持原样（别的脚本可能依赖），只**新增**字段
    return {
        "text": result,
        "prompt": request.prompt,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "new_tokens": len(new_ids),
    }

@app.get("/", response_class=HTMLResponse)
def home():
    """返回前端页面"""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(app_dir, 'templates', 'index.html'), encoding='utf-8') as f:
        return f.read()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    """当前实际加载的模型信息（★ 全部从 state_dict 推断，没有任何写死的架构常量）。

    前端用它填副标题 + 模型信息条 + prompt 长度上限，并识别回退状态。
    """
    if not MODEL_INFO:
        # 理论上不会走到（导入/启动时都已加载），留个明确的错误而不是空对象
        return {"error": "模型尚未加载完成"}
    return MODEL_INFO


def main():
    """命令行启动: python app/server.py [--ckpt ...] [--tokenizer ...]"""
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(description='MiniGPT 中文续写 Web 服务')
    ap.add_argument('--ckpt', default=DEFAULT_CKPT, help='模型权重 checkpoint_best.pt')
    ap.add_argument('--tokenizer', default=DEFAULT_TOK, help='分词器 tokenizer_best.pkl')
    ap.add_argument('--n-head', type=int, default=None,
                    help='注意力头数。默认省略 = 自动推断（RoPE 存档按 '
                         'head_dim 反推；35M=8 / 50M v3_alpha=9 / 100M v3_beta=11 / 150M v3_gamma=12）')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()

    global gpt, tokenizer
    gpt, tokenizer = load_model(os.path.abspath(args.ckpt),
                                os.path.abspath(args.tokenizer),
                                n_head=args.n_head)
    # ★ 只有"没传 --ckpt 且确实走了 150M→100M 回退"才算 fallback；
    #   用户显式指定 --ckpt（例如小 checkpoint 快速迭代）时一律 False。
    MODEL_INFO['is_fallback'] = bool(_DEFAULT_IS_FALLBACK
                                     and os.path.abspath(args.ckpt) == os.path.abspath(DEFAULT_CKPT))
    print(f'服务启动: http://127.0.0.1:{args.port} （Ctrl+C 停止）')
    uvicorn.run(app, host=args.host, port=args.port)


# 命令行直接跑: python app/server.py ...
if __name__ == '__main__':
    main()

# 被 uvicorn server:app 导入时: 加载默认模型
else:
    gpt, tokenizer = load_model(DEFAULT_CKPT, DEFAULT_TOK)
    MODEL_INFO['is_fallback'] = _DEFAULT_IS_FALLBACK
