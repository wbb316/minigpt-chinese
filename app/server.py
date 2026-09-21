"""FastAPI 后端：加载训练好的模型，提供中文续写接口。"""
import sys
import os
import json
import time
import codecs
import itertools

# ★ 必须先加 sys.path，才能 import 到上级目录的 model 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, field_validator
import torch
import pickle
import html
import re

from model.gpt import GPT
from model.generation import generate_ids, stream_ids

# 特殊 token：UNK / EOS。流式里**不喂解码器**（见 stream_text_frames 的说明）。
#   与 data/tokenizer.py 的 UNK_ID/EOS_ID 一致；这里重复写成常量是为了让
#   app/ 不依赖 data/ 的 import（server 只 import model 包），一致性由
#   test/test_streaming.py 断言（它会 import data.tokenizer 对比）。
_STREAM_SKIP_IDS = frozenset({0, 1})


def clean_text(text: str) -> str:
    """清理生成的文本：还原 HTML 实体，过滤控制字符。

    模型可能输出 &#8943; 这类 HTML 实体（因为字节级 BPE 会把字符转义），
    这里把它们还原成正常字符，并过滤掉乱码控制字符。
    """
    text = html.unescape(text)  # &#8943; → ⋯
    # 过滤控制字符（除换行/制表符外）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text


# ---------------------------------------------------------------------------
# 流式（SSE）辅助：增量解码 + 清洗后取 diff
# ---------------------------------------------------------------------------
# ★★★ 为什么不能用 `tokenizer.decode([单个新 token])`（坑 1）★★★
#   data/tokenizer.py 的 decode 是 `b''.join(parts).decode('utf-8', 'replace')` ——
#   **整体**解码。而 8192 个 token 里有 458 个单独不是合法 UTF-8（一个汉字的
#   3 个字节可能被切成 2~3 个 token）。逐 token 独立 decode 的后果不只是乱码：
#   '龘'→[779,154]，tok.decode([779]) 得到 U+FFFD，拼起来变成 '�' + '…'；
#   实测某真实序列整体 decode 是 '…王嫣笑嘻嘻'（263 字符、0 个 U+FFFD），
#   naive 逐 token 拼接却在 char36 处把 '王嫣' 变成 '王��'（16 个 U+FFFD）。
#   —— 这是**静默改字**，不是可见的报错，所以必须用**有状态**增量解码器：
#   codecs.getincrementaldecoder 会把跨 token 的半个字符缓冲到下一步再吐。
#   ❌ 不要改 BPETokenizer.decode 去加状态：test/test_tokenizer*.py 依赖现有语义。
def make_incremental_decoder():
    """新建一个有状态 UTF-8 增量解码器（errors='replace'，与 decode 同口径）。"""
    return codecs.getincrementaldecoder('utf-8')(errors='replace')


def _hold_back_incomplete_entity(buf: str) -> str:
    """把 buf 尾部**可能还没写完的 HTML 实体**摘出来（返回要扣住的后缀）。

    ★ 坑 2：clean_text() 不能逐帧调用。`html.unescape('&#8943;')` 整体 → '⋯'，
      但按字节边界拆成 ['&#89', '43;'] 分帧调用 → '&#89' 被解成 'Y'，拼出 'Y43;'。
      实测 1082 个生成字符里 '&' 出现 0 次（概率极低），但这是**静默改字**，
      而且一旦发生就无法在客户端修回来，所以用最低成本的办法堵住：
      尾部 `&` 起始、尚未出现 `;` 的那一段先不发，等后续帧补齐了再一起发。
      · 'x&amp;'      → 完整，扣住 ''
      · 'x&amp'       → 可能是 '&amp;' 的半截 → 扣住 '&amp'（下一帧补 ';' 再发）
      · 'x&'          → 扣住 '&'
      · 'x&a'*10000   → 超过实体长度上限 → 判定不是实体，不扣（退化为按原文发）
    """
    i = buf.rfind('&')
    if i < 0:
        return ''
    tail = buf[i:]
    if ';' in tail:
        return ''
    if len(tail) > 32:          # '&#x1F600;' 最长 10 字符；32 已远超任何合法实体
        return ''
    return tail


def stream_text_frames(vocab, ids_iter, initial_raw: str = ''):
    """(token id → bytes 的 vocab, token id 生成器) → SSE 文本帧生成器。

    逐 token 增量解码 → **累积原文** → 对"整体 clean 后的文本"取 diff 来产帧。

    vocab: `{id: bytes}`（就是 `BPETokenizer.vocab`，data/tokenizer.py:57）。
        显式传入而不是读模块级 tokenizer —— 单测可以直接喂假词表，不需要加载模型。

    产出 `(delta, cumulative_clean_text, n_tokens)`：
      · delta 只在非空时产出（约 7.7% 的 token 天然不吐字 —— 它只是半个汉字，
        正好把这个半字符并进下一帧，不会出现空帧）；
      · cumulative 是"清洗后的全文"（供调用方算 meta），与 /generate 的
        clean_text(tokenizer.decode(ctx_ids)) **逐字相同**（等价性有测试守）。

    首帧为什么含 prompt：调用方把 prompt 的 token id 也喂进来（见 /generate/stream），
    于是累计全文从头就是 prompt + 新生成，与 /generate 的"全文"语义一致，
    前端 text.startswith(prompt) 这类既有逻辑继续成立。
    """
    dec = make_incremental_decoder()
    raw = initial_raw          # 累积的**原始**文本（未清洗）
    sent = ''                  # 已经发出去并完成清洗的部分（sent == 累计清洗前缀）
    n_tokens = 0
    for nid in ids_iter:
        if nid in _STREAM_SKIP_IDS:
            # UNK/EOS 不喂解码器：BPETokenizer.decode 会把它们变成字面量
            # '<unk>'/'<eos>'，流式里出现这种字样是明显的穿帮。
            # 实测 600+ 生成 token 里 0 次特殊 token（train/train.py 明确写
            # "纯续写训练未用 EOS"），属廉价防御。
            continue
        piece = dec.decode(vocab.get(nid, b''), final=False)
        n_tokens += 1
        if piece:
            raw += piece
        # 尾部半截实体先扣住（见 _hold_back_incomplete_entity）
        hold = _hold_back_incomplete_entity(raw)
        body = raw[:len(raw) - len(hold)] if hold else raw
        cleaned = clean_text(body)
        if not cleaned.startswith(sent):
            # 理论上不会发生（扣住半截实体后 clean_text 单调不减）。
            # 真发生了说明清洗规则把已发内容改了 —— 宁可整段重发也不静默丢字。
            sent = ''
        delta = cleaned[len(sent):]
        sent = cleaned
        if delta:
            yield delta, sent, n_tokens
    # 收尾：把增量解码器残留的半个字符吐出来（errors='replace' → U+FFFD 兜底），
    # 再补发被扣住的尾部（它已经不可能再被补全了）。
    tail = dec.decode(b'', final=True)
    if tail:
        raw += tail
    cleaned = clean_text(raw)
    if not cleaned.startswith(sent):
        sent = ''
    delta = cleaned[len(sent):]
    if delta:
        yield delta, cleaned, n_tokens

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
# 上限 = **context 总长**（block_size），不是拍脑袋的 300。
#   ⚠️ 2026-09-20 修正：原值 300 是个任意上限，会让用户明明有 512 的窗口却只能生成 300。
#   语义澄清：这个上限是「滑动窗口每次能看到的总 token 数」的保守代理值，
#   **不是**「必须留出这么多余量」—— 前端会用 block_size 做动态计算：
#       可生成上限 = block_size − 当前输入字数
#   所以这里只要给一个不会误拦合法请求的天花板即可（block_size 本身）。
#   本机 50M/150M 存档的 block_size 都是 512，故取 512；换更长 context 的模型时
#   应同步调整（真正权威的值是 checkpoints 里的 block_size）。
MAX_TOKENS_MAX = 512


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


def _sse(event: str, payload: dict) -> str:
    """一帧 SSE。event 为空串 → 默认 message 事件（前端 onmessage 收）。

    ⚠️ JSON 里可能有中文/换行：json.dumps 默认 ensure_ascii=True 会把中文转成
      \\uXXXX —— 合法且前端 JSON.parse 后还原，但会让"肉眼抓包"很难读。
      实测体积也只涨 ~10%（中文 3 字节 → 6 个 ASCII 字符），故本可以关掉。
      **但这里保持 ensure_ascii=True（默认）**：turn 的多字节字符不会出现在
      SSE 帧的**行内**，从而不可能被"按行切分"的中间层（代理/日志）切坏。
    """
    head = f'event: {event}\n' if event else ''
    return head + 'data: ' + json.dumps(payload) + '\n\n'


@app.post("/generate/stream")
def generate_stream(request: Generation):
    """流式续写（SSE）：逐 token 把新文字推给前端，像豆包那样"逐步出现"。

    ★ 为什么是**同步 def** 而不是 async def：
      starlette/responses.py 对非 AsyncIterable 的内容自动走
      `iterate_in_threadpool(content)` —— 每次 `next()` 用
      `anyio.to_thread.run_sync` 跑在**工作线程**里。所以生成器的每一步都在线程池
      执行，事件循环不会被 26ms/token 的 torch 前向堵住（实测流式进行中并发
      /health 延迟 4.2ms）。
      ⚠️ 绝不要写成 `async def` 里直接调 sample() —— 那会把整个事件循环锁死，
        所有并发请求（含 /health）一起卡住。

    ★ 协议（前端按 \\n\\n 分帧）
      · `data: {"delta": "..."}`  —— 文本增量（**开头是 prompt**，见下）
      · `event: done`  + `data: {"elapsed_ms":..., "new_tokens":...}`
      · `event: error` + `data: {"message":...}`

    ★ 文本语义与 /generate 完全一致：`text` 是**全文**（prompt + 新生成，来自
      tokenizer.decode(ctx_ids)）。做法是把 prompt 的 token id **串在新 token
      前面**喂进同一个增量解码器（itertools.chain），于是流式拼出来的全文与
      /generate 的 text 逐字相同 → 前端最终 DOM 不变，且 `text.startsWith(prompt)`
      （细栏字数口径）等既有逻辑继续成立。
      ⚠️ 措辞要准：prompt 是**逐 token 逐帧长出来**的，不是"第一帧就吐出整个 prompt"
        （增量解码就是逐 token 吐字）。第一版注释写成了后者，测试跟着写错，已修。

    ★ abort 语义（与前端文案配套，改文案时必须同步改这里）：
      客户端断开 → starlette 在下一步 `next()` 时取消这个线程池迭代 →
      **生成器在 yield 边界就停了**（实测日志 5/20 步后再无推进）。
      所以"/generate 那种'服务端照旧算完'"的说法对流式**不成立**；
      但已经发出去的帧是**有效结果**（不是没算完的垃圾）—— 模型每步产出就是那一步的
      最终答案，后续步骤不会回头改写它。
    """
    t0 = time.perf_counter()

    def gen():
        # 这一层 try 覆盖"生成 + 发帧"全过程：任何异常都要变成一帧 error，
        # 而不是让连接无声断掉（前者前端能显示原因，后者只能显示网络错误）。
        n_frames = 0        # 已推给客户端的帧对应的 token 数（含 prompt 段）
        n_gen = 0           # 模型**真正算出来**的 token 数（从生成器拿到几个就是几个）
        ids = []
        ok = False
        try:
            ids = tokenizer.encode(request.prompt)
            # ★★ 必须把 prompt 的 id **串在新 token 前面**喂进解码器 ★★
            #   `stream_ids` 只 yield **新生成**的 token（不是 ctx），prompt 不在里面；
            #   而 /generate 的 text 是「prompt + 新生成」的全文。漏掉这一步 →
            #   流式拼出的文本**没有开头**，前端的 startsWith(prompt) 立刻失效
            #   （实测就是这样被 e2e 测试抓到的）。
            #   ⚠️ itertools.chain 是**惰性**的：prompt 段没有额外内存/时间开销，
            #      而且生成器照旧是"逐步消费"（客户端断开时不会先跑完 prompt 段）。
            def counted():
                """包一层只为数"真正算了几步"（与"推了几帧"区分开，见 finally 的日志）。"""
                nonlocal n_gen
                for tok_id in stream_ids(
                        gpt, ids,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        repetition_penalty=request.repetition_penalty):
                    n_gen += 1
                    yield tok_id

            ids_iter = itertools.chain(ids, counted())
            frames = stream_text_frames(tokenizer.vocab, ids_iter)
            for delta, _full, n in frames:
                n_frames = n
                yield _sse('', {'delta': delta})
            # ⚠️ new_tokens 必须等于 /generate 的 `len(new_ids)`，而 `n_gen` 就是它
            #   （counted() 对 stream_ids 吐出的**每一个** token 加一）——
            #   所以这里**直接用 n_gen**，不要拿"发了几帧"去反推：
            #   `stream_text_frames` 的第三个返回值数的是**喂进解码器的** token，
            #   它会跳过 UNK/EOS（见 _STREAM_SKIP_IDS），尾部不吐字的 token 也不增帧。
            #   实测 600+ token 没采到过特殊 token，所以两条口径平时一样；
            #   但"平时一样"不该写进契约 —— 用 n_gen 是**构造上**精确相等。
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            yield _sse('done', {
                'elapsed_ms': elapsed,
                'new_tokens': n_gen,
            })
            ok = True
        except GeneratorExit:
            # 客户端断开时 starlette 关闭生成器会抛到这里 —— 不是错误，
            # 不要发 error 帧（连接早没了），安静退出即可。
            raise
        except Exception as e:                       # noqa: BLE001（要的就是"全都兜住"）
            yield _sse('error', {
                'message': f'{type(e).__name__}: {e}',
                'elapsed_ms': round((time.perf_counter() - t0) * 1000, 1),
            })
        finally:
            # 正常跑完打"完成"；生成器**被关闭**时打"提前结束"，并报出实际算了多少步。
            #   `已算` 是**模型真正算出来的** token 数（< max_tokens 就说明停了）。
            # ⚠️ 但**别把这行日志当成"客户端断开就停"的证据**（实测教训）：
            #   客户端断开时 starlette 只是**不再 next()** 这个生成器，并不保证马上
            #   close() 它 —— 被遗弃的生成器会挂着，`finally` 可能几秒到几分钟后才跑
            #   （实测 2.8s ~ 266s，且与后续连接活动/GC 时机有关；300s 静默期内甚至
            #    从未见过它执行）。所以"提前结束"这一行在真实中止场景下**常常看不到**。
            #   ✅ "真的停了"的正确判据是 **CPU**：中止后 300s 采样，进程 CPU 增量
            #      0.3%（纯空转噪声），且 `已算` 恰好等于客户端已消费的 token 数
            #      —— 一步都没多算。
            #   ⚠️ flush=True 必须加：stdout 重定向到文件/管道时是**块缓冲**的，
            #      不 flush 的话这条日志会卡在缓冲区里（实测排障时被坑过）。
            print(f'[stream] {"完成" if ok else "提前结束"} · 请求上限 {request.max_tokens} · '
                  f'prompt {len(ids)} token · 已算 {n_gen} token · 已发 {n_frames} 帧 · '
                  f'{round((time.perf_counter() - t0) * 1000, 1)} ms', flush=True)

    return StreamingResponse(
        gen(),
        media_type='text/event-stream; charset=utf-8',
        headers={
            # 防中间层缓存/缓冲掉增量（nginx 默认会把 SSE 攒起来，那就白流了）
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )

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
