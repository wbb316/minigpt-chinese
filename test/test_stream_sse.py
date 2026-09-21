# -*- coding: utf-8 -*-
"""`POST /generate/stream` 的 SSE 帧格式 / 契约测试（FastAPI TestClient，走真实 ASGI 栈）。

与 `test_streaming.py` 的分工：
  · test_streaming.py  —— 增量解码 / 实体 hold-back / stream_ids 等价（不碰 HTTP）
  · 本文件             —— HTTP 层：SSE 分帧、headers、done 帧字段、错误帧、
                          `/generate` 契约未变、两者文本一致

★ 不加载真实权重：import app.server 会加载默认 checkpoint（~3.8s），随后本文件的
  fixture 把 `server.gpt / server.tokenizer` **换成极小模型**，所以每个用例只跑
  几十 ms。换掉是安全的 —— 路由函数是在**调用时**才查模块全局的。
"""
import json
import os
import pickle

import pytest
import torch

try:
    from fastapi.testclient import TestClient
except ImportError:                                   # pragma: no cover
    TestClient = None

pytestmark = pytest.mark.skipif(TestClient is None,
                                reason='需要 fastapi.testclient（httpx）')

import app.server as server                       # noqa: E402
from model.gpt import GPT                         # noqa: E402

# ★ 用**真实分词器**（能从磁盘读到的话）：中文 prompt 在自建小词表上会编码出越界 id
#   （实测 IndexError），那会让本文件变成"测我的夹具"而不是测 SSE。权重不在仓库里
#   （*.pt/*.pkl 被 .gitignore 排除），所以读不到就 skip，不伪装成通过。
TOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result',
                        '150M参数v3+1.5Btokens', 'tokenizer_best.pkl')


def _load_tokenizer():
    if not os.path.exists(TOK_PATH):
        pytest.skip(f'真实分词器不存在（{TOK_PATH}），跳过 SSE 端点测试')
    with open(TOK_PATH, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope='module')
def client():
    """极小模型 + **真实**分词器 塞进 server 的模块全局，再建 TestClient。

    模型换成 2 层小 GPT（随机权重）：本文件只验证帧格式/契约，不验证文本质量，
    所以不需要真权重 —— 但词表必须与分词器一致（8K），否则 id 越界。
    """
    tok = _load_tokenizer()
    torch.manual_seed(0)
    tiny = GPT(vocab_size=tok.vocab_size, n_layer=2, n_head=2, n_embd=32,
               block_size=16, position_encoding='rope', ff_type='relu').eval()
    old = (server.gpt, server.tokenizer)
    server.gpt, server.tokenizer = tiny, tok
    try:
        with TestClient(server.app) as c:
            yield c
    finally:
        server.gpt, server.tokenizer = old


def parse_sse(body: str):
    """把 SSE 响应体解析成 [(event, payload), ...]（与前端同样的分帧规则）。"""
    out = []
    for block in body.split('\n\n'):
        if not block.strip():
            continue
        event = ''
        data = []
        for line in block.split('\n'):
            if line.startswith('event:'):
                event = line[6:].strip()
            elif line.startswith('data:'):
                data.append(line[5:].lstrip(' '))
        if data:
            out.append((event, json.loads('\n'.join(data))))
    return out


BODY = {'prompt': '从前有座山', 'max_tokens': 12,
        'temperature': 0.8, 'top_p': 0.9, 'repetition_penalty': 1.15}


# ---------------------------------------------------------------------------
# 帧格式 / headers
# ---------------------------------------------------------------------------
def test_stream_content_type_and_headers(client):
    with client.stream('POST', '/generate/stream', json=BODY) as r:
        assert r.status_code == 200
        ctype = r.headers['content-type']
        assert ctype.startswith('text/event-stream')
        assert 'charset=utf-8' in ctype.lower()
        # 防中间层缓存/缓冲（nginx 默认会把 SSE 攒起来 → 流式白做）
        assert r.headers['cache-control'] == 'no-cache'
        assert r.headers['x-accel-buffering'] == 'no'


def test_stream_frames_delta_and_done(client):
    r = client.post('/generate/stream', json=BODY)
    assert r.status_code == 200
    frames = parse_sse(r.text)
    assert frames, '应当至少有一帧'

    done = [p for e, p in frames if e == 'done']
    assert len(done) == 1, '必须恰好有一个 done 帧'
    assert isinstance(done[0]['elapsed_ms'], (int, float))
    assert done[0]['elapsed_ms'] >= 0
    assert done[0]['new_tokens'] == BODY['max_tokens']

    deltas = [p['delta'] for e, p in frames if e == '']
    assert deltas, '应当有 data 帧'
    assert all(isinstance(d, str) and d for d in deltas), '不能有空 delta 帧'
    # done 必须是最后一帧
    assert frames[-1][0] == 'done'


def test_stream_first_delta_contains_prompt(client):
    """★ 流式文本 = **prompt + 新生成**（与 /generate 的"全文"语义一致）。

    ⚠️ "首帧含 prompt"的准确含义是：prompt 的 token **排在新 token 前面**喂进同一个
      增量解码器，所以开头一定是 prompt（前几帧逐字长出 prompt）。
      不是"第一帧就吐出整个 prompt" —— 增量解码是逐 token 吐字的。
      （第一版这里写成了"frames[0] 就含整个 prompt"，是测试本身写错。）
    """
    r = client.post('/generate/stream', json=BODY)
    frames = parse_sse(r.text)
    deltas = [p['delta'] for e, p in frames if e == '']
    full = ''.join(deltas)

    assert full.startswith(BODY['prompt'])          # ★ 全文以 prompt 开头
    # 前缀链：第 k 帧结束后，累积文本仍必须是完整全文的前缀
    acc = ''
    for d in deltas:
        acc += d
        assert full.startswith(acc)
    # prompt 是"最早抵达的那部分"：出现它的帧一定在最后一帧之前（除非只生成了一帧）
    first_prompt_frame = next(i for i, d in enumerate(deltas) if BODY['prompt'].startswith(d)
                              and d and BODY['prompt'].startswith(''.join(deltas[:i + 1])))
    assert first_prompt_frame == 0


def test_stream_decodes_exactly_the_ids_generate_ids_would(client, monkeypatch):
    """★ 语义等价（强）：流式拼出的全文 == 把同一串 id 交给整体 decode + clean_text。

    采样没有固定 seed，两次请求的文本必然不同，所以不能直接对比两次请求的输出。
    做法：把流式路由**实际消费的 id 序列**录下来，再走一遍 /generate 的等价代码路径
    （`clean_text(tokenizer.decode(prompt_ids + new_ids))`），逐字比对。
    """
    from model.generation import stream_ids as real_stream_ids
    from app.server import clean_text

    recorded = {}
    orig = server.stream_ids

    def spy(g, ids, max_new_tokens, **kw):
        got = list(real_stream_ids(g, ids, max_new_tokens, **kw))
        recorded['ctx'] = list(ids)[-g.block_size:] + got
        recorded['new'] = got
        return iter(got)

    monkeypatch.setattr(server, 'stream_ids', spy)
    body = dict(BODY, max_tokens=10)
    frames = parse_sse(client.post('/generate/stream', json=body).text)
    streamed = ''.join(p['delta'] for e, p in frames if e == '')
    assert recorded['ctx'], '没有录到 id 序列'

    with torch.no_grad():                       # 整体 decode 路径（与 /generate 同一行代码）
        whole = clean_text(server.tokenizer.decode(recorded['ctx']))
    assert streamed == whole, (
        f'流式全文与整体 decode 不一致!\n streamed={streamed[:80]!r}\n whole={whole[:80]!r}')
    assert frames[-1][1]['new_tokens'] == len(recorded['new']) == 10


def test_stream_text_matches_generate(client):
    """流式与 /generate 的结构等价：都从 prompt 开头、new_tokens 相同。

    ⚠️ 不比对文本内容 —— 随机采样两次必然不同。
    ⚠️ 也不断言 U+FFFD：这里跑的是**随机权重的 2 层小模型**，它的输出本来就是垃圾
      （会生成 id 很低的字节 token，单独/整体 decode 都可能出替换字符）。真实权重下
      U+FFFD=0 由 test_streaming.py 用真实 tokenizer 序列验证。
    """
    body = dict(BODY, max_tokens=16)
    r1 = client.post('/generate', json=body).json()

    frames = parse_sse(client.post('/generate/stream', json=body).text)
    streamed = ''.join(p['delta'] for e, p in frames if e == '')

    assert r1['text'].startswith(body['prompt'])
    assert streamed.startswith(body['prompt'])
    done = [p for e, p in frames if e == 'done'][0]
    assert done['new_tokens'] == r1['new_tokens'] == body['max_tokens']
    # 两者都带"新生成"的部分（而不只是回显 prompt）
    assert streamed != body['prompt']
    assert r1['text'] != body['prompt']


def test_stream_done_new_tokens_counts_generated_only(client):
    """new_tokens 必须只数新生成的 token（不含 prompt），且与请求的 max_tokens 一致。"""
    body = dict(BODY, prompt='从前有座山，山里有座庙', max_tokens=7)
    frames = parse_sse(client.post('/generate/stream', json=body).text)
    done = [p for e, p in frames if e == 'done'][0]
    assert done['new_tokens'] == 7


def test_stream_works_with_over_window_prompt(client):
    """超窗 prompt（远超 block_size=16）：截断 + 多轮重建下流式仍要正常收尾。"""
    body = dict(BODY, prompt='从前有座山，山里有座庙，庙里有个老和尚对小和尚说：' * 6,
                max_tokens=30)
    r = client.post('/generate/stream', json=body)
    assert r.status_code == 200
    frames = parse_sse(r.text)
    assert frames[-1][0] == 'done'
    assert frames[-1][1]['new_tokens'] == 30
    joined = ''.join(p['delta'] for e, p in frames if e == '')
    # 全文 = **完整 prompt**（截断只影响模型能看多远，不影响回给前端的全文）
    assert joined.startswith(body['prompt'])
    # 超窗时每个 token 都要走到"重建"分支：仍必须逐帧到达（不能只在最后一次性吐完）
    assert len([1 for e, _ in frames if e == '']) > 5


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------
def test_invalid_params_return_json_422_not_sse(client):
    """★ 前端"先判 response.ok"的前提：非 2xx 是**普通 JSON 体**（不是 SSE）。"""
    bad = dict(BODY, max_tokens=0)
    r = client.post('/generate/stream', json=bad)
    assert r.status_code == 422
    assert 'text/event-stream' not in r.headers.get('content-type', '')
    data = r.json()
    assert 'detail' in data                       # FastAPI 校验错误体
    detail = json.dumps(data['detail'], ensure_ascii=False)
    assert 'max_tokens' in detail


def test_missing_field_422(client):
    """字段值非法（不是"缺字段"—— max_tokens 有默认值）→ 422 + JSON detail。"""
    r = client.post('/generate/stream', json={'prompt': '你好', 'temperature': 0})
    assert r.status_code == 422
    assert 'detail' in r.json()
    assert 'text/event-stream' not in r.headers.get('content-type', '')


def test_generate_contract_unchanged(client):
    """★ /generate 的字段契约未变（README 把它当契约；前端有"不改字段"的注释）。"""
    r = client.post('/generate', json=dict(BODY, max_tokens=5))
    assert r.status_code == 200
    data = r.json()
    assert set(data) == {'text', 'prompt', 'elapsed_ms', 'new_tokens'}
    assert data['prompt'] == BODY['prompt']
    assert data['new_tokens'] == 5
    assert isinstance(data['text'], str) and data['text'].startswith(BODY['prompt'])
    assert isinstance(data['elapsed_ms'], (int, float))


def test_health_and_home_still_ok(client):
    assert client.get('/health').json() == {'status': 'ok'}
    assert client.get('/').status_code == 200
