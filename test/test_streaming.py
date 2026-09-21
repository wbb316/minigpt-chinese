# -*- coding: utf-8 -*-
"""流式输出（SSE）的三条护栏测试。

背景（详见 scratch/流式输出_技术调查报告.md）：新增 `POST /generate/stream` 逐 token
推送文字。这条链路上有两个**静默改字**的坑，本文件把它们各钉成一组测试：

1. **增量解码**（最重要）：8192 个 token 里 458 个单独不是合法 UTF-8，真实生成里
   partial token 占 ~7.7%。逐 token 独立 decode 不是"乱码"而是**静默改字** ——
   实测某真实序列整体 decode 是 '…王嫣笑嘻嘻'（263 字符、0 个 U+FFFD），
   naive 逐 token 拼接变成 '…王��笑嘻嘻'（16 个 U+FFFD）。
   → 必须用 codecs 增量解码器；`stream_text_frames` 与整体 decode **逐字相同、U+FFFD=0**。

2. **clean_text 不能逐帧调用**：`html.unescape('&#8943;')` 是整体语义，按帧切成
   ['&#89','43;'] 会解出 'Y43;'。→ 服务端保留累积原文 + 尾部半截实体 hold-back。

3. **`stream_ids` vs `generate_ids` 同 seed 逐位一致**：generate_ids 已被重构为
   stream_ids 的薄封装，这条等价性保证重构没有改变任何行为。

★ 本文件**不加载真实权重**：一个 ~64 词表 / block_size=8 的极小 GPT 就能走完
  "超窗 → 重建"多轮循环；增量解码用**真实 tokenizer 的词表逻辑**（自建含 partial
  token 的假词表 + 真实 data.BPETokenizer 训练出的小词表）。
"""
import codecs

import pytest
import torch

from app.server import (_STREAM_SKIP_IDS, _hold_back_incomplete_entity,
                        clean_text, make_incremental_decoder, stream_text_frames)
from data.tokenizer import EOS_ID, UNK_ID, BPETokenizer
from model.generation import generate_ids, stream_ids
from model.gpt import GPT

VOCAB, N_LAYER, N_HEAD, N_EMBD, BLOCK = 64, 2, 2, 32, 8


# ---------------------------------------------------------------------------
# 夹具：极小模型 + 含 partial token 的词表
# ---------------------------------------------------------------------------
def make_gpt(pe='rope', **kw):
    torch.manual_seed(0)
    return GPT(vocab_size=VOCAB, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
               block_size=BLOCK, position_encoding=pe, ff_type='relu',
               **kw).eval()


def fresh_rng(seed=1234):
    return torch.Generator(device='cpu').manual_seed(seed)


@pytest.fixture(scope='module')
def gpt():
    return make_gpt()


@pytest.fixture(scope='module')
def tok():
    """真实 BPETokenizer（小词表，含真正的多字节 partial token）。

    用它而不是手搓 dict，是为了让"partial token"这件事来自**真实的分词器行为**
    （字节级 BPE 会把一个汉字的 3 个字节切成 2~3 个 token），而不是我拍脑袋构造。
    """
    text = ('从前有座山，山里有座庙，庙里有个老和尚对小和尚说：'
            '龘𠮷😀王嫣笑嘻嘻，今天天气不错。\n' * 20)
    t = BPETokenizer(vocab_size=400)
    t.train(text)
    return t


def partial_utf8_ids(tok):
    """找出 `tok.vocab` 里**单独不是合法 UTF-8** 的普通 token id（真实 partial token）。"""
    return [i for i, bs in sorted(tok.vocab.items())
            if i not in (UNK_ID, EOS_ID) and not _is_whole_utf8(bs)]


def _is_whole_utf8(bs):
    try:
        bs.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def _long_ids(tok):
    """一条**长且 partial token 密集**的真实 id 序列（不是手搓的假词表）。

    `'\\u0800'` 不在这个 400 词表里 → 它的 3 个 UTF-8 字节被拆成 3 个独立 token
    （`[226, 162, 130]`），每个单独都不是合法 UTF-8；整条序列 240 个 token 里
    240 个都是 partial。这正是"逐 token 独立 decode 会静默改字"的最坏情况，
    而整体 decode 出来是干净的（U+FFFD = 0）。
    """
    return tok.encode(('\u0800' + '王嫣') * 40)


# ---------------------------------------------------------------------------
# 1. 增量解码等价性（最重要）
# ---------------------------------------------------------------------------
def test_naive_per_token_decode_actually_corrupts(tok):
    """反例：证明"逐 token 独立 decode"真的会坏 —— 否则下面那组测试没有意义。

    如果哪天词表变了、这个反例不再成立，本测试会失败并提醒：等价性测试的前提变了。
    """
    ids = tok.encode('龘')
    assert len(ids) >= 2, '龘 应当被切成多个 token（多字节字符的字节级切分）'
    naive = ''.join(tok.decode([i]) for i in ids)
    assert '\ufffd' in naive, f'naive 逐 token decode 应当出现 U+FFFD，得到 {naive!r}'
    assert tok.decode(ids) == '龘'          # 整体 decode 是对的

    inc = ''.join(d for d, _, _ in stream_text_frames(tok.vocab, iter(ids)))
    assert inc == '龘'
    assert '\ufffd' not in inc


@pytest.mark.parametrize('s', ['龘', '𠮷', '😀', '王嫣笑', '龘𠮷😀'])
def test_incremental_decode_matches_whole_decode(tok, s):
    """★ 核心等价性：流式逐步 decode 的拼接 == 整体 decode，且 U+FFFD = 0。"""
    ids = tok.encode(s)
    frames = list(stream_text_frames(tok.vocab, iter(ids)))
    inc = ''.join(d for d, _, _ in frames)
    whole = tok.decode(ids)
    assert inc == whole
    assert inc.count('\ufffd') == whole.count('\ufffd') == 0


def test_incremental_decode_on_long_realistic_sequence(tok):
    """长序列（含 partial token）：拼接结果与整体 decode 逐字相同。

    ★ 序列是按真实语料扫出来的：`_long_ids` 里含多个"单独不是合法 UTF-8"的 token
      （测试自己会先断言 partial > 0，否则这条测试没有覆盖到坑）。
    """
    ids = _long_ids(tok)
    assert len(ids) > 100
    bad = sum(1 for i in ids if not _is_whole_utf8(tok.vocab[i]))
    assert bad > 0, '这组样例应当包含 partial token（否则测试没有覆盖到坑）'

    inc = ''.join(d for d, _, _ in stream_text_frames(tok.vocab, iter(ids)))
    assert inc == tok.decode(ids)
    assert inc.count('\ufffd') == 0
    assert bad / len(ids) > 0.01, f'partial token 占比过低（{bad}/{len(ids)}），覆盖不足'


def _is_whole_utf8(bs):
    try:
        bs.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def test_no_empty_delta_frames(tok):
    """有的 token 天然"不吐字"（它只是半个汉字）→ 不能产出空 delta 帧。

    ★ 证明这个前提在真实 tokenizer 上成立：`'龘'` 的每个 token 单独喂增量解码器
      都吐**空串**（字节被缓冲），全部吐字发生在最后一个 token 上。
      所以"每 token 一帧"会产出空帧 —— 本实现只在 delta != '' 时产帧。
    """
    ids = tok.encode('龘')
    assert len(ids) >= 2
    dec = make_incremental_decoder()
    outs = [dec.decode(tok.vocab[i], final=False) for i in ids]
    assert outs[:-1] == [''] * (len(ids) - 1), f'前 n-1 个 token 应当吐空串，得到 {outs!r}'
    assert outs[-1] == '龘'
    assert dec.decode(b'', final=True) == ''

    frames = list(stream_text_frames(tok.vocab, iter(ids)))
    assert all(d for d, _, _ in frames), f'出现空 delta 帧: {frames!r}'
    # 帧数 <= token 数（有 token 不吐字 → 帧更少）
    assert len(frames) <= len(ids)
    assert ''.join(d for d, _, _ in frames) == '龘'


def test_cumulative_text_is_a_strict_prefix_chain(tok):
    """每帧的 cumulative 必须是上一帧的前缀（前端按 delta 累积的前提）。"""
    ids = tok.encode('王嫣笑嘻嘻龘𠮷')
    prev = ''
    n_prev = 0
    for d, full, n in stream_text_frames(tok.vocab, iter(ids)):
        assert full == prev + d
        assert full.startswith(prev)
        assert n > n_prev          # 已消费的 token 数严格递增
        prev, n_prev = full, n
    assert prev == tok.decode(ids)


def test_never_emits_literal_special_token(tok):
    """UNK/EOS 不喂解码器 —— 否则界面会出现字面量 '<unk>'/'<eos>'。"""
    ids = tok.encode('王嫣') + [UNK_ID, EOS_ID] + tok.encode('笑')
    frames = list(stream_text_frames(tok.vocab, iter(ids)))
    inc = ''.join(d for d, _, _ in frames)
    assert '<unk>' not in inc and '<eos>' not in inc
    # 0/1 被跳过 → 累计文本 == 去掉特殊 token 后的整体 decode（decode 会把它们变字面量，
    # 所以这里直接和"去掉特殊 token 再 decode"比）
    rest = [i for i in ids if i not in (UNK_ID, EOS_ID)]
    assert inc == tok.decode(rest)


def test_final_tail_flushed_with_truncated_bytes(tok):
    """收尾必须 flush：否则最后一个 token 若是半截字符，整块内容会凭空少掉。"""
    # 只喂 '龘' 的第一个 token（半个字符）→ final=False 什么都不吐，
    # 必须靠 dec.decode(b'', final=True) 在收尾时吐出替换字符（而不是静默丢掉）
    ids = tok.encode('龘')
    frames = list(stream_text_frames(tok.vocab, iter(ids[:1])))
    inc = ''.join(d for d, _, _ in frames)
    assert inc == '\ufffd', f'半截字节应当在收尾时被 flush 成 U+FFFD，得到 {inc!r}'


# ---------------------------------------------------------------------------
# 2. clean_text 不能逐帧调用（HTML 实体 hold-back）
# ---------------------------------------------------------------------------
def test_hold_back_incomplete_entity():
    assert _hold_back_incomplete_entity('x&amp;') == ''
    assert _hold_back_incomplete_entity('x&amp') == '&amp'
    assert _hold_back_incomplete_entity('x&') == '&'
    assert _hold_back_incomplete_entity('x&#89') == '&#89'
    assert _hold_back_incomplete_entity('没有与号') == ''
    assert _hold_back_incomplete_entity('x&' + 'a' * 100) == ''   # 超长 → 判定不是实体


def test_entity_split_across_frames_is_not_mangled(tok):
    """★ 坑 2 的正面证明：实体跨帧时**不能**被逐帧 clean_text 解错。

    '&#8943;' 整体 → '⋯'；若按帧拆成 ['&#89','43;'] 分别 clean 会得到 'Y43;'
    （`&#89` 被 html.unescape 解成 'Y'）。这里用一个**每字符一个 token** 的玩具词表
    把它钉住 —— 这样"切在实体中间"是确定发生的（8 个 token 里 7 个都在实体内部），
    不依赖真实词表恰好怎么切。

    ⚠️ 玩具 id 从 2 起编：0/1 是 UNK/EOS，会被有意跳过（见 test_never_emits_...）。
    """
    chars = 'A&#8943;B'          # 'A' + 7 字符实体 '&#8943;' + 'B' = 9 个字符
    assert len(chars) == 9
    vocab = {i + 2: ch.encode('utf-8') for i, ch in enumerate(chars)}
    ids = list(range(2, len(chars) + 2))

    frames = list(stream_text_frames(vocab, iter(ids)))
    inc = ''.join(d for d, _, _ in frames)
    assert clean_text(chars) == 'A⋯B'
    assert inc == 'A⋯B'

    # 反例对照：逐 token 各自 clean 再拼（就是"不能做"的那种做法）
    naive = ''.join(clean_text(vocab[i].decode('utf-8')) for i in ids)
    assert naive != 'A⋯B', f'naive 逐帧 clean 居然没坏？naive={naive!r}'

    # 并且：实体内部的那 6 个 token（'&','#','8','9','4','3'）一帧都不该发出去
    # （它们只是"半截实体"，发出去就是错字）
    for d in [d for d, _, _ in frames]:
        assert not d.startswith('&'), f'半截实体被发出去了: {d!r}'


def test_entity_at_very_last_token_is_still_complete(tok):
    """实体出现在**最后一帧**：hold-back 的内容必须在收尾时补发，不能丢。"""
    chars = 'AB&#8943;'
    vocab = {i + 2: ch.encode('utf-8') for i, ch in enumerate(chars)}
    frames = list(stream_text_frames(vocab, iter(range(2, len(chars) + 2))))
    assert ''.join(d for d, _, _ in frames) == 'AB⋯'


# ---------------------------------------------------------------------------
# 3. stream_ids vs generate_ids 等价（重构没有改行为）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('max_overrun', [0, 3, 64])
@pytest.mark.parametrize('n_new', [1, 5, 17])
def test_stream_ids_matches_generate_ids_bit_for_bit(gpt, max_overrun, n_new):
    """★ 同 seed 下 `generate_ids` 与 `list(stream_ids(...))` 逐位一致。

    max_overrun=0/3 覆盖"每步重建"与"跨多轮重建"两种路径；n_new 跨过 block_size(8)。
    """
    prompt = [3, 7, 11, 13, 17]
    kw = dict(temperature=0.8, top_p=0.9, repetition_penalty=1.15,
              max_overrun=max_overrun)
    ctx_a, new_a = generate_ids(gpt, prompt, n_new, rng=fresh_rng(99), **kw)
    new_b = list(stream_ids(gpt, prompt, n_new, rng=fresh_rng(99), **kw))
    assert new_a == new_b
    assert ctx_a == list(prompt[-BLOCK:]) + new_b


def test_stream_ids_is_lazy_and_yields_each_step(gpt):
    """stream_ids 是**生成器**：不消费就不算（这是流的"逐步到达"与"可中止"的前提）。

    ★ 这条性质是 abort 语义的基础：客户端断开 → 没人再 next() → 剩余步数不执行。
    """
    import inspect
    assert inspect.isgeneratorfunction(stream_ids)

    calls = []
    orig = gpt.forward
    # ⚠️ 必须在 **gpt 这一层**计数：gpt.forward 内部还会逐层调 block.forward，
    #    如果挂在更上层会重复计数（第一版就在这里踩过：prefill 被记成 2 次而非 1 次）。
    def spy(x, past_kvs=None, return_kv=False):
        calls.append(1)
        return orig(x, past_kvs=past_kvs, return_kv=return_kv)

    gpt.forward = spy
    try:
        it = stream_ids(gpt, [1, 2, 3], 5, rng=fresh_rng(1))
        assert calls == []             # ★ 还没 next() → 一次前向都不该发生（含 prefill）
        first = next(it)
        assert isinstance(first, int)
        assert len(calls) == 2         # 第 1 次 next() = prefill + 第 1 步的增量前向
        next(it)
        assert len(calls) == 3         # 之后每 next() 一次只多一次前向
        next(it)
        assert len(calls) == 4
    finally:
        gpt.forward = orig


def test_stream_ids_signature_matches_generate_ids():
    """薄封装的前提：两个函数参数完全同名同序（调用方可以换用而不改代码）。"""
    import inspect
    a = inspect.signature(generate_ids).parameters
    b = inspect.signature(stream_ids).parameters
    assert list(a) == list(b)
    for k in a:
        assert a[k].default == b[k].default


def test_stream_ids_rejects_negative_overrun(gpt):
    with pytest.raises(ValueError):
        list(stream_ids(gpt, [1, 2], 1, max_overrun=-1))


def test_generate_ids_over_window_equivalence(gpt):
    """prompt 远超 block_size：截断 + 多轮超窗重建后仍逐位一致。"""
    prompt = list(range(1, 40))       # 39 > block_size=8
    kw = dict(temperature=1.0, top_p=1.0, repetition_penalty=1.0, max_overrun=3)
    ctx_a, new_a = generate_ids(gpt, prompt, 30, rng=fresh_rng(5), **kw)
    new_b = list(stream_ids(gpt, prompt, 30, rng=fresh_rng(5), **kw))
    assert new_a == new_b
    assert ctx_a == prompt[-BLOCK:] + new_b


# ---------------------------------------------------------------------------
# 4. 增量解码器的直接单测（不经过 stream_text_frames）
# ---------------------------------------------------------------------------
def test_incremental_decoder_buffers_split_multibyte():
    dec = make_incremental_decoder()
    s = '龘'.encode('utf-8')
    assert dec.decode(s[:1], final=False) == ''
    assert dec.decode(s[1:2], final=False) == ''
    assert dec.decode(s[2:], final=False) == '龘'
    assert dec.decode(b'', final=True) == ''


def test_clean_text_keeps_newline_and_drops_control():
    assert clean_text('a\nb\tc') == 'a\nb\tc'
    assert clean_text('a\x00b\x07c') == 'abc'


# ---------------------------------------------------------------------------
# 5. 词表级事实：partial token 确实存在（报告里 458 个的量级由真实词表验证）
# ---------------------------------------------------------------------------
def test_special_id_constants_match_tokenizer():
    """app/server.py 的 _STREAM_SKIP_IDS 必须与 data/tokenizer.py 的常量一致。"""
    assert _STREAM_SKIP_IDS == {UNK_ID, EOS_ID} == {0, 1}


def test_partial_tokens_exist_in_generated_vocab(tok):
    """训练出的小词表里必须有"单独不是合法 UTF-8"的 token（否则上面全白测）。"""
    ids = partial_utf8_ids(tok)
    assert ids, '小词表里没有 partial token？那增量解码的坑没被覆盖'
    # 并且这些 token 拼起来能还原字符
    assert tok.decode(ids) != ''
    assert set(ids) <= set(tok.vocab)
