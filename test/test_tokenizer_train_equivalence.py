# -*- coding: utf-8 -*-
"""tokenizer trainer 等价性测试：legacy vs fast 必须逐 merges/vocab 一致。

覆盖 10 类语料（任务书第六节要求），另加大规模真实语料测试
（vocab 3256/6144，保证 heap 过期路径被充分触发）。
"""
import random

import pytest

from data.tokenizer import BPETokenizer

ZH = ('她抬起头望向窗外的天空心里想着那个人的模样，夜色渐渐深了。'
      '他走进教室看见黑板上的字迹模糊不清。'
      '不知道为什么总觉得今天会有什么事情发生。')

CASES = [
    ('中文短文本', '你好世界，这是测试。'),
    ('英文', 'The quick brown fox jumps over the lazy dog.'),
    ('数字', '1234567890 0987654321 555-1234 3.14159'),
    ('标点', '！？。，、；：“”‘’（）【】《》——…'),
    ('空白', '   \t\n  a  b \n\n  c d  \t '),
    ('中英数混合', '他买了 iPhone 15 和 MacBook，花了 8888 元！'),
    ('重复字符串', 'a' * 2000),
    ('重复中文', '哈' * 2000),
    ('多段落', '第一段内容。\n\n第二段内容，含"引号"。\n\n第三段 tail。\n' * 30),
    ('中文长文', ZH * 100),
]


@pytest.mark.parametrize('name,text', CASES, ids=[c[0] for c in CASES])
def test_legacy_fast_equivalence_small(name, text):
    t1 = BPETokenizer(vocab_size=1024)
    t1.train(text, 'legacy')
    t2 = BPETokenizer(vocab_size=1024)
    t2.train(text, 'fast')
    assert t1.merges == t2.merges, f'[{name}] merges 不一致'
    assert t1.vocab == t2.vocab, f'[{name}] vocab 不一致'
    # encode 也一致
    probe = text[:300]
    assert t1.encode(probe) == t2.encode(probe), f'[{name}] encode 不一致'


def test_legacy_fast_equivalence_random_text():
    """随机文本（可控 seed），覆盖 heap 过期路径。"""
    rng = random.Random(42)
    pool = list('她他你我天地人一二三四五 abcXYZ0123！？，。\n\t ')
    text = ''.join(rng.choice(pool) for _ in range(20000))
    t1 = BPETokenizer(vocab_size=1024)
    t1.train(text, 'legacy')
    t2 = BPETokenizer(vocab_size=1024)
    t2.train(text, 'fast')
    assert t1.merges == t2.merges
    assert t1.vocab == t2.vocab


@pytest.mark.parametrize('vocab', [1024, 3256], ids=['v1024', 'v3256'])
def test_legacy_fast_equivalence_full_merges(vocab):
    """中文大文本 + 较大 vocab：触发完整 merge 序列与 heap 过期。"""
    text = ZH * 400          # ~3 万字符
    t1 = BPETokenizer(vocab_size=vocab)
    t1.train(text, 'legacy')
    t2 = BPETokenizer(vocab_size=vocab)
    t2.train(text, 'fast')
    assert len(t1.merges) == len(t2.merges) > 0
    assert t1.merges == t2.merges
    assert t1.vocab == t2.vocab
    # roundtrip
    ids = t2.encode(text)
    assert t2.decode(ids) == text


def test_decode_roundtrip_special_tokens():
    """特殊 token 语义不变（<unk>/<eos> 名字显示）。"""
    t = BPETokenizer(vocab_size=512)
    t.train('测试文本。', 'fast')
    ids = t.encode('测试文本。')
    assert t.decode(ids) == '测试文本。'
    # 特殊 id 直接 decode
    assert t.decode([0]) == '<unk>'
    assert t.decode([1]) == '<eos>'
    # 越界 id 跳过不崩
    assert t.decode([99999]) == ''


def test_backend_selector():
    """train(backend=...) 选择器工作正常，默认 legacy。"""
    text = '他走进教室，看见黑板上写满了公式。' * 50
    a = BPETokenizer(vocab_size=512)
    a.train(text)                        # 默认 legacy
    b = BPETokenizer(vocab_size=512)
    b.train(text, 'legacy')
    c = BPETokenizer(vocab_size=512)
    c.train(text, 'fast')
    assert a.merges == b.merges
    assert b.merges == c.merges
