# -*- coding: utf-8 -*-
"""encode pipeline 正确性测试（任务书第 5 节）。

验证:
1. 优化后 encode() 输出与旧实现逐 token 一致（各类语料）
2. 单进程整段 encode == 多进程分片 encode 按序拼接（逐 token）
3. uint16 分片缓存读取 == 整段 encode
4. memmap 随机切片正确（含跨分片）
5. decode(encode(text)) roundtrip
6. 几 MB 真实 webnovel 文本
"""
import os
import random
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import BPETokenizer  # noqa: E402
from data.token_cache import encode_to_cache, ShardMemmap  # noqa: E402
from data.dataset import PackedDataset, TextDataset  # noqa: E402

ZH = ('她抬起头望向窗外的天空心里想着那个人的模样，夜色渐渐深了。'
      '他走进教室看见黑板上的字迹模糊不清。'
      '不知道为什么总觉得今天会有什么事情发生。')

CASES = [
    ('中文', '你好世界，这是测试。' * 100),
    ('英文', 'The quick brown fox jumps over the lazy dog. ' * 60),
    ('数字', '1234567890 0987654321 555-1234 3.14159 ' * 50),
    ('标点', '！？。，、；：“”‘’（）【】《》——…' * 80),
    ('空白换行', '  \t\n a  b \n\n  c d  \t ' * 60),
    ('中英数混合', '他买了 iPhone 15 和 MacBook，花了 8888 元！' * 50),
    ('重复字符', 'a' * 3000),
    ('重复中文', '哈' * 3000),
    ('多段落', '第一段内容。\n\n第二段内容，含"引号"。\n\n第三段 tail。\n' * 60),
    ('中文长文', ZH * 200),
]


@pytest.fixture(scope='module')
def tokenizer():
    tok = BPETokenizer(vocab_size=1024)
    tok.train((ZH + 'abc 123!? ') * 50, 'fast')
    return tok


@pytest.mark.parametrize('name,text', CASES, ids=[c[0] for c in CASES])
def test_encode_equivalence_optimized_vs_legacy(name, text, tokenizer):
    """优化后 encode() == 旧实现（内嵌 reference 实现对照）。"""
    tok = tokenizer

    # legacy reference（与旧 tokenizer.py encode 完全相同的逻辑）
    def legacy_encode(t, text):
        out = []
        for chunk in __import__('data.tokenizer',
                                fromlist=['PRETOK_RE']).PRETOK_RE.findall(text):
            raw = chunk.encode('utf-8', errors='ignore')
            ids = [b + __import__('data.tokenizer',
                                  fromlist=['NUM_SPECIAL']).NUM_SPECIAL
                   for b in raw]
            ids = [i if i in t.vocab else
                   __import__('data.tokenizer', fromlist=['UNK_ID']).UNK_ID
                   for i in ids]
            stack = []
            for token in ids:
                stack.append(token)
                while len(stack) >= 2:
                    nid = t.merges.get((stack[-2], stack[-1]))
                    if nid is None:
                        break
                    stack.pop()
                    stack.pop()
                    stack.append(nid)
            out.extend(stack)
        return out

    assert tok.encode(text) == legacy_encode(tok, text), f'[{name}] encode 不一致'
    # roundtrip
    ids = tok.encode(text)
    assert tok.decode(ids) == text, f'[{name}] roundtrip 失败'


def test_parallel_split_equals_full(tokenizer):
    """多进程分片 encode 拼接 == 单进程整段（逐 token）。"""
    text = (ZH + '\n\n' + 'He said "hi" 123! 「对话」\n' + ZH[:80]) * 200
    text = text[:50000]
    full = tokenizer.encode(text)
    with tempfile.TemporaryDirectory() as td:
        cp = encode_to_cache(text, tokenizer, td, 'par', n_procs=4)
        mm = ShardMemmap(cp)
        assert mm.token_count == len(full)
        assert list(mm[0:mm.token_count]) == full
        mm.close()


def test_random_slices_across_shards(tokenizer):
    """memmap 随机切片正确（含跨分片边界）。"""
    text = (ZH * 3 + '\n\nabc 123!? ') * 300
    text = text[:30000]
    full = tokenizer.encode(text)
    with tempfile.TemporaryDirectory() as td:
        cp = encode_to_cache(text, tokenizer, td, 'slice', n_procs=4)
        mm = ShardMemmap(cp)
        rng = random.Random(7)
        for _ in range(300):
            a = rng.randint(0, len(full) - 400)
            b = a + rng.randint(1, 200)
            assert list(mm[a:b]) == full[a:b], f'slice [{a},{b}) mismatch'
        # 负索引
        assert list(mm[-10:]) == full[-10:]
        mm.close()


def test_datasets_with_memmap(tokenizer):
    """PackedDataset/TextDataset 用 memmap 出 batch 与整段一致。"""
    text = (ZH + '\n\nxyz 456 ') * 200
    text = text[:20000]
    full = tokenizer.encode(text)
    with tempfile.TemporaryDirectory() as td:
        cp = encode_to_cache(text, tokenizer, td, 'ds', n_procs=4)
        mm = ShardMemmap(cp)
        ds = PackedDataset(mm, block_size=256)
        x, y = ds[5]
        assert list(x) == full[5 * 256:5 * 256 + 256]
        assert list(y) == full[5 * 256 + 1:5 * 256 + 257]
        ds2 = TextDataset(mm, block_size=128)
        x2, _ = ds2[700]
        assert list(x2) == full[700:828]
        mm.close()


def test_real_webnovel_equivalence(tokenizer):
    """真实 webnovel 文本（几 MB）整段 == 分片。"""
    real = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'train_webnovel_v2.txt')
    if not os.path.exists(real):
        pytest.skip('真实语料不存在')
    with open(real, 'rb') as f:
        raw = f.read(2_000_000 * 3)
    text = raw.decode('utf-8', errors='replace')[:2_000_000]
    full = tokenizer.encode(text)
    with tempfile.TemporaryDirectory() as td:
        cp = encode_to_cache(text, tokenizer, td, 'real', n_procs=4)
        mm = ShardMemmap(cp)
        assert mm.token_count == len(full)
        assert list(mm[0:mm.token_count]) == full
        mm.close()


def test_uint16_bounds():
    """vocab > 65535 时应报错（uint16 溢出保护）。"""
    from data.token_cache import encode_to_cache
    with pytest.raises(AssertionError):
        # 伪造大 vocab tokenizer
        class FakeTok:
            vocab = list(range(70000))
            merges = {}
            def encode(self, t):
                return [0]
        with tempfile.TemporaryDirectory() as td:
            encode_to_cache('abc', FakeTok(), td, 'big')
