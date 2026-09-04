"""BPETokenizer v2 验证：
1) 增量式 train 与"朴素全量重扫 + 预分词"结果完全一致（merges/vocab 逐项相等）
2) 编码-解码 roundtrip 不丢信息
"""
import re
from collections import Counter

from data.tokenizer import BPETokenizer, PRETOK_RE, NUM_SPECIAL

# 朴素参考实现：预分词 + 全量重扫计数 + 左到右不重叠合并（v1 算法的分块版）
def naive_chunked_train(text, vocab_size):
    chunks = PRETOK_RE.findall(text)
    seq = []
    for ch in chunks:
        seq.extend(b + NUM_SPECIAL for b in ch.encode('utf-8'))
        seq.append(-1)                       # 块间哨兵, 不参与 pair

    vocab = {i + NUM_SPECIAL: bytes([i]) for i in range(256)}
    vocab.update({0: b'<unk>', 1: b'<eos>'})
    merges = {}

    while len(vocab) < vocab_size:
        counts = Counter()
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a >= 0 and b >= 0:
                counts[(a, b)] += 1
        if not counts:
            break
        # 与 data/tokenizer.py 一致的确定性平局规则
        pair = max(counts, key=lambda k: (counts[k], -k[0], -k[1]))
        c = len(vocab)
        # 左到右、不重叠替换
        new = []
        i = 0
        while i < len(seq):
            if (i < len(seq) - 1 and seq[i] == pair[0]
                    and seq[i + 1] == pair[1] and seq[i] >= 0):
                new.append(c)
                i += 2
            else:
                new.append(seq[i])
                i += 1
        seq = new
        merges[pair] = c
        vocab[c] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab


def _make_test_text():
    """混合中文(含标点)/ASCII/数字/空白, 制造大量可合并的重复片段。"""
    zh = '人工智能改变世界，她轻轻握住我的手。\n「我说啊……你说得对。」他低声说道。'
    en = 'hello world! 123 abc hello hello '
    base = (zh * 40 + '\n' + en * 60 + '\n') * 25
    base += ('ab' * 200 + '，' + '喜欢' * 300 + '。') * 30
    return base


def test_train_equals_naive_chunked():
    text = _make_test_text()
    vocab_size = 700
    fast = BPETokenizer(vocab_size=vocab_size)
    fast.train(text)
    naive_merges, naive_vocab = naive_chunked_train(text, vocab_size)

    assert fast.merges == naive_merges, 'merges 不一致!'
    assert fast.vocab == naive_vocab, 'vocab 不一致!'
    assert len(fast.vocab) == len(naive_vocab)
    assert len(fast.merges) == len(naive_merges)


def test_train_equals_naive_chunked_ascii():
    text = ('the quick brown fox jumps over the lazy dog. ' * 200
            + '\n' + 'word word pair pair ' * 200)
    vocab_size = 500
    fast = BPETokenizer(vocab_size=vocab_size)
    fast.train(text)
    naive_merges, naive_vocab = naive_chunked_train(text, vocab_size)
    assert fast.merges == naive_merges
    assert fast.vocab == naive_vocab


def test_encode_decode_roundtrip():
    text = _make_test_text()
    tok = BPETokenizer(vocab_size=900)
    tok.train(text)
    encoded = tok.encode(text)
    assert tok.decode(encoded) == text, 'roundtrip 失败'


def test_merges_never_cross_punctuation():
    """预分词语义: '词+句号' 这类跨块 pair 不应出现在 merges 里。"""
    text = '我爱你。' * 500 + '你好吗？' * 500 + '等等！' * 500
    tok = BPETokenizer(vocab_size=600)
    tok.train(text)
    for (a, b) in tok.merges:
        # 句号/问号/叹号是独立块, 其字节不应与其他块内容合并
        combined = tok.vocab[a] + tok.vocab[b]
        for p in '。？！':
            pb = p.encode('utf-8')
            assert combined.find(pb) < 0 or combined == pb, f'跨标点合并: {combined}'
