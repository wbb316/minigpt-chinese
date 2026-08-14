"""测试 BPE tokenizer。"""
from data.tokenizer import BPETokenizer


def test_bpe_merges_common_pairs():
    """训练后，频繁出现的组合应该被合并成一个 token。"""
    tokenizer = BPETokenizer(vocab_size=256 + 5)   # 初始256字节 + 合并5次
    # 一段大量重复的文本（"ab" 反复出现，应该被合并）
    text = "ab" * 100
    tokenizer.train(text)

    # 验证：合并规则里有 (a, b) 这一对（因为 ab 反复出现）
    assert (97, 98) in tokenizer.merges, "ab 应该被合并，但没找到合并规则"

    # 验证：encode 后 ab 是一个 token
    encoded = tokenizer.encode("ababab")
    assert len(encoded) < len("ababab")  # 编码后应该比原始字符少（合并了）
    assert tokenizer.decode(encoded) == "ababab"   # 解码能还原


def test_bpe_roundtrip():
    """编码→解码 必须还原原文（不丢信息）。"""
    tokenizer = BPETokenizer(vocab_size=256 + 10)
    tokenizer.train("hello world hello world hello world")

    text = "hello world"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text, f"roundtrip 失败: {decoded} != {text}"
