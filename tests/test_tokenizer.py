"""测试 BPE tokenizer。"""
from data.tokenizer import BPETokenizer


def test_bpe_merges_common_pairs():
    """训练后，频繁出现的组合应该被合并成一个 token。"""
    tokenizer = BPETokenizer(vocab_size=256 + 5)   # 初始256字节 + 合并5次
    # 一段大量重复的文本（"ab" 反复出现，应该被合并）
    text = "ab" * 100
    tokenizer.train(text)

    # 验证：合并规则里有 (a, b) 这一对（因为 ab 反复出现）
    # 字节 'a'=97, 'b'=98，偏移 NUM_SPECIAL=2 后是 (99, 100)
    assert (99, 100) in tokenizer.merges, "ab 应该被合并，但没找到合并规则"

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


def test_chinese_merges_reduce_tokens():
    """中文：合并后 token 数应比原始字节数少（说明合并生效）。"""
    tokenizer = BPETokenizer(vocab_size=256 + 50)
    text = "人工智能技术正在快速发展，人工智能改变世界。" * 50
    tokenizer.train(text)

    encoded = tokenizer.encode("人工智能技术")
    raw_bytes = len("人工智能技术".encode('utf-8'))
    assert len(encoded) < raw_bytes, f"中文未合并: {len(encoded)} 字节 vs 原始 {raw_bytes}"
    assert tokenizer.decode(encoded) == "人工智能技术", "解码未能还原中文"

    # 看合并出了什么（供观察，不参与断言）
    merged = tokenizer.decode([list(tokenizer.merges.keys())[-1][0]]) if tokenizer.merges else ""
    print(f"\n'人工智能技术' 原始 {raw_bytes} 字节 -> 编码后 {len(encoded)} 个 token, 解码还原: {tokenizer.decode(encoded)}")
