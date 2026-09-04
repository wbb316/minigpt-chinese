from data.tokenizer import BPETokenizer


def test_chinese():
    text = "人工智能技术正在快速发展，人工智能改变世界。" * 50
    tokenizer = BPETokenizer(vocab_size=256 + 50)
    tokenizer.train(text)
    # 看它合并出了什么词
    print("合并出的新词表（部分）:")
    for pair, new_id in tokenizer.merges.items():
        merged = tokenizer.decode([pair[0]]) + tokenizer.decode([pair[1]])
        print(f"  {new_id}: {merged}")

    # 验证：编码后 token 数大幅减少（说明合并生效）
    encoded = tokenizer.encode("人工智能技术")
    print(f"\n'人工智能技术' 原始字节数: {len('人工智能技术'.encode())}, 编码后 token 数: {len(encoded)}")
    print("解码还原:", tokenizer.decode(encoded))


def test_bpe_merges_common_pairs():
    """训练后，频繁出现的组合应该被合并成一个 token。"""
    tokenizer = BPETokenizer(vocab_size=256+5)
    text='ab'*100
    tokenizer.train(text)
    # 字节 'a'=97, 'b'=98，偏移 NUM_SPECIAL=2 后是 (99, 100)
    assert (99, 100) in tokenizer.merges
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
