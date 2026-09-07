# -*- coding: utf-8 -*-
"""TRAINING_PERFORMANCE_AUDIT 专项等价性测试（P0 修复验证）。

覆盖:
1. ShardMemmap 二分定位后随机切片 == 线性参考实现（含跨分片边界）
2. PackedDataset memmap 路径（新 init 无全量 slice + x/y 单读）== ndarray 路径
   逐 token 一致（__len__ 一致 + 随机 100 idx x/y equal）
3. _split_points 全部切分点严格落在 PRETOK chunk 边界上
   + 分片 encode 拼接 == 整段 encode（多形态文本 exact equality）
4. cache hash 校验：tokenizer merges 变化后 encode_to_cache 不错误复用旧 cache
   （旧目录保留，返回带 hash 后缀的新目录）
"""
import os
import random

import numpy as np
import pytest
import torch

from data.token_cache import (_split_points, ShardMemmap, encode_to_cache,
                              load_index)
from data.dataset import PackedDataset, TextDataset
from data.tokenizer import BPETokenizer, PRETOK_RE

rng = random.Random(1234)


# ---------------------------------------------------------------- helpers
def make_cache_dir(tmp_path, tokens, n_shards=5, vocab=1024, tag='t'):
    """把 tokens 切 n_shards 段写 uint16 bin + index.json，返回 (dir, tokens 参考)。
    边界故意选在 shard 边界上，制造跨片切片场景。"""
    cache = tmp_path / f'{tag}_v{vocab}'
    cache.mkdir()
    # 切分点（确保覆盖 int 切片与跨片场景）
    cuts = sorted(rng.sample(range(1, len(tokens)), n_shards - 1))
    pts = [0] + cuts + [len(tokens)]
    shards = []
    for i in range(n_shards):
        seg = tokens[pts[i]:pts[i + 1]]
        fp = cache / f'shard_{i:06d}.bin'
        seg.astype(np.uint16).tofile(fp)
        shards.append({'file': fp.name, 'token_count': int(len(seg)),
                       'chars_count': int(len(seg))})
    idx = {'version': 1, 'dtype': 'uint16', 'vocab_size': vocab,
           'tokenizer_hash': 'x', 'tokenizer_hash_full': 'x',
           'token_count': int(len(tokens)), 'chars_count': int(len(tokens)),
           'n_shards': n_shards, 'shards': shards}
    import json
    with open(cache / 'index.json', 'w', encoding='utf-8') as f:
        json.dump(idx, f)
    return str(cache)


@pytest.fixture(scope='module')
def tokenizer():
    tok = BPETokenizer(vocab_size=1024)
    zh = ('她抬起头望向窗外的天空心里想着那个人的模样。\n\n'
          '他走进教室看见黑板上的字迹模糊不清。\t  \n'
          'The quick brown fox 12345 你好 world!?')
    tok.train(zh * 60, 'fast')
    return tok


# ------------------------------------------------- 1. ShardMemmap 切片
def test_memmap_slices_after_bisect(tmp_path):
    n = 100_000
    tokens = np.array([rng.randrange(256) for _ in range(n)], dtype=np.uint16)
    cp = make_cache_dir(tmp_path, tokens)
    mm = ShardMemmap(cp)
    assert len(mm) == n
    ref = np.concatenate([np.fromfile(cp + f'/shard_{i:06d}.bin',
                                      dtype=np.uint16) for i in range(5)])
    # int 访问
    for _ in range(200):
        i = rng.randrange(n)
        assert int(mm[i]) == int(ref[i])
    # 随机 slice（含跨片），范围故意贴近边界
    for _ in range(500):
        a = rng.randrange(n - 100)
        b = min(n, a + rng.randrange(1, 900))
        got = np.asarray(mm[a:b])
        assert got.shape == ref[a:b].shape
        assert np.array_equal(got, ref[a:b])
    # 已知跨片边界精确值：逐片端点
    for i in range(5 - 1):
        seg_end = int(np.loadtxt  # noqa: 占位防误用
                      ) if False else 0
    # 显式检查每个 shard 接缝前后的小窗口
    offs = []
    with open(os.path.join(cp, 'index.json'), encoding='utf-8') as f:
        import json
        for s in json.load(f)['shards']:
            offs.append(s['token_count'])
    seams = np.cumsum(offs)[:-1]
    for s in seams:
        win = np.asarray(mm[s - 3:s + 3])
        assert np.array_equal(win, ref[s - 3:s + 3])
    mm.close()


# --------------------------------------------- 2. PackedDataset memmap == ndarray
def test_packed_dataset_memmap_equals_ndarray(tmp_path):
    n = 20_000
    tokens = np.array([rng.randrange(300) for _ in range(n)], dtype=np.uint16)
    cp = make_cache_dir(tmp_path, tokens, n_shards=7)
    mm = ShardMemmap(cp)
    for block in (16, 64):
        for offset in (0, 11, 512):
            ds_mm = PackedDataset(mm, block, offset=offset)
            ds_np = PackedDataset(tokens.astype(np.int64), block, offset=offset)
            assert len(ds_mm) == len(ds_np), (block, offset)
            for idx in rng.sample(range(len(ds_mm)), min(100, len(ds_mm))):
                xm, ym = ds_mm[idx]
                xn, yn = ds_np[idx]
                assert torch.equal(xm, xn), f'x 不一致 block={block} off={offset} idx={idx}'
                assert torch.equal(ym, yn), f'y 不一致 block={block} off={offset} idx={idx}'
    # TextDataset memmap 也对照
    for block in (16, 64):
        ds_mm = TextDataset(mm, block)
        ds_np = TextDataset(tokens.astype(np.int64), block)
        assert len(ds_mm) == len(ds_np)
        for idx in rng.sample(range(len(ds_mm)), min(100, len(ds_mm))):
            xm, ym = ds_mm[idx]
            xn, yn = ds_np[idx]
            assert torch.equal(xm, xn) and torch.equal(ym, yn)
    mm.close()


# --------------------------------------------- 3. safe split 严格 chunk 边界
def test_split_points_on_chunk_boundaries(tokenizer):
    texts = {
        '中文段落': '第一段内容。\n\n第二段内容，含"引号"。\n\n第三段。\n' * 300,
        '连续空格': 'a   b \t\t  c \n\n\n d \n' * 200,
        '标点密集': '！？。，、；：""''（）【】《》——…' * 150,
        '中英数混合': '他买了 iPhone 15 和 MacBook 花了 8888 元！' * 180
                     + '\n' + 'Speed=100km/h value_3.14 test-item\n' * 40,
        '长词块': ('x' * 500 + '哈' * 500 + '\n') * 30,
        '混合小说': ('夜色渐深，他推开窗。"走吧，"她说。\n\n'
                     '外面下着雨，2026 年 9 月的第 6 天。\t' * 120),
    }
    for name, text in texts.items():
        # ① 切分点必须在 chunk 边界集合内
        bounds = {m.end() for m in PRETOK_RE.finditer(text)}
        bounds.discard(len(text))
        for n_parts in (2, 5, 16):
            cuts = _split_points(text, n_parts)
            for c in cuts:
                assert c in bounds, f'[{name}] cut {c} 不在 chunk 边界'
        # ② 分片 encode 拼接 == 整段 encode（exact）
        full = tokenizer.encode(text)
        cuts = _split_points(text, 16)
        pts = [0] + cuts + [len(text)]
        pieces = [text[pts[i]:pts[i + 1]] for i in range(len(pts) - 1)]
        pieces = [p for p in pieces if p]
        joined = []
        for p in pieces:
            joined.extend(tokenizer.encode(p))
        assert joined == full, f'[{name}] 分片拼接 ≠ 整段 encode'
    # 大空白文本做 100 片压力测试
    big = ('\n' * 3 + 'abc' + '  ' * 2 + '\n' * 5 + '中文段' * 30) * 400
    full = tokenizer.encode(big)
    cuts = _split_points(big, 100)
    pts = [0] + cuts + [len(big)]
    joined = []
    for i in range(len(pts) - 1):
        sub = big[pts[i]:pts[i + 1]]
        if sub:
            joined.extend(tokenizer.encode(sub))
    assert joined == full


# --------------------------------------------- 4. cache hash 失效
def test_cache_hash_invalidation(tmp_path):
    tok = BPETokenizer(vocab_size=1024)
    zh = '你好世界 hello world 123 测试文本。\n'
    tok.train(zh * 200, 'fast')
    text = (zh * 800)[:50_000]

    cp1 = encode_to_cache(text, tok, str(tmp_path), 'hash', n_procs=1)
    idx1 = load_index(cp1)
    assert 'tokenizer_hash_full' in idx1          # 新 index 有 full hash

    # 再调（tokenizer 未变）→ 命中同一目录
    cp2 = encode_to_cache(text, tok, str(tmp_path), 'hash', n_procs=1)
    assert cp2 == cp1

    # tokenizer merges 变化（模拟重训：内容变、vocab 大小不变）→ 不错误复用
    items = sorted(tok.merges.items())
    (ka, va), (kb, vb) = items[0], items[1]
    tok.merges[ka] = vb          # 交换两条 merge 的目标 id → hash 变化
    tok.merges[kb] = va
    cp3 = encode_to_cache(text, tok, str(tmp_path), 'hash', n_procs=1)
    assert cp3 != cp1, 'tokenizer 变化后仍命中旧 cache'
    assert '_h' in os.path.basename(cp3), f'新目录应带 hash 后缀: {cp3}'
    assert os.path.isdir(cp1), '旧 cache 不应被删除'
    # 同 tokenizer 再调 → 命中 hash 目录
    cp4 = encode_to_cache(text, tok, str(tmp_path), 'hash', n_procs=1)
    assert cp4 == cp3
    # chars 变化 → 目录名变（len），天然失效
    cp5 = encode_to_cache(text[:9_000], tok, str(tmp_path), 'hash', n_procs=1)
    assert cp5 != cp3
