"""token cache 管线：并行编码 → uint16 分片落盘 → memmap 只读加载。

目标（encode pipeline 专项优化，不改 BPE 语义）：
- 700M token 用 int64 (5.6GB) 太浪费 → uint16 (1.4GB)，vocab<=65535 时安全
- 不把整段文本/整批 token 载入内存 → worker 各自写分片文件
- 训练侧 memmap 按需读分片，不 np.load 整个 1.4GB

分片布局（cache_dir 下）:
  {tag}_v{vocab}_s{sample}_len{chars}_e{version}/
    shard_000000.bin      # uint16 原生数组（无 npy 头，纯 token）
    shard_000001.bin
    ...
    index.json            # 元数据: dtype/vocab/tokenizer_hash/token_count/分片表

与旧 tokens_*.npy 的差异:
- 旧: 单个 int64 npy，整载内存
- 新: 多个 uint16 bin + index，memmap 按片读（可只映射需要的片段）

正确性:
- 切分严格在预分词块间隙进行 → 每片独立编码 == 整段编码（逐 token 一致）
- tests/test_encode_equivalence.py 验证
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np

from data.tokenizer import PRETOK_RE  # noqa: E402

_VERSION = 1

_ENC_TOK = None


def _worker_init(tok):
    global _ENC_TOK
    _ENC_TOK = tok


def _split_points(text: str, n_parts: int):
    """在预分词块间隙处取 n_parts-1 个尽量均匀的切分点（字符索引）。

    保证每个切分点落在两个 chunk 之间（空白/标点区），
    因此每片独立 encode 的结果 == 整段 encode（逐 token 一致）。
    若无足够间隙（极端文本），退化为按字符均分（可能跨块，仅用于极端兜底）。
    """
    spans = [(m.start(), m.end()) for m in PRETOK_RE.finditer(text)]
    gaps = []
    for i in range(len(spans) - 1):
        a_end = spans[i][1]
        b_start = spans[i + 1][0]
        if b_start > a_end:
            gaps.append((a_end + b_start) // 2)
    if not gaps:
        # 退化：无空白分隔的极端文本（几乎不会发生）
        return [len(text) * k // n_parts for k in range(1, n_parts)]
    step = len(gaps) / n_parts
    return [gaps[int(step * k)] for k in range(1, n_parts)]


def _encode_write(args):
    """worker: 编码一段文本并直接写 uint16 bin 文件。返回元数据（小）。"""
    text, path = args
    ids = _ENC_TOK.encode(text)
    arr = np.asarray(ids, dtype=np.uint16)
    # 原子写：先写临时再 rename（避免半截文件被读到）
    tmp = path + '.tmp'
    arr.tofile(tmp)
    os.replace(tmp, path)
    return {
        'path': path,
        'token_count': len(ids),
        'chars_count': len(text),
    }


def encode_to_cache(text: str, tokenizer, cache_dir: str, tag: str,
                    n_procs: int = 8, chunk_chars: int = 8_000_000) -> str:
    """并行编码整段文本 → 分片 uint16 cache。返回 cache 目录路径。

    - text 按 chunk_chars 粗切成大块，每块再按 n_procs 分片并行
    - worker 直写 bin 文件，只回传 (path, token_count, chars_count)
    - 主进程最后写 index.json
    """
    vocab_size = len(tokenizer.vocab)
    assert vocab_size <= 65535, f'vocab {vocab_size} > 65535，uint16 溢出'
    tok_hash = hashlib.md5(
        repr(sorted(tokenizer.merges.items())[:1000]).encode()).hexdigest()[:12]

    cache_path = os.path.join(
        cache_dir, f'{tag}_v{vocab_size}_len{len(text)}_e{_VERSION}')
    if os.path.exists(os.path.join(cache_path, 'index.json')):
        print(f'从缓存加载 {tag}: {cache_path}')
        return cache_path
    os.makedirs(cache_path, exist_ok=True)

    if n_procs <= 1 or len(text) < 2_000_000:
        n_procs = 1

    # 分块：每块 ~n_procs×~1M 字符 → worker 各编码一片
    jobs = []
    n_chunks = max(1, (len(text) + chunk_chars - 1) // chunk_chars)
    chunk_size = (len(text) + n_chunks - 1) // n_chunks
    print(f'编码 {len(text):,} 字符 → {n_chunks} 块 × {n_procs} worker '
          f'(uint16 分片落盘) ...', flush=True)

    for ci in range(n_chunks):
        c_start = ci * chunk_size
        c_end = min(len(text), c_start + chunk_size)
        c_text = text[c_start:c_end]
        cuts = _split_points(c_text, n_procs)
        pts = [0] + cuts + [len(c_text)]
        for pi in range(len(pts) - 1):
            sub = c_text[pts[pi]:pts[pi + 1]]
            if not sub:
                continue
            path = os.path.join(cache_path, f'shard_{ci:03d}_{pi:03d}.bin')
            jobs.append((sub, path))

    meta_list = []
    if n_procs > 1:
        with mp.Pool(n_procs, initializer=_worker_init,
                     initargs=(tokenizer,)) as pool:
            for m in pool.imap_unordered(_encode_write, jobs):
                meta_list.append(m)
                if len(meta_list) % 20 == 0:
                    print(f'  {len(meta_list)}/{len(jobs)} 片完成', flush=True)
    else:
        _worker_init(tokenizer)
        for j in jobs:
            meta_list.append(_encode_write(j))

    meta_list.sort(key=lambda m: os.path.basename(m['path']))
    total_tokens = sum(m['token_count'] for m in meta_list)
    total_chars = sum(m['chars_count'] for m in meta_list)

    index = {
        'version': _VERSION,
        'dtype': 'uint16',
        'vocab_size': vocab_size,
        'tokenizer_hash': tok_hash,
        'token_count': total_tokens,
        'chars_count': total_chars,
        'n_shards': len(meta_list),
        'shards': [
            {'file': os.path.basename(m['path']),
             'token_count': m['token_count'],
             'chars_count': m['chars_count']}
            for m in meta_list
        ],
    }
    with open(os.path.join(cache_path, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f'编码完成: {total_chars:,} 字符 → {total_tokens:,} tokens '
          f'({total_tokens/1e6:.0f}M) → {cache_path}')
    return cache_path


def load_index(cache_path: str) -> dict:
    with open(os.path.join(cache_path, 'index.json'), encoding='utf-8') as f:
        return json.load(f)


class ShardMemmap:
    """多个 uint16 bin 分片的只读 memmap 视图（不整载内存）。

    提供 len() 与 __getitem__ 切片（返回 np.uint16 数组），
    训练侧转 torch.long 前只保留当前 batch 需要的片段。
    """

    def __init__(self, cache_path: str):
        self.path = cache_path
        self.index = load_index(cache_path)
        assert self.index['dtype'] == 'uint16'
        self.vocab_size = self.index['vocab_size']
        self.token_count = self.index['token_count']
        # 预建每个分片的 (start, end, memmap)
        self._shards = []
        off = 0
        for s in self.index['shards']:
            fp = os.path.join(cache_path, s['file'])
            n = s['token_count']
            mm = np.memmap(fp, dtype=np.uint16, mode='r')
            self._shards.append((off, off + n, mm))
            off += n

    def __len__(self):
        return self.token_count

    def __getitem__(self, sl):
        """支持 int 与 slice；返回 np.uint16 数组（slice 可能跨分片）。"""
        if isinstance(sl, slice):
            start, stop, step = sl.indices(self.token_count)
            if step != 1:
                raise NotImplementedError('仅支持连续切片 (step=1)')
            return self._slice_range(start, stop)
        if sl < 0:
            sl += self.token_count
        for s0, s1, mm in self._shards:
            if s0 <= sl < s1:
                return np.uint16(mm[sl - s0])
        raise IndexError(sl)

    def _slice_range(self, start, stop):
        # 定位覆盖 [start, stop) 的分片并拼接
        parts = []
        for s0, s1, mm in self._shards:
            if s1 <= start or s0 >= stop:
                continue
            lo = max(start, s0) - s0
            hi = min(stop, s1) - s0
            parts.append(mm[lo:hi])
        if not parts:
            return np.empty(0, dtype=np.uint16)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts)

    def close(self):
        """关闭所有分片 memmap（Windows 下必须释放文件句柄）。"""
        for s0, s1, mm in self._shards:
            try:
                mm._mmap.close()   # noqa: SLF001 - numpy memmap 底层释放
            except Exception:
                pass
        self._shards = []
