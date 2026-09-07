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
    """取 n_parts-1 个尽量均匀的"安全切分点"（字符索引）。

    ★ 安全切分点 = PRETOK chunk 之间的边界（严格正确，P0d 修复）：
    旧实现把 \\s+ 空白串内部当安全切分点——但 \\s+ 本身是一个完整的
    pretoken chunk，BPE merge 允许在 chunk 内部发生，空白内部切开可能
    改变该 chunk 的 merge 结果。真正安全的位置只有 chunk 间隙
    （PRETOK_RE = \\w+|[^\\w\\s]|\\s+ 无缝覆盖文本，chunk 之间无间隙，
    因此安全边界 = 除最后一个外的每个 match.end()）。

    切分点 = 离均匀目标位置最近的 chunk 边界（ends 单调递增 → searchsorted
    二分，O(log n_chunks)/切分点；全量扫描一次 finditer 收集边界）。

    正确性：每片由完整 chunk 组成 → 每片独立 encode == 整段 encode（逐 token）。
    """
    if n_parts <= 1:
        return []
    # 所有 chunk 的 end（升序；无缝覆盖 → 相邻 chunk 的 end == 下一 start）
    ends = np.fromiter((m.end() for m in PRETOK_RE.finditer(text)),
                       dtype=np.int64)
    if ends.size <= 1:
        return []          # 无法切分 → 单片处理
    ends = ends[:-1]       # 去掉末尾 = len(text)（不可作为切分点）
    total = len(text)
    cuts = []
    for k in range(1, n_parts):
        target = total * k // n_parts
        j = int(np.searchsorted(ends, target, side='left'))
        best = None
        for cand in (j - 1, j):
            if 0 <= cand < ends.size:
                c = int(ends[cand])
                if best is None or abs(c - target) < abs(best - target):
                    best = c
        cuts.append(best)
    # 去重、保序
    cuts = sorted(set(cuts))
    return cuts


def _encode_write(args):
    """worker: 从 txt 分片读文本 → 编码 → 直接写 uint16 bin。返回元数据（小）。

    args = (txt_path, bin_path)。文本经文件传递，不走 multiprocessing IPC
    （spawn 模式下大字符串 pickle 传输极慢）。
    """
    txt_path, bin_path = args
    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    ids = _ENC_TOK.encode(text)
    arr = np.asarray(ids, dtype=np.uint16)
    tmp = bin_path + '.tmp'
    arr.tofile(tmp)
    os.replace(tmp, bin_path)
    return {
        'path': bin_path,
        'token_count': len(ids),
        'chars_count': len(text),
    }


def _tokenizer_hash_legacy(tokenizer):
    """旧版 tokenizer hash（前 1000 merges 取 12 位）——兼容 2026-09 建的存量 cache。

    不足：merges>1000 时只覆盖前缀，tokenizer 尾部变化可能察觉不到。
    仅用于校验无 tokenizer_hash_full 字段的旧 index。
    """
    return hashlib.md5(
        repr(sorted(tokenizer.merges.items())[:1000]).encode()).hexdigest()[:12]


def _tokenizer_hash_full(tokenizer):
    """完整稳定 hash：全部 merges（sorted 保证跨进程稳定）。

    vocab 可由 256 字节表 + merges 序列唯一推出 → 覆盖 merges 即覆盖 tokenizer
    全部状态（vocab_size/NUM_SPECIAL 为固定常量，另由 index.vocab_size 校验）。
    """
    return hashlib.md5(repr(sorted(tokenizer.merges.items())).encode()).hexdigest()


def _index_ok(idx, tokenizer, vocab_size, text_len):
    """cache index 与当前 tokenizer/文本匹配？（P0e：防错误复用）"""
    if idx.get('version') != _VERSION:
        return False
    if idx.get('dtype') != 'uint16':
        return False
    if idx.get('vocab_size') != vocab_size:
        return False
    if idx.get('chars_count') != text_len:
        return False
    fh = idx.get('tokenizer_hash_full')
    if fh is not None:
        return fh == _tokenizer_hash_full(tokenizer)
    # 旧 cache（无 full 字段）：legacy hash 比对
    return idx.get('tokenizer_hash') == _tokenizer_hash_legacy(tokenizer)


def encode_to_cache(text: str, tokenizer, cache_dir: str, tag: str,
                    n_procs: int = 8, chunk_chars: int = 8_000_000) -> str:
    """并行编码整段文本 → 分片 uint16 cache。返回 cache 目录路径。

    - text 按 chunk_chars 粗切成大块，每块再按 n_procs 分片并行
    - worker 直写 bin 文件，只回传 (path, token_count, chars_count)
    - 主进程最后写 index.json
    - cache 命中前做完整校验（tokenizer hash / vocab / chars / version / dtype）；
      校验失败不删除旧 cache，另建带 hash 后缀的新目录（存量 cache 不受破坏）。
    """
    vocab_size = len(tokenizer.vocab)
    assert vocab_size <= 65535, f'vocab {vocab_size} > 65535，uint16 溢出'
    tok_hash_legacy = _tokenizer_hash_legacy(tokenizer)
    tok_hash_full = _tokenizer_hash_full(tokenizer)

    cache_path = os.path.join(
        cache_dir, f'{tag}_v{vocab_size}_len{len(text)}_e{_VERSION}')
    index_path = os.path.join(cache_path, 'index.json')
    if os.path.exists(index_path):
        idx = load_index(cache_path)
        if _index_ok(idx, tokenizer, vocab_size, len(text)):
            print(f'从缓存加载 {tag}: {cache_path}')
            return cache_path
        # tokenizer/语料等已变化：不删旧 cache，用 hash 后缀目录重建
        cache_path = os.path.join(
            cache_dir, f'{tag}_v{vocab_size}_len{len(text)}_e{_VERSION}'
                       f'_h{tok_hash_full[:10]}')
        index_path = os.path.join(cache_path, 'index.json')
        print(f'⚠️ 缓存校验失败（tokenizer/内容变化）→ 使用新缓存目录: {cache_path}')
        if os.path.exists(index_path):
            idx = load_index(cache_path)
            assert _index_ok(idx, tokenizer, vocab_size, len(text)), \
                'hash 后缀目录仍校验失败（hash 碰撞或逻辑错误）'
            return cache_path
    os.makedirs(cache_path, exist_ok=True)

    if n_procs <= 1 or len(text) < 2_000_000:
        n_procs = 1

    # 分片：整段文本在所有预分词块间隙处切分（全局切分，绝无块边界切断 chunk）
    # 目标每片 ~1.5M 字符（控制单 worker 内存），片数 = ceil(len / 1.5M)
    target_parts = max(n_procs, (len(text) + 1_500_000 - 1) // 1_500_000)
    cuts = _split_points(text, target_parts)
    pts = [0] + cuts + [len(text)]
    # 相邻切分点可能很近（间隙密集），合并过小的片
    jobs = []
    txt_dir = os.path.join(cache_path, '_txt')
    os.makedirs(txt_dir, exist_ok=True)
    pieces = []
    for pi in range(len(pts) - 1):
        sub = text[pts[pi]:pts[pi + 1]]
        if sub:
            pieces.append(sub)
    del text

    print(f'编码 → {len(pieces)} 片 × {n_procs} worker '
          f'(uint16 分片落盘，全部切分点在块间隙) ...', flush=True)
    for pi, sub in enumerate(pieces):
        txt_path = os.path.join(txt_dir, f'part_{pi:04d}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(sub)
        bin_path = os.path.join(cache_path, f'shard_{pi:04d}.bin')
        jobs.append((txt_path, bin_path))
    del pieces

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
        'tokenizer_hash': tok_hash_legacy,     # 兼容旧读取（前 1000 merges）
        'tokenizer_hash_full': tok_hash_full,  # P0e：完整 tokenizer 状态 hash
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
    # 清理临时 txt 分片（编码已完成，不再需要）
    import shutil
    shutil.rmtree(txt_dir, ignore_errors=True)
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
        # 预建每个分片的 (start, end, memmap) + 二分索引（P0b）
        # 旧实现每次 slice 从头线性遍历全部 shard（828 片 × 每样本 2 次 slice）
        self._shards = []
        starts = []
        ends = []
        off = 0
        for s in self.index['shards']:
            fp = os.path.join(cache_path, s['file'])
            n = s['token_count']
            mm = np.memmap(fp, dtype=np.uint16, mode='r')
            self._shards.append((off, off + n, mm))
            starts.append(off)
            ends.append(off + n)
            off += n
        self._starts = np.asarray(starts, dtype=np.int64)
        self._ends = np.asarray(ends, dtype=np.int64)
        self._n_shards = len(self._shards)

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
        # 二分定位：最后 start <= sl 的 shard（O(log n)）
        i = int(np.searchsorted(self._starts, sl, side='right')) - 1
        if 0 <= i < self._n_shards:
            s0, s1, mm = self._shards[i]
            if s0 <= sl < s1:
                return np.uint16(mm[sl - s0])
        raise IndexError(sl)

    def _slice_range(self, start, stop):
        # 二分定位起始 shard；绝大多数 513-token slice 不跨 shard → 单 view
        if stop <= start:
            return np.empty(0, dtype=np.uint16)
        i = int(np.searchsorted(self._starts, start, side='right')) - 1
        if i >= 0 and self._ends[i] <= start:
            i += 1                       # start 落在 shard i 与 i+1 的间隙
        if i < 0:
            i = 0
        parts = []
        while i < self._n_shards:
            s0, s1, mm = self._shards[i]
            if s0 >= stop:
                break
            lo = max(start, s0) - s0
            hi = min(stop, s1) - s0
            parts.append(mm[lo:hi])
            i += 1
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
