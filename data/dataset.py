"""训练数据集：滑动窗口 / 打包两种模式。

- TextDataset:   滑动窗口 stride=1（标准做法），样本数 = len(tokens) - block_size
- PackedDataset: 打包不重叠 (stride=block_size)，每 token 每 epoch 只看一次

两种模式底层都是整块 tensor 视图切片（零拷贝），替换掉旧的
"Python list 逐样本切片 + torch.tensor()" 写法（20M token 的 list ≈ 720MB，
且每个样本都要新建 tensor，数据加载成为瓶颈）。

v2 (encode pipeline 专项): 支持 uint16 memmap 分片（data/token_cache.py 的
ShardMemmap）——不再 np.load 整段转 int64，而是按需切片 uint16 → torch.long。
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def _as_tensor(tokens):
    """list / numpy 数组 / ShardMemmap → torch.LongTensor 的惰性来源。

    对 ShardMemmap（uint16 memmap）不整载：__getitem__ 时才切片转 long。
    """
    if isinstance(tokens, torch.Tensor):
        return tokens.long()
    if isinstance(tokens, np.ndarray):
        if tokens.dtype == np.int64:
            return torch.from_numpy(tokens)
        return torch.from_numpy(tokens.astype(np.int64))
    return torch.tensor(tokens, dtype=torch.long)


def _is_memmap(tokens):
    return type(tokens).__name__ == 'ShardMemmap'


class TextDataset(Dataset):
    """滑动窗口（stride=1）：每个 token 出现在 block_size 个窗口里。

    与旧版语义完全一致，但 __getitem__ 是整块 tensor 的视图切片，零拷贝。
    样本数 = len(tokens) - block_size（最后不够一个窗口的部分舍弃）。
    支持 uint16 ShardMemmap（按需切片，不整载）。
    """

    def __init__(self, tokens, block_size: int = 128):
        self.memmap = _is_memmap(tokens)
        self.block_size = block_size
        if self.memmap:
            self._n = len(tokens)
            self._tokens = tokens
        else:
            self.tokens = _as_tensor(tokens)
            self._n = len(self.tokens)
        self.num_samples = self._n - block_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.memmap:
            # 单次 slice 拿 block+1 个 token，x/y 共享同一份 copy+转换
            seq = self._tokens[idx:idx + self.block_size + 1]
            t = torch.from_numpy(np.array(seq, copy=True)).long()
            return t[:-1], t[1:]
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + 1 + self.block_size]
        return x, y


class PackedDataset(Dataset):
    """打包模式（stride=block_size，不重叠）：样本数 = len//block_size。

    每 token 每 epoch 只看 1 次 → 每 epoch 步数少、CPU 上快 ~100 倍；
    代价是窗口边界处的跨序列依赖看不到（对 block_size 内依赖无影响）。
    offset 可错开切分位置（val 用它避免与 train 恰好切在同一处）。
    支持 uint16 ShardMemmap（按需切片，不整载）。
    """

    def __init__(self, tokens, block_size: int = 128, offset: int = 0):
        self.memmap = _is_memmap(tokens)
        self.block_size = block_size
        self._offset = offset
        if self.memmap:
            # 只保存引用与元数据，绝不在 init 里做任何全量 slice
            # （旧版 arr = tokens[offset:offset+_n*block+1] 会触发 ShardMemmap
            #   全库 828 片扫描拼接 ~1GB，且 arr 从未被使用 —— P0a 修复）
            self._tokens = tokens
            n = (len(tokens) - offset - 1) // block_size
            if n <= 0:
                raise ValueError(
                    f'语料太短: {len(tokens)} token，至少需要 '
                    f'block_size+1={block_size + 1}')
            self._num = n
        else:
            arr = _as_tensor(tokens)
            if offset:
                arr = arr[offset:]
            n = (len(arr) - 1) // block_size
            if n <= 0:
                raise ValueError(f'语料太短: {len(arr)} token，'
                                 f'至少需要 block_size+1={block_size + 1}')
            self.x = arr[:n * block_size].view(n, block_size)
            self.y = arr[1:n * block_size + 1].view(n, block_size)
            self._num = n

    def __len__(self) -> int:
        return self._num

    def __getitem__(self, idx: int):
        if self.memmap:
            # x/y 单次读取（P0c）：一次 shard lookup + copy + uint16→long
            start = self._offset + idx * self.block_size
            seq = self._tokens[start:start + self.block_size + 1]
            t = torch.from_numpy(np.array(seq, copy=True)).long()
            return t[:-1], t[1:]
        return self.x[idx], self.y[idx]
