"""训练数据集：滑动窗口 / 打包两种模式。

- TextDataset:   滑动窗口 stride=1（标准做法），样本数 = len(tokens) - block_size
- PackedDataset: 打包不重叠 (stride=block_size)，每 token 每 epoch 只看一次

两种模式底层都是整块 tensor 视图切片（零拷贝），替换掉旧的
"Python list 逐样本切片 + torch.tensor()" 写法（20M token 的 list ≈ 720MB，
且每个样本都要新建 tensor，数据加载成为瓶颈）。
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def _as_tensor(tokens):
    """list / numpy 数组 → torch.LongTensor（numpy 走零拷贝 from_numpy）。"""
    if isinstance(tokens, torch.Tensor):
        return tokens.long()
    if isinstance(tokens, np.ndarray):
        if tokens.dtype == np.int64:
            return torch.from_numpy(tokens)
        return torch.from_numpy(tokens.astype(np.int64))
    return torch.tensor(tokens, dtype=torch.long)


class TextDataset(Dataset):
    """滑动窗口（stride=1）：每个 token 出现在 block_size 个窗口里。

    与旧版语义完全一致，但 __getitem__ 是整块 tensor 的视图切片，零拷贝。
    样本数 = len(tokens) - block_size（最后不够一个窗口的部分舍弃）。
    """

    def __init__(self, tokens, block_size: int = 128):
        self.tokens = _as_tensor(tokens)
        self.block_size = block_size
        self.num_samples = len(self.tokens) - block_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + 1 + self.block_size]
        return x, y


class PackedDataset(Dataset):
    """打包模式（stride=block_size，不重叠）：样本数 = len//block_size。

    每 token 每 epoch 只看 1 次 → 每 epoch 步数少、CPU 上快 ~100 倍；
    代价是窗口边界处的跨序列依赖看不到（对 block_size 内依赖无影响）。
    offset 可错开切分位置（val 用它避免与 train 恰好切在同一处）。
    """

    def __init__(self, tokens, block_size: int = 128, offset: int = 0):
        arr = _as_tensor(tokens)
        if offset:
            arr = arr[offset:]
        n = (len(arr) - 1) // block_size
        if n <= 0:
            raise ValueError(f'语料太短: {len(arr)} token，至少需要 block_size+1={block_size + 1}')
        self.x = arr[:n * block_size].view(n, block_size)          # 视图，不复制
        self.y = arr[1:n * block_size + 1].view(n, block_size)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
