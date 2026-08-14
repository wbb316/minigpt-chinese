"""测试数据集封装。"""
import torch
from data.dataset import TextDataset


def test_dataset_shapes():
    """每个样本返回 (输入, 标签)，形状都是 [block_size]。"""
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]     # 8 个 token
    block_size = 3
    ds = TextDataset(tokens, block_size)

    # 样本数 = 8 - 3 = 5（idx 0~4）
    assert len(ds) == 5

    x, y = ds[0]
    assert x.shape == (3,) and y.shape == (3,)
    assert x.tolist() == [1, 2, 3]     # 输入：从 idx0 开始的3个
    assert y.tolist() == [2, 3, 4]     # 标签：右移一位
    assert torch.is_tensor(x) and torch.is_tensor(y)


def test_dataset_last_sample():
    """最后一个样本：x 和 y 长度相同（都是 block_size）。"""
    tokens = [1, 2, 3, 4, 5]
    block_size = 3
    ds = TextDataset(tokens, block_size)

    assert len(ds) == 2
    x, y = ds[1]                        # 最后一个样本（idx=1）
    assert x.tolist() == [2, 3, 4]      # tokens[1:4]
    assert y.tolist() == [3, 4, 5]      # tokens[2:5]
    assert x.shape == y.shape == (3,)
