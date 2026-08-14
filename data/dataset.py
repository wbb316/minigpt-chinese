import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """把编码后的 token 序列切成可训练的样本。

    每次取一段长度为 block_size 的窗口作为输入，
    标签就是这段窗口右移一位（预测下一个 token）。
    """
    def __init__(self,tokens:list[int],block_size:int=512):
        self.tokens=tokens
        self.block_size=block_size
        # 样本总数 = 总 token 数 - block_size（最后不够一个窗口的部分舍弃）
        self.num_samples=len(tokens)-block_size


    def __len__(self) ->int:
        return self.num_samples

    def __getitem__(self, idx:int)  :
        # 从 idx 开始取 block_size 个 token 作为输入
        x=torch.tensor(self.tokens[idx:idx+self.block_size])
        # 标签 = 输入右移一位（x 的下一个 token）
        y=torch.tensor(self.tokens[idx+1:idx+1+self.block_size])
        return x,y