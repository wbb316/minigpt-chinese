"""自研 BPE tokenizer（升级版：含特殊 token + save/load + <unk> 兜底）。

特殊 token 约定：
  id 0 = <unk>  未知字符兜底
  id 1 = <eos>  句子结束标记
  字节从 id 2 开始
"""
import json
from collections import Counter

# 特殊 token 的 id 和名字
UNK_ID = 0
EOS_ID = 1
SPECIAL_TOKENS = {UNK_ID: '<unk>', EOS_ID: '<eos>'}
NUM_SPECIAL = 2  # 特殊 token 数量


class BPETokenizer:
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.merges = {}   # {(旧1, 旧2): 新id}
        self.vocab = {}    # {id: bytes串}（特殊token存名字字节）

        # 初始化特殊 token（固定不参与合并）
        self.vocab[UNK_ID] = b'<unk>'
        self.vocab[EOS_ID] = b'<eos>'

    def train(self, text: str):
        # ① 初始：字节从 NUM_SPECIAL 开始编号（id 2..257）
        ids = list(text.encode('utf-8'))
        ids = [b + NUM_SPECIAL for b in ids]  # 字节 id 偏移
        vocab = {i + NUM_SPECIAL: bytes([i]) for i in range(256)}
        vocab.update(self.vocab)  # 加入特殊 token

        # ② 循环合并直到词表满
        while len(vocab) < self.vocab_size:
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = len(vocab)
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        self.vocab = vocab

    def _merge(self, ids, pair, new_id):
        """把 ids 里所有出现的 pair (a,b) 替换成 new_id（从左到右、不重叠）。"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def encode(self, text: str) -> list[int]:
        """文本 → token id 列表。遇到未知字节用 <unk> 兜底。"""
        raw = text.encode('utf-8', errors='ignore')  # 忽略无法编码的
        ids = [b + NUM_SPECIAL for b in raw]
        # 检查是否有词表外的 token，用 <unk> 兜底
        ids = [i if i in self.vocab else UNK_ID for i in ids]

        # 合并
        while len(ids) >= 2:
            count = Counter(zip(ids, ids[1:]))
            pair = min(count, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
        return ids

    def decode(self, ids: list[int]) -> str:
        """token id 列表 → 文本。特殊 token 用名字显示。"""
        parts = []
        for i in ids:
            if i in SPECIAL_TOKENS:
                parts.append(SPECIAL_TOKENS[i].encode('utf-8'))  # 特殊token转bytes
            elif i in self.vocab:
                parts.append(self.vocab[i])
            else:
                parts.append(b'')  # 未知 id 跳过
        return b''.join(parts).decode('utf-8', errors='replace')

    # ---------- save / load ----------
    def save(self, path: str):
        """保存词表 + 合并规则到文件（不依赖 pickle）。"""
        # 把 bytes 转成可序列化的 list
        vocab_serializable = {str(k): list(v) for k, v in self.vocab.items()}
        merges_serializable = {f'{a},{b}': c for (a, b), c in self.merges.items()}
        data = {
            'vocab_size': self.vocab_size,
            'vocab': vocab_serializable,
            'merges': merges_serializable,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """从文件加载分词器。"""
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(vocab_size=data['vocab_size'])
        tok.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
        tok.merges = {}
        for k, v in data['merges'].items():
            a, b = k.split(',')
            tok.merges[(int(a), int(b))] = int(v)
        return tok
