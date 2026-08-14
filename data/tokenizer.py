import re
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size: int =512):
        self.vocab_size = vocab_size
        self.merges={}  # {(旧1, 旧2): 新id}
        self.vocab={}   # {id: bytes串}

    def train(self,text:str):
        # ① 初始：把文本转成字节 id 列表（每个字符一个 id）
        ids=list(text.encode("utf-8"))
        vocab={i:bytes([i]) for i in range (256)}
        # ② 循环：词表没到目标大小就继续合并
        while len(vocab)<self.vocab_size:
            # ③ 统计相邻对（用你答的 zip 方法！）
            counts=Counter(zip(ids,ids[1:]))
            if not counts:
                break
            # ④ 找出出现最多的对
            pair=max(counts,key=counts.get)
            # ⑤ 合并：把 ids 里所有 (a,b) 替换成新 id
            new_id=len(vocab)
            ids=self._merge(ids,pair,new_id)
            # ⑥ 记录规则
            self.merges[pair]=new_id
            vocab[new_id]=vocab[pair[0]] + vocab[pair[1]]
        self.vocab=vocab

    def _merge(self,ids,pair,new_id):
        """把 ids 里所有出现的 pair (a,b) 替换成 new_id（从左到右、不重叠）。"""
        new_ids=[]
        i=0
        while i<len(ids):
            if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
                new_ids.append(new_id)
                i=i+2
            else:
                new_ids.append(ids[i])
                i+=1
        return new_ids

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode('utf-8'))  # 初始：字符转字节 id
        # TODO: 从左到右扫描，遇到 (a,b) 在 merges 里就合并
        while len(ids)>=2:
            count=Counter(zip(ids,ids[1:]))
            pair=min(count, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx=self.merges[pair]
            ids=self._merge(ids,pair,idx)
        return ids

        # ---------- 解码 ----------

    def decode(self, ids: list[int]) -> str:
        # TODO: 每个 id 查 vocab，拼接字节，转回文本
        # 提示：b''.join(self.vocab[i] for i in ids).decode('utf-8')
        tokens=b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')
        return tokens
