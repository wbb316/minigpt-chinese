"""自研 BPE tokenizer v2（正则预分词 + 增量式合并训练）

v2 相比 v1 的两个变化：
1. GPT-2 风格预分词：文本先按 词块(\\w+，含中文/字母/数字) | 单个标点 | 空白串 切块，
   合并规则永不跨块 —— token 更规范（GPT-2/GPT-3 标准做法），encode 也按块更快。
2. train() 提速：v1 每轮合并都对整段文本全量重数相邻对（O(N×M)，
   4M 字符采样 × 6K 次合并 ≈ 1.5~2 小时）；
   v2 把语料建成"单链表（前驱/后继数组）"，每轮只做：
     ① numpy 向量化找出目标对 (a,b) 的所有出现位置
     ② 链表 O(1) 摘除合并（节点复用，不重建序列）
     ③ 只更新受影响邻接的频次（增量计数，不再全量重扫）
   —— 实测 200K 字符/vocab6144 ≈ 84s，4M 字符 ≈ 25~30min、2M ≈ 13~15min
   （若追求更快可改用 per-pair 出现索引，代价是额外内存）。

特殊 token / id 方案 / decode / save / load 与 v1 完全一致，对外 API 不变。
"""
import json
import re

UNK_ID = 0
EOS_ID = 1
SPECIAL_TOKENS = {UNK_ID: '<unk>', EOS_ID: '<eos>'}
NUM_SPECIAL = 2
SEP = -1    # 块间哨兵（节点值，不参与任何 pair）
DEAD = -2   # 已摘除节点的标记值

# GPT-2 风格预分词
PRETOK_RE = re.compile(r'\w+|[^\w\s]|\s+')


def _inc(cd: dict, key):
    cd[key] = cd.get(key, 0) + 1


def _dec(cd: dict, key):
    v = cd.get(key, 0) - 1
    if v > 0:
        cd[key] = v
    else:
        cd.pop(key, None)


class BPETokenizer:
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.merges = {}   # {(旧1, 旧2): 新id}
        self.vocab = {}    # {id: bytes串}（特殊token存名字字节）

        # 初始化特殊 token（固定不参与合并）
        self.vocab[UNK_ID] = b'<unk>'
        self.vocab[EOS_ID] = b'<eos>'

    # ---------------- 训练（预分词 + 链表增量） ----------------
    def train(self, text: str):
        import numpy as np

        vocab = {i + NUM_SPECIAL: bytes([i]) for i in range(256)}
        vocab.update(self.vocab)

        # ① 预分词: 块之间插 SEP 哨兵, 合并永不跨块
        seq = []
        for chunk in PRETOK_RE.findall(text):
            seq.extend(b + NUM_SPECIAL for b in chunk.encode('utf-8'))
            seq.append(SEP)

        n = len(seq)
        if n <= 1 or len(vocab) >= self.vocab_size:
            self.vocab = vocab
            return

        # ② 单链表: node id == 位置; prv/nxt 数组; 节点总数固定(合并复用节点)
        vals = np.array(seq, dtype=np.int32)                 # 节点当前值
        nxt = np.arange(1, n + 1, dtype=np.int32)            # nxt[i] = i+1
        nxt[n - 1] = -1
        prv = np.arange(-1, n - 1, dtype=np.int32)           # prv[i] = i-1

        # ③ 初始 pair 频次（只统计真实邻接, 跳过 -1 哨兵）
        counts = {}
        for i in range(n - 1):
            a, b = vals[i], vals[i + 1]
            if a >= 0 and b >= 0:
                _inc(counts, (int(a), int(b)))

        # ④ 循环合并直到词表满
        while len(vocab) < self.vocab_size:
            if not counts:
                break
            # 确定性平局规则: 频次最高; 并列时取 (a,b) 字典序最小者
            pair = max(counts, key=lambda k: (counts[k], -k[0], -k[1]))
            a, b = pair
            c = len(vocab)

            # 找 (a,b) 所有出现起点 u: vals[u]==a 且 vals[nxt[u]]==b
            safe = nxt >= 0
            nxt_safe = np.where(safe, nxt, 0)                # 防 -1 回绕
            cand = np.flatnonzero((vals == a) & safe & (vals[nxt_safe] == b))

            # 按链序(节点id递增)贪心合并, 不重叠; 合并只影响局部邻接
            for u in cand.tolist():
                if vals[u] != a:          # 受前面相邻合并影响或已摘除
                    continue
                v = nxt[u]
                if v < 0 or vals[v] != b:
                    continue
                w = prv[u]
                z = nxt[v]

                # 删除的邻接: (u,v)→(a,b); (w,u)→(vals[w],a); (v,z)→(b,vals[z])
                _dec(counts, (a, b))
                if w >= 0 and vals[w] != SEP:      # 哨兵值 -1 不参与 pair
                    _dec(counts, (int(vals[w]), a))
                if z >= 0 and vals[z] != SEP:
                    _dec(counts, (b, int(vals[z])))

                # 链表: 摘除 v, u 复用并改值为 c
                nxt[u] = z
                if z >= 0:
                    prv[z] = u
                vals[v] = DEAD
                vals[u] = c

                # 新增的邻接: (w,u)→(vals[w],c); (u,z)→(c,vals[z])
                if w >= 0 and vals[w] != SEP:
                    _inc(counts, (int(vals[w]), c))
                if z >= 0 and vals[z] != SEP:
                    _inc(counts, (c, int(vals[z])))

            self.merges[pair] = c
            vocab[c] = vocab[a] + vocab[b]

        self.vocab = vocab

    # ---------------- 编码（预分词, 块内栈式合并） ----------------
    def encode(self, text: str, verbose=False) -> list[int]:
        """文本 → token id 列表。按块编码: 合并永不跨块, 每块独立栈式合并。"""
        out = []
        for chunk in PRETOK_RE.findall(text):
            raw = chunk.encode('utf-8', errors='ignore')
            ids = [b + NUM_SPECIAL for b in raw]
            # <unk> 兜底
            ids = [i if i in self.vocab else UNK_ID for i in ids]

            # 栈: 压入后不断尝试合并栈顶两个（可连续合并）
            stack = []
            for token in ids:
                stack.append(token)
                while len(stack) >= 2:
                    nid = self.merges.get((stack[-2], stack[-1]))
                    if nid is None:
                        break
                    stack.pop()
                    stack.pop()
                    stack.append(nid)
            out.extend(stack)
        return out

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
