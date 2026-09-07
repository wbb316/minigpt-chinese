# -*- coding: utf-8 -*-
"""本地对照：legacy 初始化（改前） vs GPT-2 初始化（改后）训练轨迹。

同数据同配置（同 seed 数据流），两个模型各自训练 N 步，记录 train loss。
验证：新初始化首步 ≈ ln(vocab)（无 300+ 爆炸），且前段下降更快/终值不差。
"""
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import BPETokenizer  # noqa: E402
from model.gpt import GPT  # noqa: E402

VOCAB, N_LAYER, N_HEAD, N_EMBD, BLOCK = 1024, 4, 4, 128, 64
BATCH, STEPS, LR = 8, 400, 1e-3


def make_legacy_init(gpt):
    """把（默认已 GPT-2 init 的）模型改回旧实现行为：
    Linear → PyTorch 默认 reset（kaiming uniform + bias uniform）；
    Embedding → N(0,1)（nn.Embedding 默认；tie 的 head.weight 同引用一并变）。
    """
    with torch.no_grad():
        for m in gpt.modules():
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()
        for m in gpt.modules():
            if isinstance(m, torch.nn.Embedding):
                m.weight.normal_(mean=0.0, std=1.0)


def corpus_tokens():
    random.seed(42)
    words = ['她', '说', '道', '望着', '窗外', '天空', '心里', '想着', '那个人',
             '夜色', '渐渐', '深了', '走进', '教室', '突然', '门', '被', '推开',
             '阳光', '照', '进来', '转身', '离去', '风', '吹过', '树梢',
             '为什么', '总觉得', '今天', '会', '有', '事情', '发生']
    text = ''.join(random.choice(words) for _ in range(60000))
    tok = BPETokenizer(vocab_size=VOCAB)
    tok.train(text, backend='fast')
    return torch.tensor(tok.encode(text), dtype=torch.long)


def train_one(gpt, tokens, tag):
    opt = torch.optim.AdamW(
        [p for p in {id(p): p for p in gpt.parameters()}.values()],
        lr=LR, weight_decay=0.05)
    n = tokens.numel()
    losses = []
    for step in range(STEPS):
        start = (step * BATCH * BLOCK) % (n - BLOCK - 1)
        idx = (start + torch.arange(BATCH * (BLOCK + 1))) % n
        idx = idx.view(BATCH, BLOCK + 1)
        xb = tokens[idx[:, :-1]]
        yb = tokens[idx[:, 1:]]
        opt.zero_grad()
        logits = gpt(xb)
        loss = F.cross_entropy(logits.view(-1, VOCAB), yb.view(-1))
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == STEPS - 1:
            losses.append((step, loss.item()))
    return losses


def main():
    torch.manual_seed(0)
    tokens = corpus_tokens()
    print(f'tokens: {len(tokens)}, 训练 {STEPS} 步 × batch{BATCH}×block{BLOCK}')

    results = {}
    for tag, legacy in (('legacy(旧,改前)', True), ('gpt2(新,改后)', False)):
        torch.manual_seed(0)
        gpt = GPT(vocab_size=VOCAB, block_size=BLOCK, n_layer=N_LAYER,
                  n_head=N_HEAD, n_embd=N_EMBD, dropout=0.0,
                  tie_embeddings=True)
        if legacy:
            make_legacy_init(gpt)
        # 首步 loss（训练前）
        idx = torch.randint(0, VOCAB, (BATCH, BLOCK))
        with torch.no_grad():
            ce0 = F.cross_entropy(gpt(idx).view(-1, VOCAB),
                                  torch.randint(0, VOCAB, (BATCH, BLOCK)).view(-1))
        losses = train_one(gpt, tokens, tag)
        results[tag] = (ce0.item(), losses)
        print(f'[{tag}] initial CE = {ce0.item():.2f} (基线 ln{VOCAB}={__import__("math").log(VOCAB):.2f})')

    print('\n===== loss 轨迹对比（每 10 步）=====')
    print(f'{"step":>6s} {"legacy(旧)":>10s} {"gpt2(新)":>10s}  差(新-旧)')
    la = dict(results['legacy(旧,改前)'][1])
    lb = dict(results['gpt2(新,改后)'][1])
    for s in sorted(set(la) | set(lb)):
        a, b = la.get(s, float('nan')), lb.get(s, float('nan'))
        print(f'{s:>6d} {a:>10.3f} {b:>10.3f}  {b - a:>+9.3f}')

    fa = la[max(la)]
    fb = lb[max(lb)]
    print(f'\n最终 loss: legacy={fa:.3f}  gpt2={fb:.3f}  差 {fb - fa:+.3f}')
    print(f'结论: 新初始化 首步 {results["gpt2(新,改后)"][0]:.1f}（正常） vs '
          f'旧 {results["legacy(旧,改前)"][0]:.1f}（爆炸）')


if __name__ == '__main__':
    main()
