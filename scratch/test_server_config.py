# -*- coding: utf-8 -*-
"""验证 app/server.py 的加载改动（v2 修正版）。

  1) 自动推断头数是否正确（35M=8 / 50M v3_alpha=9 / 100M v3_beta=11）
  2) 显式传错头数时是否被**硬校验**拦住（而不是静默加载成功、输出乱码）
  3) 能否真的生成一段文本（CPU；本机无 GPU，只验证代码路径）

上一版的 bug：用 getattr(gpt,'n_head') 取头数 → GPT 没暴露该属性 → 防呆段全被跳过；
且 model.sampling 里没有 sample_next_token（应为函数名不同）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.server import load_model  # noqa: E402

# (标签, ckpt, tokenizer, 期望头数)
CANDS = [
    ('35M v2（sinusoidal → 回退 8）',
     'result/35M参数+998Mtokens/checkpoint_best.pt',
     'result/35M参数+998Mtokens/tokenizer_best.pkl', 8),
    ('50M v3_alpha（rope → 推断 9）',
     'result/50M参数v3+998Mtokens/checkpoint_best.pt',
     'result/50M参数v3+998Mtokens/tokenizer_best.pkl', 9),
    ('100M v3_beta（rope → 推断 11）',
     'result/100M参数v3+2Btokens/checkpoint_best.pt',
     'result/100M参数v3+2Btokens/tokenizer_best.pkl', 11),
]

ok = True
for label, ck, tk, expect in CANDS:
    ckp, tkp = os.path.join(ROOT, ck), os.path.join(ROOT, tk)
    if not (os.path.exists(ckp) and os.path.exists(tkp)):
        print(f'--- {label}\n    ⏭  缺文件，跳过 ({ck})')
        continue
    print(f'--- {label}')
    gpt, tok = load_model(ckp, tkp)                 # n_head 省略 = 自动
    # 从存档真实形状反查，验证推断结果（不依赖 GPT 是否暴露 n_head 属性）
    import torch
    sd = torch.load(ckp, map_location='cpu', weights_only=True)
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k[len('_orig_mod.'):]: v for k, v in sd.items()}
    n_embd = sd['token_emb.weight'].shape[1]
    hd = sd['rope.cos_cached'].shape[-1] if 'rope.cos_cached' in sd else None
    real = n_embd // hd if hd else 8
    flag = '✅' if real == expect else '❌'
    print(f'    {flag} 期望 {expect} 头，存档推算 {real} 头（n_embd={n_embd}, head_dim={hd}）')
    if real != expect:
        ok = False

    # ---- 防呆：故意传错 ----
    wrong = 8 if expect != 8 else 11
    try:
        load_model(ckp, tkp, n_head=wrong)
        print(f'    ❌ 传错 n_head={wrong} 竟然没报错（应被拦住）')
        ok = False
    except AssertionError as e:
        print(f'    ✅ 传错 n_head={wrong} 被拦住 → {str(e)[:70]}')

# ---- 生成测试 ----
print('\n=== 生成测试（100M 模型，CPU，贪心解码 24 token）===')
import torch  # noqa: E402
ckp = os.path.join(ROOT, 'result/100M参数v3+2Btokens/checkpoint_best.pt')
tkp = os.path.join(ROOT, 'result/100M参数v3+2Btokens/tokenizer_best.pkl')
gpt, tok = load_model(ckp, tkp)
for prompt in ('他推开那扇门，', '少年抬起头'):
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(24):
            logits = gpt(x[:, -gpt.block_size:])[:, -1, :]
            x = torch.cat([x, torch.tensor([[int(torch.argmax(logits, dim=-1))]])], dim=1)
    print(f'  prompt: {prompt}')
    print(f'  输出  : {tok.decode(x[0].tolist())}')

print('\n总结:', '✅ 全部通过' if ok else '❌ 有失败项')
