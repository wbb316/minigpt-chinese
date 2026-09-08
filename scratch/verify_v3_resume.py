"""验证 resume 路径：checkpoint_latest.pt 是完整包 {'model': ...带前缀...}，
_strip_compile_prefix 后应能 load 进 raw_gpt。"""
import sys
import os
import torch

ROOT = r'D:\WBB_Python\pytorch'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'train'))
from model.gpt import GPT  # noqa: E402

ckpt = os.path.join(ROOT, 'log', '50M参数_v3+998Mtokens', 'checkpoint_latest.pt')
ck = torch.load(ckpt, map_location='cpu', weights_only=True)
print('顶层 keys:', list(ck.keys()))
m = ck['model']
print('model 层带前缀:', any(k.startswith('_orig_mod.') for k in m))

# 复刻 train.py: raw_gpt.load_state_dict(_strip_compile_prefix(ck['model']))
from train import _strip_compile_prefix  # noqa: E402
sd = _strip_compile_prefix(m)
print('剥离后仍带前缀:', any(k.startswith('_orig_mod.') for k in sd))

n_layer = max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.')) + 1
n_embd = sd['token_emb.weight'].shape[1]
vocab = sd['token_emb.weight'].shape[0]
block = sd['pos_emb.pe'].shape[1]
pe = 'rope' if 'rope.cos_cached' in sd else 'sinusoidal'
print(f'解析: {n_layer}层/{n_embd}d/vocab{vocab}/block{block}/PE={pe}')

gpt = GPT(vocab_size=vocab, n_layer=n_layer, n_head=9, n_embd=n_embd,
          block_size=block, position_encoding=pe)
missing, unexpected = gpt.load_state_dict(sd)
print('missing:', missing if missing else '无', '| unexpected:', unexpected if unexpected else '无')
print('optimizer keys:', len(ck['optimizer']['state']))
print('step:', ck['step'], '| best_val:', round(ck['best_val'], 4))
print('\n✅ resume 包加载验证通过')
