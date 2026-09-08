import torch

path = r'D:\WBB_Python\pytorch\log\50M参数_v3+998Mtokens\checkpoint_best.pt'
sd = torch.load(path, map_location='cpu', weights_only=True)

# 模拟 generate.py / app/server.py / visualize_attention.py 的解析逻辑
try:
    n_layer = max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.')) + 1
    print('n_layer 解析 OK:', n_layer)
except Exception as e:
    print('架构解析失败:', type(e).__name__, str(e)[:80])

try:
    sd['token_emb.weight']
    print('token_emb 访问 OK')
except Exception as e:
    print('token_emb 访问失败:', type(e).__name__, str(e)[:80])

print("rope 检测 ('rope.cos_cached' in sd):", 'rope.cos_cached' in sd)
print("带 _orig_mod 前缀的 key:", [k for k in sd if k.startswith('_orig_mod')][:3], '...')
