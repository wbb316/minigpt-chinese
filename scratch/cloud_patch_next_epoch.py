"""给旧版 checkpoint 补 next_epoch 字段（Epoch1 完整跑完 → next_epoch=1）。

只修改 metadata dict（加键），权重/优化器 tensor 原样保留。
用法: python cloud_patch_next_epoch.py <path> [next_epoch]
"""
import sys
import torch

path = sys.argv[1]
next_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 1

ck = torch.load(path, map_location='cpu', weights_only=True)
print('BEFORE:', {k: ck[k] for k in ck if k not in ('model', 'optimizer', 'scaler')})
has_next = ck.get('next_epoch') is not None
if has_next:
    print(f'已有 next_epoch={ck["next_epoch"]}，无需补丁')
    sys.exit(0)
ck['next_epoch'] = next_epoch
torch.save(ck, path)
ck2 = torch.load(path, map_location='cpu', weights_only=True)
print('AFTER :', {k: ck2[k] for k in ck2 if k not in ('model', 'optimizer', 'scaler')})
print('补丁完成: next_epoch =', next_epoch, '(权重未改动)')
