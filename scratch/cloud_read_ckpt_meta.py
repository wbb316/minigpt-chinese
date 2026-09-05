"""读云端 checkpoint metadata（不加载模型权重，只读 dict 顶层 + optimizer 状态）。"""
import torch

for f in ['/root/result_webnovel_v2/checkpoint_latest.pt',
          '/root/result_webnovel_v2/checkpoint_best.pt']:
    print('=' * 20, f)
    ck = torch.load(f, map_location='cpu', weights_only=True)
    keys = list(ck.keys())
    print('top-level keys:', keys)
    for k in keys:
        if k in ('model', 'optimizer', 'scaler'):
            continue
        print(f'  {k} = {ck[k]!r}')
    if 'optimizer' in ck:
        st = ck['optimizer']
        print('  optimizer keys:', list(st.keys()))
        pg = st.get('param_groups')
        if pg:
            g = pg[0]
            print('  param_groups[0]: lr =', g.get('lr'),
                  '| initial_lr =', g.get('initial_lr'),
                  '| betas =', g.get('betas'),
                  '| wd =', g.get('weight_decay'))
        state = st.get('state', {})
        if state:
            first = next(iter(state.values()))
            print('  state sample keys:', list(first.keys()))
            print('  step in state?', 'step' in first, '->', first.get('step'))
    if 'scaler' in ck and ck['scaler'] is not None:
        sc = ck['scaler']
        print('  scaler keys:', list(sc.keys()))
        for k2 in sc:
            if k2 != 'per_optimizer_states':
                print(f'    {k2} = {sc[k2]!r}')
