"""MiniGPT-Chinese 注意力热力图可视化

用法:
    python visualize_attention.py                          # 默认: 第3层 第0个头, 输入"我喜欢她，她也喜欢我"
    python visualize_attention.py --layer 5 --head 3
    python visualize_attention.py --text "她轻轻握住我的手" --layer 0 --head 7

说明:
    - 不改 model/ 任何代码: 用 register_forward_hook 挂到目标层的 MultiHeadAttention 上,
      在 hook 里用该层自己的 q/k/v 权重重算注意力权重 (与 forward 内部 softmax 后的结果一致)
    - 模型架构从 checkpoint state_dict 自动推断 (层数/维度/词表/block_size),
      只有注意力头数无法从权重推断, 默认 8 (可通过 --n-head 覆盖)
输出:
    result/attention_heatmap_L{layer}_H{head}.png
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')  # 无显示环境, 直接存图
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import torch.nn.functional as F

# 让脚本在任何目录下运行都能找到 model / data 包
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.gpt import GPT  # noqa: E402


# ---------------------------------------------------------------- 中文字体
def setup_chinese_font():
    """配置 matplotlib 中文字体 (Windows: 微软雅黑/黑体; 其他系统常见 CJK 字体)。"""
    candidates = [
        'Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun',
        'PingFang SC', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'AR PL UMing CN',
    ]
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f'✓ 使用中文字体: {name}')
            return name
        except Exception:
            continue
    print('⚠ 未找到中文字体, 中文标签可能显示为方框')
    return None


# ---------------------------------------------------------------- 模型加载
def load_model(ckpt_path, tok_path, n_head, device):
    """加载 checkpoint + tokenizer, 架构从 state_dict 自动推断。"""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    # torch.compile 训练的 checkpoint 带 '_orig_mod.' 前缀（OptimizedModule 痕迹），
    # 剥离后所有解析逻辑按无前缀 key 统一处理（旧存档本来无前缀，兼容）
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k[len('_orig_mod.'):]: v for k, v in sd.items()}

    n_layer = max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.')) + 1
    n_embd = sd['token_emb.weight'].shape[1]
    vocab_size = sd['token_emb.weight'].shape[0]
    block_size = sd['pos_emb.pe'].shape[1]  # PositionalEncoding 的 buffer 记录了训练时的序列长度
    pe = 'rope' if 'rope.cos_cached' in sd else 'sinusoidal'   # 自动检测
    print(f'从 checkpoint 推断架构: {n_layer}层 / {n_head}头 / {n_embd}维, '
          f'词表{vocab_size}, block_size={block_size}, 位置编码: {pe}')

    gpt = GPT(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
              n_embd=n_embd, block_size=block_size,
              position_encoding=pe)
    gpt.load_state_dict(sd)
    gpt.to(device).eval()

    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f'加载分词器: 词表 {len(tokenizer.vocab)}')

    return gpt, tokenizer


# ---------------------------------------------------------------- 注意力提取
def extract_attention(gpt, layer, head, tokens, device):
    """前向一次, 返回第 layer 层第 head 个头的注意力矩阵 (T, T)。

    用 hook 捕获 MultiHeadAttention 的输入 x, 然后用模块自己的权重重算
    qk^T -> 因果mask -> softmax, 与 forward 内部的结果完全一致
    (eval 模式下 dropout 是恒等映射, 不影响)。
    """
    assert 0 <= layer < len(gpt.blocks), f'layer {layer} 超出范围 0..{len(gpt.blocks) - 1}'
    attn_mod = gpt.blocks[layer].attn
    assert 0 <= head < attn_mod.n_heads, f'head {head} 超出范围 0..{attn_mod.n_heads - 1}'

    captured = {}

    def hook(module, inp, out):
        x = inp[0]                       # (B, T, C), 即 Block 里 ln1 之后的输入
        B, T, C = x.shape
        q = module.q(x).view(B, T, module.n_heads, module.head_dim).transpose(1, 2)
        k = module.k(x).view(B, T, module.n_heads, module.head_dim).transpose(1, 2)
        # 因果 mask: 位置 t 只能看 <= t (与 attention.py 的默认行为一致)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.masked_fill(~mask[None, None, :T, :T], float('-inf'))
        attn = torch.softmax(attn, dim=-1)   # (B, H, T, T), 每行和为 1
        captured['attn'] = attn[0, head]     # (T, T)

    handle = attn_mod.register_forward_hook(hook)
    idx = torch.tensor(tokens[-gpt.block_size:], device=device).unsqueeze(0)
    try:
        with torch.no_grad():
            gpt(idx)
    finally:
        handle.remove()

    return captured['attn'].cpu().numpy()  # (T, T)


# ---------------------------------------------------------------- 画图
def plot_heatmap(attn, token_labels, layer, head, save_path):
    T = attn.shape[0]
    fig, ax = plt.subplots(figsize=(max(8.0, T * 0.62), max(7.0, T * 0.56)))
    im = ax.imshow(attn, cmap='Blues', vmin=0.0, vmax=float(attn.max()))
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(token_labels, rotation=90, fontsize=10)
    ax.set_yticklabels(token_labels, fontsize=10)
    ax.tick_params(axis='x', which='major', pad=2)
    ax.set_xlabel('被注意的词 (key)', fontsize=12)
    ax.set_ylabel('查询的词 (query)', fontsize=12)
    ax.set_title(f'Attention Heatmap — Layer {layer} / Head {head} '
                 f'(max={attn.max():.3f})', fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('注意力权重', fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ 热力图已保存: {save_path}')


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description='可视化 GPT 注意力热力图')
    parser.add_argument('--text', default='我喜欢她，她也喜欢我', help='输入的中文提示')
    parser.add_argument('--layer', type=int, default=3, help='要可视化的层 (0~n_layer-1, v2 为 10 层)')
    parser.add_argument('--head', type=int, default=0, help='要可视化的头')
    parser.add_argument('--n-head', type=int, default=8,
                        help='模型注意力头数 (权重中无法推断, 训练时为 8)')
    parser.add_argument('--ckpt', default='result/35M参数+998Mtokens/checkpoint_best.pt')
    parser.add_argument('--tokenizer', default='result/35M参数+998Mtokens/tokenizer_best.pkl')
    parser.add_argument('--out-dir', default='result/35M参数+998Mtokens')
    args = parser.parse_args()

    setup_chinese_font()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    # 相对路径基于脚本所在目录解析
    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    ckpt_path = resolve(args.ckpt)
    tok_path = resolve(args.tokenizer)
    out_dir = resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    gpt, tokenizer = load_model(ckpt_path, tok_path, args.n_head, device)

    # 编码输入
    tokens = tokenizer.encode(args.text)
    print(f'输入: "{args.text}"')
    print(f'编码为 {len(tokens)} 个 token: {tokens}')
    labels = [tokenizer.decode([t]) for t in tokens]
    # 空白/乱码 token 换成可读占位, 避免标签空着
    labels = [l if l.strip() else f'<{t}>' for l, t in zip(labels, tokens)]
    for i, (t, l) in enumerate(zip(tokens, labels)):
        print(f'  [{i}] id={t:4d} → "{l}"')

    if len(tokens) > gpt.block_size:
        print(f'⚠ 输入超过 block_size({gpt.block_size}), 只保留最后 {gpt.block_size} 个 token')

    attn = extract_attention(gpt, args.layer, args.head, tokens, device)
    T = attn.shape[0]

    # 打印注意力最高的几对 (query -> key), 验证"有深浅对比"
    pairs = []
    for q in range(T):
        for k in range(T):
            pairs.append((attn[q, k], q, k))
    pairs.sort(reverse=True)
    print(f'\nTop-5 注意力最高的 (query → key):')
    for w, q, k in pairs[:5]:
        if q >= k:  # 因果: 只看有效位置
            print(f'  "{labels[q]}" → "{labels[k]}": {w:.3f}')

    row_sums = attn.sum(axis=1)
    print(f'每行和 (应≈1): min={row_sums.min():.3f} max={row_sums.max():.3f}')

    save_path = os.path.join(out_dir, f'attention_heatmap_L{args.layer}_H{args.head}.png')
    plot_heatmap(attn, labels, args.layer, args.head, save_path)


if __name__ == '__main__':
    main()
