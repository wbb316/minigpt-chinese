"""MiniGPT-Chinese 训练脚本（优化版）

相对旧版的变化（对应"训练优化方案"）：
- 模型 7层→6层（默认），降低 容量:数据 比，缓解泛化瓶颈
- warmup 500 步 + cosine 衰减的学习率调度（收敛到平坦处）
- 正则化增强：FFN dropout + embedding dropout（dropout=0.1），weight_decay 0.01→0.05
- 步级验证：每 val_every 步验证一次，抓 epoch 内真实最优点；连续 patience 次不改善即早停
- AMP 混合精度保持（无 GPU 时自动退回 FP32）

用法：
    python train/train.py                       # 默认配置（6层/8头/256维/128上下文）
    python train/train.py --max-steps 200       # 冒烟测试：只跑 200 步
    python train/train.py --n-layer 4 --lr 3e-4 # 自定义
"""
import os
# ★ 必须在 import torch 前设置线程数（否则 OMP_NUM_THREADS 报错，训练退化为单核）
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

import argparse
import math
import pickle
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ★ 让 Python 能找到上级目录的 data/model 包（train.py 在子目录运行）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.tokenizer import BPETokenizer       # noqa: E402
from data.dataset import TextDataset          # noqa: E402
from model.gpt import GPT                     # noqa: E402


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def main():
    parser = argparse.ArgumentParser(description='MiniGPT-Chinese 训练')
    parser.add_argument('--train-txt', default='data/train.txt')
    parser.add_argument('--val-txt', default='data/val.txt')
    parser.add_argument('--out-dir', default='result')
    parser.add_argument('--cache-dir', default='data')
    parser.add_argument('--vocab-size', type=int, default=256 + 3000)   # 3256
    parser.add_argument('--tokens-sample', type=int, default=2_000_000, help='训练分词器用采样字符数')
    # 模型
    parser.add_argument('--n-layer', type=int, default=6)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-embd', type=int, default=256)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    # 优化
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup-steps', type=int, default=500)
    # 验证 / 早停（步级）
    parser.add_argument('--val-every', type=int, default=5000, help='每多少步验证一次')
    parser.add_argument('--patience', type=int, default=2, help='连续几次验证不改善就早停')
    # 其他
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=0, help='最多训练多少步（0=不限，用于冒烟测试）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    print(f'使用设备: {device}  (AMP: {use_amp})')

    out_dir = resolve(args.out_dir)
    cache_dir = resolve(args.cache_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # ---------- 读取语料 ----------
    with open(resolve(args.train_txt), encoding='utf-8') as f:
        train_text = f.read()
    with open(resolve(args.val_txt), encoding='utf-8') as f:
        val_text = f.read()
    print(f'训练文本：{len(train_text)} 字符，验证文本：{len(val_text)} 字符')

    # ---------- 分词器（用训练文本采样训练） ----------
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    n_parts = max(1, len(train_text) // max(1, args.tokens_sample // 100))
    sample_parts = [train_text[s:s + args.tokens_sample // 100]
                    for s in range(0, len(train_text), max(1, len(train_text) // n_parts))]
    sample_text = ''.join(sample_parts)
    if len(sample_text) > args.tokens_sample:
        sample_text = sample_text[:args.tokens_sample]
    tokenizer.train(sample_text)
    print(f'tokenizer词表大小：{len(tokenizer.vocab)}')

    # ---------- 编码（带缓存，缓存名含语料长度，语料变了自动失效） ----------
    vocab_size = len(tokenizer.vocab)

    def get_tokens(text, name):
        cache_path = os.path.join(cache_dir, f'tokens_{name}_v{vocab_size}_len{len(text)}.npy')
        if os.path.exists(cache_path):
            tokens = np.load(cache_path).tolist()
            print(f'从缓存加载 {name}: {len(tokens)} token')
            return tokens
        tokens = tokenizer.encode(text)
        np.save(cache_path, np.array(tokens))
        print(f'编码并缓存 {name}: {len(text)} 字符 → {len(tokens)} token')
        return tokens

    train_tokens = get_tokens(train_text, 'train')
    val_tokens = get_tokens(val_text, 'val')
    print(f'训练集 {len(train_tokens)} token, 验证集 {len(val_tokens)} token')

    # ---------- 数据集 / DataLoader（滑动窗口 stride=1，标准做法，不改） ----------
    train_ds = TextDataset(tokens=train_tokens, block_size=args.block_size)
    val_ds = TextDataset(tokens=val_tokens, block_size=args.block_size)
    loader_kwargs = dict(batch_size=args.batch_size)
    if args.num_workers > 0:
        loader_kwargs.update(num_workers=args.num_workers, pin_memory=True,
                             prefetch_factor=4, persistent_workers=True)
    else:
        loader_kwargs.update(num_workers=0)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)
    steps_per_epoch = len(train_loader)
    print(f'训练集样本数: {len(train_ds)}，验证集样本数: {len(val_ds)}，'
          f'每批 {args.batch_size}，每 epoch {steps_per_epoch} 步')

    # ---------- 模型 ----------
    gpt = GPT(vocab_size=vocab_size, block_size=args.block_size, n_layer=args.n_layer,
              n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    gpt = gpt.to(device)
    print(f'GPT 参数量: {gpt.get_num_params():,}')

    optimizer = torch.optim.AdamW(gpt.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------- warmup + cosine 调度（lambda 收到的是已调用 step 的次数） ----------
    total_steps = steps_per_epoch * args.epochs
    warmup = args.warmup_steps

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup                       # 线性升到 1.0
        t = (step - warmup) / max(1, total_steps - warmup)   # 0 → 1
        t = min(t, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))           # cosine 衰减到接近 0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ---------- 验证 ----------
    def evaluate():
        gpt.eval()
        total, cnt = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast('cuda', enabled=use_amp):
                    logits = gpt(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                total += loss.item()
                cnt += 1
        gpt.train()
        return total / cnt

    # ---------- 训练主循环（步级验证 + 早停） ----------
    best_val = float('inf')
    no_improve = 0
    global_step = 0
    best_path = os.path.join(out_dir, 'checkpoint_best.pt')
    tok_best_path = os.path.join(out_dir, 'tokenizer_best.pkl')
    latest_path = os.path.join(out_dir, 'checkpoint_latest.pt')
    tok_latest_path = os.path.join(out_dir, 'tokenizer_latest.pkl')

    def save_checkpoint(tag, ckpt_path, tok_path):
        torch.save(gpt.state_dict(), ckpt_path)
        with open(tok_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        tqdm.write(f'  💾 已保存 {tag}: {ckpt_path}')

    stopped = False
    for epoch in range(args.epochs):
        gpt.train()
        running, steps_here = 0.0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', unit='step', ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.autocast('cuda', enabled=use_amp):
                logits = gpt(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            running += loss.item()
            steps_here += 1
            global_step += 1
            pbar.set_postfix(loss=f'{loss.item():.3f}',
                             lr=f'{scheduler.get_last_lr()[0]:.1e}')

            # 冒烟测试：步数上限
            if args.max_steps and global_step >= args.max_steps:
                stopped = True
                break

            # ★ 步级验证
            if global_step % args.val_every == 0:
                val_loss = evaluate()
                tqdm.write(f'\n===== step {global_step} (epoch {epoch}) '
                           f'train loss: {running / steps_here:.3f}, '
                           f'val loss: {val_loss:.3f}, lr: {scheduler.get_last_lr()[0]:.1e} =====')
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    save_checkpoint('最佳模型', best_path, tok_best_path)
                    tqdm.write(f'  ✓ 新最好模型已保存 (val {val_loss:.3f})')
                else:
                    no_improve += 1
                    tqdm.write(f'  ⚠️ val 未改善 ({no_improve}/{args.patience})')
                save_checkpoint('最近模型', latest_path, tok_latest_path)  # 防中断白跑

                if no_improve >= args.patience:
                    tqdm.write(f'🚫 早停: val 连续 {args.patience} 次验证未改善，停止训练')
                    stopped = True
                    break
        pbar.close()

        avg = running / max(steps_here, 1)
        tqdm.write(f'epoch {epoch} 平均 train loss: {avg:.3f} '
                   f'(已完成 {global_step}/{total_steps} 步)')
        if stopped:
            break

    tqdm.write(f'训练结束。最佳模型: {best_path} (val {best_val:.3f})')


if __name__ == '__main__':
    main()
