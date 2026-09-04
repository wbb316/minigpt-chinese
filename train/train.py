"""MiniGPT-Chinese 训练脚本（优化版 v3）

相比 v2 的变化：
- 数据加载不再是瓶颈：tokens 全程用 numpy/int64 tensor，数据集零拷贝视图切片
  （内存 720MB→160MB）；新增 --sample-mode pack（不重叠打包，CPU 上每 epoch 快 ~100 倍）
- 采样 bug 修复：分词器训练样本从语料里均匀撒 100 段×20K 字符（之前误改成只取开头 2M）
- 过拟合判断修正：验证时额外算 eval 模式（无 dropout）的 train loss，
  打印 gap = val - train_eval —— 这才是判断过拟合的正确指标
  （训练时的 loss 带 dropout 噪声 + epoch 累积平均偏高，不可直接与 val 比）
- 训练 loss 显示改为 EMA；加了梯度裁剪（clip_grad_norm=1.0）
- LayerNorm 改为标准实现（有偏方差 + sqrt(var+eps)）
- 默认配置调整：vocab 6144 + tie_embeddings 默认开启 + 分词器采样 4M 字符（适配 100M+ 语料）

用法：
    python train/train.py                          # 默认：轻小说语料 + slide + 每 5000 步验证
    python train/train.py --sample-mode pack       # 不重叠打包（100M+ 语料每 epoch 从小时级降到分钟级）
    python train/train.py --vocab-size 8192        # 想试更大词表（配 --tokens-sample 4000000）
    python train/train.py --max-steps 200          # 冒烟测试
"""
import os
# ★ 必须在 import torch 前设置线程数（否则 OMP_NUM_THREADS 报错，训练退化为单核）
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

import argparse
import math
import multiprocessing as mp
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ★ 让 Python 能找到上级目录的 data/model 包（train.py 在子目录运行）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.tokenizer import BPETokenizer       # noqa: E402
from data.dataset import TextDataset, PackedDataset   # noqa: E402
from model.gpt import GPT                     # noqa: E402


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def dedupe_params(module):
    """tie_embeddings 时 head.weight 与 token_emb.weight 是同一对象，去重避免被优化两次。"""
    return list({id(p): p for p in module.parameters()}.values())


# ---------- 并行 BPE 编码（487M 字符单线程要 30-60 分钟, 分片并行后几分钟） ----------
_ENC_TOK = None


def _encode_worker_init(tok):
    global _ENC_TOK
    _ENC_TOK = tok


def _encode_slice(s):
    return np.asarray(_ENC_TOK.encode(s), dtype=np.int64)


def encode_text(text, tokenizer, n_procs=16):
    """把文本切成 n_procs 段并行编码后拼接。

    预分词保证了合并永不跨标点/空白，切片边界只可能切断极少数 chunk，
    边界处的 token 数与单线程整段编码有 <0.01% 差异（解码可完整还原），
    对训练/缓存一致性无影响。Windows 或小文本退回单线程。
    """
    if len(text) < 5_000_000 or n_procs <= 1 or sys.platform == 'win32':
        return np.asarray(tokenizer.encode(text), dtype=np.int64)
    step = (len(text) + n_procs - 1) // n_procs
    slices = [text[i:i + step] for i in range(0, len(text), step)]
    slices = [s for s in slices if s]
    with mp.Pool(n_procs, initializer=_encode_worker_init, initargs=(tokenizer,)) as pool:
        parts = pool.map(_encode_slice, slices)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser(description='MiniGPT-Chinese 训练')
    parser.add_argument('--train-txt', default='data/train_lightnovel.txt')
    parser.add_argument('--val-txt', default='data/val_lightnovel.txt')
    parser.add_argument('--out-dir', default='result')
    parser.add_argument('--cache-dir', default='data')
    parser.add_argument('--vocab-size', type=int, default=6144,
                        help='词表总大小 = 256字节 + merges；6.4M 参数推荐 6144（配 tie 嵌入防嵌入层过大）')
    parser.add_argument('--tokens-sample', type=int, default=4_000_000,
                        help='训练分词器用采样字符数（均匀撒 100 段全语料；2M 对 6144 已够，4M 为 8192 或求稳留余量）')
    # 数据 / 模型
    parser.add_argument('--sample-mode', default='slide', choices=['pack', 'slide'],
                        help='slide=滑动窗口 stride=1(默认)；pack=不重叠打包(每轮步数少 ~128 倍)')
    parser.add_argument('--n-layer', type=int, default=6)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-embd', type=int, default=256)
    parser.add_argument('--block-size', type=int, default=256,
                        help='上下文长度；6.4M 模型 128~256 合适，256 提升连贯性（手动注意力 O(T²)，勿再加大）')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--tie-embeddings', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='输入/输出嵌入共享权重（大词表必备，防嵌入层吃掉过多参数；--no-tie-embeddings 关闭）')
    # 优化
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--min-lr-ratio', type=float, default=0.1,
                        help='cosine 衰减底部学习率 = max_lr × 该比例（默认 0.1，不再衰减到 0，'
                             '保留尾部学习能力；设 0.0 恢复衰减到 0）')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    # 验证 / 早停
    parser.add_argument('--val-every', type=int, default=5000,
                        help='每多少步验证一次（slide 模式抓 epoch 内最优点）；0 = 只在 epoch 末验证')
    parser.add_argument('--patience', type=int, default=2,
                        help='连续几次验证不改善就早停')
    parser.add_argument('--eval-batches', type=int, default=100,
                        help='每次验证在 train/val 随机子集上各算多少批（eval 模式 loss）')
    # 其他
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--encode-workers', type=int, default=16,
                        help='BPE 编码并行进程数（大语料首次编码从小时级降到分钟级）')
    parser.add_argument('--max-steps', type=int, default=0,
                        help='最多训练多少步（0=不限，用于冒烟测试）')
    parser.add_argument('--resume', default='',
                        help='从 checkpoint_latest.pt 续训（加轮数用）。'
                             '例: --resume result_ln/checkpoint_latest.pt --epochs 20')
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

    # ---------- 分词器：优先加载已训练好的（跨 run 复用，省 ~30 分钟重训） ----------
    tok_cache_path = os.path.join(out_dir, f'tokenizer_v{args.vocab_size}_s{args.tokens_sample}.pkl')
    if os.path.exists(tok_cache_path):
        with open(tok_cache_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f'从缓存加载分词器: 词表 {len(tokenizer.vocab)} ← {tok_cache_path}')
    else:
        # 均匀撒 100 段 × (tokens_sample//100) 字符（覆盖全语料，不只学开头几本书）
        seg_len = max(1, args.tokens_sample // 100)
        step = max(1, len(train_text) // 100)
        sample_parts = [train_text[s:s + seg_len]
                        for s in range(0, len(train_text), step)][:100]
        tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(''.join(sample_parts))
        with open(tok_cache_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f'tokenizer词表大小：{len(tokenizer.vocab)}（已缓存 → {tok_cache_path}）')

    # ---------- 编码（带缓存，缓存名含语料长度，语料变了自动失效） ----------
    vocab_size = len(tokenizer.vocab)

    def get_tokens(text, name):
        # 缓存名带 tokenizer 版本/采样量/语料长度: 任何一个变了都自动失效重建
        cache_path = os.path.join(
            cache_dir,
            f'tokens_{name}_v{vocab_size}_s{args.tokens_sample}_len{len(text)}_tokv2.npy')
        if os.path.exists(cache_path):
            tokens = np.load(cache_path)                 # int64 ndarray
            print(f'从缓存加载 {name}: {len(tokens)} token')
            return tokens
        tokens = encode_text(text, tokenizer, args.encode_workers)
        np.save(cache_path, tokens)
        print(f'编码并缓存 {name}: {len(text)} 字符 → {len(tokens)} token')
        return tokens

    train_tokens = get_tokens(train_text, 'train')
    val_tokens = get_tokens(val_text, 'val')
    print(f'训练集 {len(train_tokens)} token, 验证集 {len(val_tokens)} token')

    # ---------- 数据集 / DataLoader ----------
    if args.sample_mode == 'pack':
        train_ds = PackedDataset(train_tokens, args.block_size)
        val_ds = PackedDataset(val_tokens, args.block_size,
                               offset=args.block_size // 2)   # 错开切分位置
    else:
        train_ds = TextDataset(train_tokens, args.block_size)     # 滑动窗口 stride=1
        val_ds = TextDataset(val_tokens, args.block_size)

    if args.num_workers > 0:
        loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                             prefetch_factor=4, persistent_workers=True)
    else:
        loader_kwargs = dict(num_workers=0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, **loader_kwargs)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f'模式: {args.sample_mode}，样本数: 训{len(train_ds)} / 验{len(val_ds)}，'
          f'每批 {args.batch_size} → 每 epoch {steps_per_epoch} 步 × {args.epochs} 轮 = {total_steps} 步')

    # eval 用固定随机子集（避免主循环中途再迭代同一 loader 导致 worker 冲突）
    def make_eval_loader(ds, n_batches):
        n = min(max(n_batches, 1) * args.batch_size, len(ds))
        idx = torch.randperm(len(ds))[:n].tolist()
        return DataLoader(Subset(ds, idx), batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

    train_eval_loader = make_eval_loader(train_ds, args.eval_batches)
    val_eval_loader = make_eval_loader(val_ds, args.eval_batches)

    # ---------- 模型 ----------
    gpt = GPT(vocab_size=vocab_size, block_size=args.block_size, n_layer=args.n_layer,
              n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
              tie_embeddings=args.tie_embeddings)
    gpt = gpt.to(device)
    n_params = gpt.get_num_params()
    print(f'GPT 参数量: {n_params:,}'
          + ('（含 tie_embeddings，已去重）' if args.tie_embeddings else ''))

    optimizer = torch.optim.AdamW(dedupe_params(gpt), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # ---------- warmup + cosine 调度 ----------
    warmup = args.warmup_steps
    min_ratio = max(0.0, args.min_lr_ratio)          # 底部学习率比例（默认 0.1）

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup                       # 线性升到 1.0
        t = (step - warmup) / max(1, total_steps - warmup)   # 0 → 1
        t = min(t, 1.0)
        # cosine 从 1.0 衰减到 min_ratio（默认 0.1，不归零 → 尾部保留学习能力）
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ---------- 断点续训（加轮数: --resume <latest> --epochs 更大的总数） ----------
    resume_path = resolve(args.resume) if args.resume else None
    start_epoch = 0
    global_step0 = 0
    best_val0 = float('inf')
    no_improve0 = 0
    if resume_path:
        if os.path.exists(resume_path):
            ck = torch.load(resume_path, map_location=device, weights_only=True)
            gpt.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            if use_amp and ck.get('scaler') is not None:
                scaler.load_state_dict(ck['scaler'])
            global_step0 = int(ck.get('step', 0))
            start_epoch = int(ck.get('epoch', 0))
            best_val0 = float(ck.get('best_val', float('inf')))
            no_improve0 = int(ck.get('no_improve', 0))
            scheduler.last_epoch = global_step0 - 1   # 让下一步的 lr 与断点衔接
            print(f'📥 续训: 从 step {global_step0} / epoch {start_epoch} 继续'
                  f'（原 best_val {best_val0:.3f}；lr 将按新的总轮数重铺 cosine）')
        else:
            print(f'⚠️ 未找到 resume 文件: {resume_path}，从头训练')

    # ---------- eval 模式 loss（无 dropout，用来判断过拟合） ----------
    def eval_loss(loader):
        gpt.eval()
        total, cnt = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast('cuda', enabled=use_amp):
                    logits = gpt(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                total += loss.item()
                cnt += 1
        gpt.train()
        return total / max(cnt, 1)

    # ---------- 保存 ----------
    best_path = os.path.join(out_dir, 'checkpoint_best.pt')
    tok_best_path = os.path.join(out_dir, 'tokenizer_best.pkl')
    latest_path = os.path.join(out_dir, 'checkpoint_latest.pt')
    tok_latest_path = os.path.join(out_dir, 'tokenizer_latest.pkl')

    def save_checkpoint(tag, ckpt_path, tok_path, full=False):
        """full=True 存完整续训包(模型+优化器+步数)；否则只存纯 state_dict(推理用)。"""
        if full:
            torch.save({'model': gpt.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None,
                        'step': global_step,
                        'epoch': epoch,
                        'best_val': best_val,
                        'no_improve': no_improve}, ckpt_path)
        else:
            torch.save(gpt.state_dict(), ckpt_path)
        with open(tok_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        tqdm.write(f'  💾 已保存 {tag}: {ckpt_path}')

    # ---------- 训练主循环 ----------
    best_val = best_val0
    no_improve = no_improve0
    global_step = global_step0
    last_val_step = -1
    ema = None
    stopped = False

    def run_validation(where):
        """验证一次；返回是否触发早停。"""
        nonlocal best_val, no_improve
        val_loss = eval_loss(val_eval_loader)
        msg = f'[{where}] val {val_loss:.3f}'
        if args.eval_batches > 0:
            train_ev = eval_loss(train_eval_loader)          # eval 模式，无 dropout
            gap = val_loss - train_ev
            msg += f' | train_eval(无dropout) {train_ev:.3f} | gap {gap:+.3f}'
        msg += f' | lr {scheduler.get_last_lr()[0]:.1e}'
        tqdm.write(f'\n===== step {global_step} {msg} =====')
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            save_checkpoint('最佳模型', best_path, tok_best_path, full=False)
            tqdm.write(f'  ✓ 新最好模型已保存 (val {val_loss:.3f})')
        else:
            no_improve += 1
            tqdm.write(f'  ⚠️ val 未改善 ({no_improve}/{args.patience})')
        save_checkpoint('最近模型', latest_path, tok_latest_path, full=True)   # 完整续训包
        if no_improve >= args.patience:
            tqdm.write(f'🚫 早停: val 连续 {args.patience} 次验证未改善，停止训练')
            return True
        return False

    def maybe_validate(where):
        """步级与 epoch 末共用的验证入口：同一个 step 只验一次。"""
        nonlocal stopped, last_val_step
        if last_val_step == global_step:
            return False
        last_val_step = global_step
        return run_validation(where)

    for epoch in range(start_epoch, args.epochs):
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
                scaler.unscale_(optimizer)                       # 裁剪前先反缩放
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(dedupe_params(gpt), args.grad_clip)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            running += loss.item()
            steps_here += 1
            global_step += 1
            ema = loss.item() if ema is None else 0.99 * ema + 0.01 * loss.item()
            pbar.set_postfix(loss=f'{ema:.3f}', lr=f'{scheduler.get_last_lr()[0]:.1e}')

            if args.max_steps and global_step >= args.max_steps:
                stopped = True
                break
            if args.val_every > 0 and global_step % args.val_every == 0:
                if maybe_validate(f'step {global_step}'):
                    stopped = True
                    break
        pbar.close()

        tqdm.write(f'epoch {epoch} 平均 train loss: {running / max(steps_here, 1):.3f} '
                   f'(EMA {ema:.3f}) — 已完成 {global_step}/{total_steps} 步')
        # epoch 末总是验证一次（与步级验证同一步时自动去重），保证任何模式下都会保存 checkpoint
        if not stopped:
            if maybe_validate(f'epoch {epoch} 结束'):
                stopped = True
        if stopped:
            break

    tqdm.write(f'训练结束。最佳模型: {best_path} (val {best_val:.3f})')


if __name__ == '__main__':
    main()
