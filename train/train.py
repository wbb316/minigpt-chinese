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

from data.tokenizer import BPETokenizer, EOS_ID  # noqa: E402
from data.dataset import TextDataset, PackedDataset   # noqa: E402
from data.token_cache import encode_to_cache, ShardMemmap  # noqa: E402
from model.gpt import GPT                     # noqa: E402


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def dedupe_params(module):
    """tie_embeddings 时 head.weight 与 token_emb.weight 是同一对象，去重避免被优化两次。"""
    return list({id(p): p for p in module.parameters()}.values())


def _opt_step_count(optimizer):
    """optimizer 累计完成的真实参数更新次数（版本无关，从 state['step'] 读）。

    AMP GradScaler 检测到 overflow 时会跳过 optimizer.step（state 不变），
    用 step 前后计数比较即可判断本步是否真正更新了参数。
    """
    for s in optimizer.state.values():
        st = s.get('step')
        if st is not None:
            return float(st)
    return 0.0


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
    parser.add_argument('--log-dir', default='log',
                        help='验证历史 CSV 输出目录（默认 log/，文件名带语料名自动区分）')
    parser.add_argument('--vocab-size', type=int, default=6144,
                        help='词表总大小 = 256字节 + merges；6.4M 参数推荐 6144（配 tie 嵌入防嵌入层过大）')
    parser.add_argument('--tokens-sample', type=int, default=4_000_000,
                        help='训练分词器用采样字符数（均匀撒 100 段全语料；2M 对 6144 已够，4M 为 8192 或求稳留余量）')
    parser.add_argument('--bpe-trainer', default='legacy', choices=['legacy', 'fast'],
                        help='BPE 训练后端：legacy=v2 原实现；fast=lazy heap 提速版'
                             '（与 legacy 输出完全等价，vocab>=3256 时 2.5-6.5x 加速；'
                             '等价性由 test/test_tokenizer_train_equivalence.py 保证）')
    parser.add_argument('--cache-format', default='shards',
                        choices=['shards', 'legacy'],
                        help='token 缓存格式：shards=uint16 分片 memmap（默认，省内存/磁盘）；'
                             'legacy=旧版单文件 int64 npy（兼容老缓存）')
    # 数据 / 模型
    parser.add_argument('--sample-mode', default='slide', choices=['pack', 'slide'],
                        help='slide=滑动窗口 stride=1(默认)；pack=不重叠打包(每轮步数少 ~128 倍)')
    parser.add_argument('--n-layer', type=int, default=6)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-embd', type=int, default=256)
    parser.add_argument('--block-size', type=int, default=256,
                        help='上下文长度；6.4M 模型 128~256 合适，256 提升连贯性（手动注意力 O(T²)，勿再加大）')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--position-encoding', default='rope',
                        choices=['sinusoidal', 'rope'],
                        help='位置编码（两套互斥，二选一）：rope=旋转位置编码'
                             '（默认——2026-09-06 短训对比胜出，val 好 ~1.0 nats）；'
                             'sinusoidal=绝对正弦（旧模型复现/兼容）')
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
    parser.add_argument('--lr-scheme', default='cosine', choices=['cosine', 'const'],
                        help='LR 调度：cosine=warmup+cosine 衰减（默认，从头训练）；'
                             'const=恒 lr=max_lr×min_lr_ratio（--resume 续跑 Epoch2+ 用：'
                             '从上一轮尾 lr 继续，不重铺 cosine，杜绝 resume 首步 LR 突跳）')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    # 性能专项（TRAINING_PERFORMANCE_AUDIT）参数
    parser.add_argument('--log-every', type=int, default=20,
                        help='每多少成功训练步做一次 loss.item()/EMA/tqdm/step 记录'
                             '（默认 20；减少 GPU→CPU 同步；1 = 旧行为每步记录）')
    parser.add_argument('--precision', default='fp16', choices=['fp16', 'bf16'],
                        help='训练精度：fp16=autocast+GradScaler（默认）；'
                             'bf16=autocast bf16 无 scaler（4090 原生支持）')
    parser.add_argument('--fused-adamw', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='用 fused AdamW（单核 kernel，需 CUDA；不可用时自动回退）')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='torch.compile 模型（首次编译开销大；仅 CUDA；需先 benchmark 再默认）')
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
    parser.add_argument('--seed', type=int, default=None,
                        help='固定随机种子（torch/numpy/random；用于对照实验保证'
                             '数据流一致）。默认 None = 不固定（原行为）')
    args = parser.parse_args()

    if args.seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f'随机种子已固定: {args.seed}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    use_amp = use_cuda                                  # autocast 仅在 CUDA 启用
    amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    use_scaler = use_cuda and args.precision == 'fp16'  # bf16 动态范围大，无需 GradScaler
    print(f'使用设备: {device}  (AMP: {use_amp}, precision: {args.precision}, '
          f'scaler: {use_scaler})')

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
        tokenizer.train(''.join(sample_parts), backend=args.bpe_trainer)
        with open(tok_cache_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f'tokenizer词表大小：{len(tokenizer.vocab)}（已缓存 → {tok_cache_path}）')

    # ---------- 编码（带缓存，缓存名含语料长度，语料变了自动失效） ----------
    vocab_size = len(tokenizer.vocab)

    def get_tokens(text, name):
        """语料 → token 视图。

        - shards 格式（默认）：uint16 分片 + memmap，不整载内存
        - legacy 格式：旧版单文件 int64 npy（兼容老缓存）
        """
        if args.cache_format == 'shards':
            cache_path = encode_to_cache(
                text, tokenizer, cache_dir, name,
                n_procs=args.encode_workers)
            mm = ShardMemmap(cache_path)
            print(f'加载 {name}: {len(mm):,} token (uint16 memmap, '
                  f'{mm.index["n_shards"]} 分片)')
            return mm

        # ---- legacy: 单文件 int64 npy（旧行为）----
        cache_path = os.path.join(
            cache_dir,
            f'tokens_{name}_v{vocab_size}_s{args.tokens_sample}_len{len(text)}_tokv2.npy')
        if os.path.exists(cache_path):
            tokens = np.load(cache_path)
            print(f'从缓存加载 {name}: {len(tokens)} token (legacy int64)')
            return tokens
        tokens = encode_text(text, tokenizer, args.encode_workers)
        np.save(cache_path, tokens)
        print(f'编码并缓存 {name}: {len(text)} 字符 → {len(tokens)} token')
        return tokens

    train_tokens = get_tokens(train_text, 'train')
    val_tokens = get_tokens(val_text, 'val')
    print(f'训练集 {len(train_tokens):,} token, '
          f'验证集 {len(val_tokens):,} token')

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
              tie_embeddings=args.tie_embeddings,
              position_encoding=args.position_encoding)
    gpt = gpt.to(device)
    n_params = gpt.get_num_params()
    print(f'位置编码: {args.position_encoding}')
    print(f'GPT 参数量: {n_params:,}'
          + ('（含 tie_embeddings，已去重）' if args.tie_embeddings else ''))

    if args.compile:
        if use_cuda:
            print('🔧 torch.compile 编译中（首次调用开销不计入吞吐 benchmark）...')
            gpt = torch.compile(gpt)
        else:
            print('⚠️ --compile 仅 CUDA 可用，已忽略（当前 CPU）')

    # ---------- optimizer（fused AdamW 可选，自动回退） ----------
    opt_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)
    if args.fused_adamw:
        try:
            optimizer = torch.optim.AdamW(dedupe_params(gpt), fused=True,
                                          **opt_kwargs)
            print('optimizer: AdamW(fused=True)')
        except (TypeError, RuntimeError) as e:
            print(f'⚠️ fused AdamW 不可用（{e}）→ 回退普通 AdamW')
            optimizer = torch.optim.AdamW(dedupe_params(gpt), **opt_kwargs)
    else:
        optimizer = torch.optim.AdamW(dedupe_params(gpt), **opt_kwargs)

    # ---------- LR 调度（cosine / const 恒温续训） ----------
    # cosine: warmup + cosine 衰减到 max_lr×min_ratio（默认路径，从头训练）
    # const : 恒 lr = max_lr×min_ratio —— 给 --resume 续跑 Epoch2+ 用：
    #         从上一轮尾 lr 继续，不按新 total_steps 重铺 cosine（重铺会使
    #         resume 首步 LR 从 5e-5 突跳回 ~4.35e-4，见 docs/RESUME_AUDIT.md §2）
    warmup = args.warmup_steps
    min_ratio = max(0.0, args.min_lr_ratio)          # 底部学习率比例（默认 0.1）

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup                       # 线性升到 1.0
        t = (step - warmup) / max(1, total_steps - warmup)   # 0 → 1
        t = min(t, 1.0)
        # cosine 从 1.0 衰减到 min_ratio（默认 0.1，不归零 → 尾部保留学习能力）
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * t))

    if args.lr_scheme == 'const':
        # 恒 lr = max_lr × min_ratio；lambda 与 step 无关 → step() 幂等、无跳变
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min_ratio)
        print(f'LR 方案: const（恒 {args.lr * min_ratio:.2e}，'
              f'continuation 用，不重铺 cosine）')
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f'LR 方案: cosine（warmup {warmup} 步 → '
              f'衰减到 {args.lr * min_ratio:.2e}）')
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None

    # ---------- 断点续训（加轮数: --resume <latest> --epochs 更大的总数） ----------
    # epoch 语义（见 docs/RESUME_AUDIT.md §1）：checkpoint 存 next_epoch =
    #   "该 checkpoint 之后下一个要跑的 epoch"（epoch 末保存 = epoch+1；步级验证保存 = epoch）。
    # resume: start_epoch = next_epoch → for range(start_epoch, epochs) 不会重复已完成 epoch。
    resume_path = resolve(args.resume) if args.resume else None
    start_epoch = 0
    global_step0 = 0
    best_val0 = float('inf')
    no_improve0 = 0
    tok_total0 = None          # checkpoint 里的精确 tokens_seen（旧包无 → None）
    if resume_path:
        if os.path.exists(resume_path):
            ck = torch.load(resume_path, map_location=device, weights_only=True)
            try:
                gpt.load_state_dict(ck['model'])
            except RuntimeError as e:
                if 'rope' in str(e) or 'Missing key' in str(e):
                    raise RuntimeError(
                        'checkpoint 结构与当前 position_encoding 不匹配——'
                        '该 checkpoint 大概率是 sinusoidal 训练，请加 '
                        '--position-encoding sinusoidal 复现/续训；'
                        'rope 模式需用 rope 训练的新 checkpoint。') from e
                raise
            optimizer.load_state_dict(ck['optimizer'])
            if use_amp and ck.get('scaler') is not None:
                scaler.load_state_dict(ck['scaler'])
            global_step0 = int(ck.get('step', 0))
            start_epoch = int(ck.get('next_epoch', ck.get('epoch', 0)))
            best_val0 = float(ck.get('best_val', float('inf')))
            no_improve0 = int(ck.get('no_improve', 0))
            tok_total0 = ck.get('tokens_seen')
            if args.lr_scheme == 'cosine':
                # 仅 cosine 需要重锚（const 的 lambda 与 step 无关，设了也无副作用）
                scheduler.last_epoch = global_step0 - 1
            print(f'📥 续训: 从 step {global_step0} / next_epoch {start_epoch} 继续'
                  f'（原 best_val {best_val0:.3f}）')
            if args.lr_scheme == 'cosine':
                print(f'  ⚠️ 注意: --lr-scheme cosine 会按新总步数重铺 LR，'
                      f'resume 首步 lr 可能跳变——续跑请用 --lr-scheme const')
        else:
            print(f'⚠️ 未找到 resume 文件: {resume_path}，从头训练')

    # ---------- 启动配置快照（终端打印 + 落盘留档，方便日后对比实验） ----------
    dataset_tag = os.path.splitext(os.path.basename(args.train_txt))[0]
    config_lines = [
        '========== RUN CONFIG ==========',
        f'resume_from_step:   {global_step0}',
        f'dataset_tag:        {dataset_tag}',
        'model:',
        f'  n_layer:          {args.n_layer}',
        f'  position_enc:     {args.position_encoding}',
        f'  n_head:           {args.n_head}',
        f'  n_embd:           {args.n_embd}',
        f'  block_size:       {args.block_size}',
        f'  vocab_size:       {vocab_size}',
        f'  tie_embeddings:   {args.tie_embeddings}',
        f'  dropout:          {args.dropout}',
        f'  parameters:       {n_params:,}',
        'data:',
        f'  train_tokens:     {len(train_tokens):,}',
        f'  val_tokens:       {len(val_tokens):,}',
        f'  sample_mode:      {args.sample_mode}',
        f'  packing:          {"pack(不重叠)" if args.sample_mode=="pack" else "slide(stride=1)"}',
        f'  eos_token:        <eos> (id={EOS_ID})  [纯续写训练未用]',
        'training:',
        f'  batch_size:       {args.batch_size}',
        f'  tokens_per_step:  {args.batch_size * args.block_size:,}',
        f'  lr:               {args.lr}',
        f'  lr_scheme:        {args.lr_scheme}',
        f'  precision:        {args.precision}',
        f'  log_every:        {args.log_every}',
        f'  fused_adamw:      {args.fused_adamw}',
        f'  compile:          {args.compile}',
        f'  warmup_steps:     {args.warmup_steps}',
        f'  decay_steps:      {max(0, total_steps - args.warmup_steps)}',
        f'  total_steps:      {total_steps}',
        f'  min_lr:           {args.lr * min_ratio:.2e} (=max_lr*{min_ratio})',
        f'  optimizer:        AdamW',
        f'  weight_decay:     {args.weight_decay}',
        f'  grad_clip:        {args.grad_clip}',
        f'  val_every:        {args.val_every}',
        f'  patience:         {args.patience}',
        '==================================',
    ]
    for cl in config_lines:
        print(cl)
    import time as _time
    _cfg_path = os.path.join(resolve(args.log_dir), 'run_config.txt')
    os.makedirs(os.path.dirname(_cfg_path), exist_ok=True)
    with open(_cfg_path, 'a', encoding='utf-8') as f:
        f.write(f'\n[{_time.strftime("%Y-%m-%d %H:%M:%S")}] run start\n')
        f.write('\n'.join(config_lines) + '\n')
    print(f'📄 配置已追加到: {_cfg_path}')

    # ---------- eval 模式 loss（无 dropout，用来判断过拟合） ----------
    def eval_loss(loader):
        gpt.eval()
        total, cnt = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast('cuda', dtype=amp_dtype, enabled=use_cuda):
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

    def save_checkpoint(tag, ckpt_path, tok_path, full=False, next_epoch=None):
        """full=True 存完整续训包(模型+优化器+步数)；否则只存纯 state_dict(推理用)。

        next_epoch: 该 checkpoint 之后下一个要跑的 epoch——
                    epoch 末验证传 epoch+1（本 epoch 已完成）；
                    步级验证传当前 epoch（本 epoch 未完成，resume 后重跑）。
        """
        if full:
            ne = (epoch + 1) if next_epoch is None else next_epoch
            torch.save({'model': gpt.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None,
                        'step': global_step,
                        'epoch': epoch,
                        'next_epoch': ne,
                        'tokens_seen': tok_total,
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
    amp_skip_steps = 0                                  # AMP 跳过（未更新）的步数
    # tokens_seen 精确累计：resume 优先用 checkpoint 存的精确值；旧包无则按步数×batch 近似
    tok_total = (tok_total0 if tok_total0 is not None
                 else global_step0 * args.batch_size * args.block_size)
    last_val_step = -1
    ema = None
    stopped = False

    # ---------- 验证历史 CSV（每次验证追加一行，resume 续训也接着记） ----------
    corpus_tag = os.path.splitext(os.path.basename(args.train_txt))[0]
    log_dir = resolve(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    val_csv_path = os.path.join(log_dir, f"val_history_{corpus_tag}.csv")
    csv_header = ('step,where,val,train_eval,gap,lr,is_best,no_improve,'
                  'best_val,wall_time')
    if not os.path.exists(val_csv_path) or os.path.getsize(val_csv_path) == 0:
        with open(val_csv_path, 'w', encoding='utf-8') as f:
            f.write(csv_header + '\n')
        print(f'📈 验证历史将记录到: {val_csv_path}')

    # ---------- step 级历史 CSV（每步一行: tokens_seen 用于跨 run 公平比较） ----------
    # 列: step,epoch,tokens_seen,train_loss(EMA),val_loss(验证步才有),lr,time
    # tokens_seen = 全局累计看到的 token 数(含 resume 前的), 与 batch/epoch 无关,
    #   以后比较 400M vs 1B tokens / 不同 batch 都看这一列。
    step_csv_path = os.path.join(log_dir, f"step_history_{corpus_tag}.csv")
    step_header = 'step,epoch,tokens_seen,train_loss,val_loss,lr,time'
    step_csv_new = (not os.path.exists(step_csv_path)
                    or os.path.getsize(step_csv_path) == 0)
    step_fh = open(step_csv_path, 'a', encoding='utf-8')
    if step_csv_new:
        step_fh.write(step_header + '\n')
    _buf: list[str] = []          # 行缓冲, 攒批落盘
    _last_flush_step = 0

    def flush_step_rows():
        nonlocal _buf, _last_flush_step
        if _buf:
            step_fh.write('\n'.join(_buf) + '\n')
            _buf = []
        _last_flush_step = global_step

    def append_step_row(train_loss_ema, val_loss=None):
        """每步调一次; val_loss 非 None(验证步) 时带上。
        tok_total 由主循环按 x.numel() 精确累计（尾批不按满 batch 计，见 §D）。"""
        tok = tok_total
        ep = tok / max(1, len(train_tokens))                    # 进度(按 token 计)
        val_s = f'{val_loss:.4f}' if val_loss is not None else ''
        row = (f'{global_step},{ep:.3f},{tok},{train_loss_ema:.4f},{val_s},'
               f'{scheduler.get_last_lr()[0]:.2e},'
               f'{_time.strftime("%Y-%m-%d %H:%M:%S")}')
        _buf.append(row)
        # 每 25 步 flush 一次
        if global_step - _last_flush_step >= 25:
            flush_step_rows()

    def log_validation_row(where, val_loss, train_ev, gap, lr, is_best):
        """把一次验证结果追加到 val CSV（细粒度每验证行）。"""
        row = (f'{global_step},{where},{val_loss:.4f},'
               f'{train_ev if train_ev is not None else ""},'
               f'{gap if gap is not None else ""},{lr:.2e},'
               f'{int(is_best)},{no_improve},{best_val:.4f},'
               f'{_time.strftime("%Y-%m-%d %H:%M:%S")}')
        with open(val_csv_path, 'a', encoding='utf-8') as f:
            f.write(row + '\n')
        # step 历史里也补一行带 val 的（与无 val 的 step 行并存, 画图取非空列即可）
        append_step_row(ema if ema is not None else float('nan'),
                        val_loss=val_loss)
        flush_step_rows()

    def run_validation(where):
        """验证一次；返回是否触发早停。"""
        nonlocal best_val, no_improve
        # epoch 末验证 → 本 epoch 已完成，next_epoch = epoch+1；步级验证 → 本 epoch 未完成
        is_epoch_end = where.startswith('epoch ') and where.endswith(' 结束')
        next_epoch_val = (epoch + 1) if is_epoch_end else epoch
        val_loss = eval_loss(val_eval_loader)
        msg = f'[{where}] val {val_loss:.3f}'
        train_ev = gap = None
        if args.eval_batches > 0:
            train_ev = eval_loss(train_eval_loader)          # eval 模式，无 dropout
            gap = val_loss - train_ev
            msg += f' | train_eval(无dropout) {train_ev:.3f} | gap {gap:+.3f}'
        lr_now = scheduler.get_last_lr()[0]
        msg += f' | lr {lr_now:.1e}'
        tqdm.write(f'\n===== step {global_step} {msg} =====')
        is_best = False
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            is_best = True
            save_checkpoint('最佳模型', best_path, tok_best_path, full=False)
            tqdm.write(f'  ✓ 新最好模型已保存 (val {val_loss:.3f})')
        else:
            no_improve += 1
            tqdm.write(f'  ⚠️ val 未改善 ({no_improve}/{args.patience})')
        save_checkpoint('最近模型', latest_path, tok_latest_path, full=True,
                        next_epoch=next_epoch_val)   # 完整续训包（含 next_epoch/tokens_seen）
        log_validation_row(where, val_loss, train_ev, gap, lr_now, is_best)
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
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # zero_grad 放 forward 前 + set_to_none（省 grad buffer 置零 kernel）
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=use_cuda):
                logits = gpt(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)                       # 裁剪前先反缩放
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(dedupe_params(gpt), args.grad_clip)

            # ---- 更新门控（docs/RESUME_AUDIT.md §3）：仅当 optimizer 确实完成
            #      参数更新才推进 scheduler/global_step/tokens_seen/step 行 ----
            # AMP GradScaler 检测到 overflow 会跳过 optimizer.step →
            # 该步不算训练步：不步进 scheduler（否则触发 "scheduler.step before
            # optimizer.step" warning 且 LR 计数错位）、不累计进度。
            if scaler is not None:
                cnt0 = _opt_step_count(optimizer)
                scaler.step(optimizer)
                scaler.update()
                updated = _opt_step_count(optimizer) > cnt0
            else:
                optimizer.step()
                updated = True

            if updated:
                scheduler.step()
                global_step += 1
                tok_total += x.numel()       # 精确 token（尾批按实际 numel，§D）
                # ---- 采样记录（P1，--log-every）：每 N 步一次 loss.item()
                #      GPU→CPU 同步 + EMA/tqdm/step 行；其余步零同步 ----
                if global_step % args.log_every == 0:
                    li = float(loss.item())
                    ema = li if ema is None else 0.99 * ema + 0.01 * li
                    running += li
                    steps_here += 1
                    pbar.set_postfix(loss=f'{ema:.3f}',
                                     lr=f'{scheduler.get_last_lr()[0]:.1e}')
                    append_step_row(ema)     # ★ step 行（采样步；tokens_seen/lr 真实）
            else:
                amp_skip_steps += 1          # AMP overflow 跳过（不消耗进度）

            if args.max_steps and global_step >= args.max_steps:
                stopped = True
                break
            if args.val_every > 0 and global_step % args.val_every == 0:
                if maybe_validate(f'step {global_step}'):
                    stopped = True
                    break
        pbar.close()

        flush_step_rows()                              # epoch 末把缓冲落盘
        tqdm.write(f'epoch {epoch} 平均 train loss（每 {args.log_every} 步采样）: '
                   f'{running / max(steps_here, 1):.3f} '
                   f'(EMA {ema if ema is not None else float("nan"):.3f}; '
                   f'AMP skipped {amp_skip_steps} 步) '
                   f'— 已完成 {global_step}/{total_steps} 步, '
                   f'tokens_seen={tok_total:,.0f}')
        # epoch 末总是验证一次（与步级验证同一步时自动去重），保证任何模式下都会保存 checkpoint
        if not stopped:
            if maybe_validate(f'epoch {epoch} 结束'):
                stopped = True
        if stopped:
            break

    flush_step_rows()
    step_fh.close()
    tqdm.write(f'训练结束。最佳模型: {best_path} (val {best_val:.3f})')
    tqdm.write(f'📈 验证历史: {val_csv_path}')
    tqdm.write(f'📈 step 历史: {step_csv_path}')
    tqdm.write(f'📊 tokens_seen 合计: {tok_total:,.0f}'
               + (f'（AMP skipped {amp_skip_steps} 步）' if amp_skip_steps else ''))


if __name__ == '__main__':
    main()
