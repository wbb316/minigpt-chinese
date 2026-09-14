# -*- coding: utf-8 -*-
"""TRAINING_PERFORMANCE_AUDIT 消融 benchmark 执行器（云端 4090 运行）。

用法（云端 /root，代码在 /root/train + /root/data）：
  python benchmark/perf_bench.py --variant B1                  # 单组（默认重复 3 次）
  python benchmark/perf_bench.py --all                         # 50m 档全部 B 系列
  python benchmark/perf_bench.py --all --model-profile 100m    # 100m 档 C 系列
  python benchmark/perf_bench.py --variant C0_100m_baseline    # C 系列自带 100m 档绑定

每次重复的流程：
  1. out/log/cache 指向 benchmark/perf_out/<variant>（tokenizer pkl 自动从 profile 的 --tok-src 拷贝）
  2. 后台 nvidia-smi 1s 采样（**带 timestamp 列**）→ benchmark/gpu_metrics_<variant>_r<n>.csv
  3. 前台跑 train/train.py（--max-steps N --val-every 0 --log-every <变体配置>）
  4. 聚合：step_history 稳定段（最后 10 个采样行）→ step/s、tokens/s；
     GPU 指标**只统计稳定段窗口内的样本**（E0-2：否则 python 启动/加载/compile 编译期会拉低 avg）
  5. 每次重复写一行 performance_results.csv（末列 `repeat`，**1-based**）；
     同一变体 N 次重复再汇总 median/min/max 一行 → benchmark/performance_summary.csv（E0-4）

E0-4 判定口径（不要靠感觉）：<1% 记为噪声；1%~2% 需更多重复；>2% 且 3 次同向才算可信收益。

配置档（50m/100m）见 MODEL_PROFILES，变体矩阵见 VARIANTS。
"""
import argparse
import csv
import datetime
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, 'benchmark')
OUT_ROOT = os.path.join(BENCH, 'perf_out')
RESULTS_CSV = os.path.join(BENCH, 'performance_results.csv')
SUMMARY_CSV = os.path.join(BENCH, 'performance_summary.csv')

# `repeat` 追加在**末尾**（E0-4）。历史文件（28 列）不重写 → 见 write_row 的说明。
CSV_FIELDS = ('variant,commit,model_params,batch_size,block_size,precision,'
              'optimizer,fused,compile,num_workers,log_every,'
              'warmup_steps_measured,measured_steps,elapsed_seconds,'
              'step_per_sec,tokens_per_sec,peak_vram_mb,gpu_util_avg,'
              'gpu_util_p50,gpu_util_p90,gpu_power_avg,cpu_util_avg,'
              'ram_peak_mb,loss_start,loss_end,nan_count,amp_skipped,notes,'
              'repeat')
SUMMARY_FIELDS = ('variant,model_profile,repeats,'
                  'tokens_per_sec_median,tokens_per_sec_min,tokens_per_sec_max,'
                  'step_per_sec_median,step_per_sec_min,step_per_sec_max,'
                  'gpu_util_avg_median,gpu_util_avg_min,gpu_util_avg_max')

# ---------- 配置档（E0-5）：云端真实路径；可被 --train-txt/--val-txt/--cache-dir/--tok-src 覆盖 ----------
# 50m 档**必须与历史 benchmark 逐字一致**（回归保护项）：12L/9H/576d / vocab6144 / shard0(v2)。
MODEL_PROFILES = {
    '50m': dict(
        model_args='--n-layer 12 --n-head 9 --n-embd 576 --block-size 512 '
                   '--vocab-size 6144 --tie-embeddings --sample-mode pack '
                   '--bpe-trainer fast --cache-format shards',
        block_size=512,                     # 必须与 model_args 的 --block-size 一致
        # 语料在**系统盘** `/root/data/`（抗克隆），缓存才放数据盘 —— 理由见 100m 档下的说明。
        # 2026-09-12 修正：原先写成 /root/autodl-tmp/data/，但语料实际不在那里，
        #   导致每次跑都得用 --train-txt/--val-txt 手工覆盖（本次 50M workers 重测即如此）。
        train_txt='/root/data/train_webnovel_v2.txt',
        val_txt='/root/data/val_webnovel_v2.txt',
        cache_dir='/root/autodl-tmp/data',
        tok_src='/root/result_50m/tokenizer_v6144_s4000000.pkl',
    ),
    # 100M 档：16L/11H/704d / vocab8192 / shard12——对齐已归档的生产 run
    # （log/100M参数_v3_2Btokens/run_config.txt：参数 101,457,280 / log_every 20 / compile on）。
    '100m': dict(
        model_args='--n-layer 16 --n-head 11 --n-embd 704 --block-size 512 '
                   '--vocab-size 8192 --tie-embeddings --sample-mode pack '
                   '--bpe-trainer fast --cache-format shards',
        block_size=512,
        # 语料在**系统盘** `/root/data/`，**不是**数据盘：AutoDL 克隆只复制系统盘 `/`，
        #   不带 `/root/autodl-tmp`（本项目已因此丢过一次语料）。
        #   缓存体积大且可重建，才放数据盘。
        # 2026-09-12 修正：原先写成 /root/autodl-tmp/data/ 是按 50m 档"对称推断"的，与实际不符。
        train_txt='/root/data/train_webnovel_shard12.txt',
        val_txt='/root/data/val_webnovel_shard12.txt',
        cache_dir='/root/autodl-tmp/data',
        # 已实测确认存在（2026-09-12；训练日志亦打印
        #   "从缓存加载分词器: 词表 8192 ← /root/result_100m/..."），175,995 字节。
        tok_src='/root/result_100m/tokenizer_v8192_s4000000.pkl',
    ),
}
DEFAULT_PROFILE = '50m'

TRAIN_PY = os.path.join(ROOT, 'train', 'train.py')  # 云端 /root/train/train.py

# 注意：这里**不要再放 `--num-workers`**——argparse 同一参数取最后一个，
# 会静默覆盖变体自己的 workers（历史 B3_dl_w0/w2/w4/w12 因此是同一次配置跑了 5 遍）。
BASE_ARGS = ('--lr 8e-4 --min-lr-ratio 0.0625 --warmup-steps 1000 '
             '--weight-decay 0.05 --epochs 1 --patience 2 --eval-batches 0 '
             '--val-every 0 --encode-workers 16')

STEPS_TOTAL = 900          # 每组总步（含 burn）；稳定段 = 最后 ~500 步
LOG_EVERY = 50             # step CSV 采样间隔兜底值（变体一般显式给定）

# 变体的 profile 绑定：没有显式 --model-profile 时，C 系列自动用 100m 档（B 系列用 50m）。
VARIANTS = {
    # B0 基线：当前最新代码 + train.py 真实默认（log_every=20），P0 修复已在代码里
    'B0_current_baseline': dict(batch=64, precision='fp16', fused=False, compile=False,
                                workers=8, log_every=20,
                                notes='P0 fixes included, log_every=20(train.py 默认)'),
    # B1 与 B0 只差日志频率（log_every=50）——真正的日志频率 A/B
    'B1_logging_reduced': dict(batch=64, precision='fp16', fused=False,
                               compile=False, workers=8, log_every=50,
                               notes='log_every=50 采样（每步 item 消除）'),
    'B3_dl_w0': dict(batch=64, precision='fp16', fused=False, compile=False,
                     workers=0, log_every=50, notes='dataloader workers=0'),
    'B3_dl_w2': dict(batch=64, precision='fp16', fused=False, compile=False,
                     workers=2, log_every=50, notes='dataloader workers=2'),
    'B3_dl_w4': dict(batch=64, precision='fp16', fused=False, compile=False,
                     workers=4, log_every=50, notes='dataloader workers=4'),
    'B3_dl_w8': dict(batch=64, precision='fp16', fused=False, compile=False,
                     workers=8, log_every=50, notes='dataloader workers=8'),
    'B3_dl_w12': dict(batch=64, precision='fp16', fused=False, compile=False,
                      workers=12, log_every=50, notes='dataloader workers=12'),
    'B5_fused_adamw': dict(batch=64, precision='fp16', fused=True,
                           compile=False, workers=8, log_every=50,
                           notes='best workers + fused AdamW'),
    'B6_compile': dict(batch=64, precision='fp16', fused=False, compile=True,
                       workers=8, log_every=50, notes='torch.compile（计时排除编译期）'),
    'B7_bf16': dict(batch=64, precision='bf16', fused=False, compile=False,
                    workers=8, log_every=50, notes='bf16 no scaler'),
    'B8_batch80': dict(batch=80, precision='fp16', fused=False, compile=False,
                       workers=8, log_every=50, notes='batch=80'),
    'B8_batch96': dict(batch=96, precision='fp16', fused=False, compile=False,
                       workers=8, log_every=50, notes='batch=96'),
    'B9_compile_batch80': dict(batch=80, precision='fp16', fused=False,
                               compile=True, workers=8, log_every=50,
                               notes='compile + batch80 组合（最优栈候选）'),
    'B9_compile_batch64_rep': dict(batch=64, precision='fp16', fused=False,
                                   compile=True, workers=8, log_every=50,
                                   notes='compile+batch64 复现（验证 B6 稳定）'),
    # ---------- C 系列：100M 档（E0-5）。C2+ 等 E1 profiler 结果出来再定 ----------
    'C0_100m_baseline': dict(profile='100m', batch=64, precision='fp16', fused=False,
                             compile=True, workers=8, log_every=20,
                             notes='100M 生产配置（16L/11H/704d/vocab8192/shard12），'
                                   'batch64/block512/fp16/compile/workers8；'
                                   'lr/warmup 仍用 BASE_ARGS（对吞吐无影响）'),
    'C1_100m_nocompile': dict(profile='100m', batch=64, precision='fp16', fused=False,
                              compile=False, workers=8, log_every=20,
                              notes='C0 只关 compile（量化 100M 上 compile 的收益）'),
}

# --all 的执行顺序：按 profile 分开（B0-B9 是 50m 保护项，C 系列是 100m）
ALL_ORDER = {
    '50m': ['B0_current_baseline', 'B1_logging_reduced',
            'B3_dl_w0', 'B3_dl_w2', 'B3_dl_w4', 'B3_dl_w8', 'B3_dl_w12',
            'B5_fused_adamw', 'B6_compile', 'B7_bf16', 'B8_batch80',
            'B8_batch96', 'B9_compile_batch64_rep'],
    '100m': ['C0_100m_baseline', 'C1_100m_nocompile'],
}

# ---------- CLI 覆盖的云端路径（main 里填；None = 用 profile 的值） ----------
PATH_OVERRIDES = {}


# ---------------------------------------------------------------- 配置/profile
def profile_env(profile_name):
    """profile 的路径配置 + CLI 显式覆盖（--train-txt 等优先级最高）。"""
    env = dict(MODEL_PROFILES[profile_name])
    for key in ('train_txt', 'val_txt', 'cache_dir', 'tok_src'):
        if PATH_OVERRIDES.get(key):
            env[key] = PATH_OVERRIDES[key]
    return env


def resolve_profile(name, cfg, profile_arg=None):
    """变体绑定（C* → 100m）优先；CLI 显式 --model-profile 覆盖一切并告警。"""
    bound = cfg.get('profile') or DEFAULT_PROFILE
    if profile_arg and profile_arg != bound:
        print(f'⚠️ [{name}] 变体绑定 profile={bound}，被 --model-profile {profile_arg} 覆盖',
              file=sys.stderr)
        return profile_arg
    return bound


def step_csv_path(log_dir, train_txt):
    """train.py 写的 step CSV 路径：log_dir/step_history_<train_txt basename>.csv
    （train.py 用 corpus_tag = splitext(basename(train_txt))[0]，见 train/train.py:641/659）。
    100M 档是 train_webnovel_shard12.txt → step_history_train_webnovel_shard12.csv。
    """
    tag = os.path.splitext(os.path.basename(train_txt))[0]
    return os.path.join(log_dir, f'step_history_{tag}.csv')


def duplicate_flags(cmd):
    """命令里出现两次以上的 --flag（argparse 取最后一个 → 静默覆盖，容易造出"空转变体"）。"""
    flags = [a for a in cmd if a.startswith('--')]
    return sorted({f for f in flags if flags.count(f) > 1})


def build_train_cmd(cfg, profile_name, steps_total, out_dir, log_dir, env=None):
    """拼 train.py 命令（纯函数，便于测试；E0-5 的 profile 参数全部来自这里）。"""
    prof = MODEL_PROFILES[profile_name]
    env = env or profile_env(profile_name)
    log_every = cfg.get('log_every', LOG_EVERY)
    cmd = [sys.executable, '-u', TRAIN_PY,
           '--train-txt', env['train_txt'], '--val-txt', env['val_txt'],
           '--cache-dir', env['cache_dir'], '--out-dir', out_dir, '--log-dir', log_dir,
           '--batch-size', str(cfg['batch']), '--precision', cfg['precision'],
           '--num-workers', str(cfg['workers']), '--log-every', str(log_every),
           '--max-steps', str(steps_total)]
    if cfg['fused']:
        cmd.append('--fused-adamw')
    if cfg['compile']:
        cmd.append('--compile')
    cmd += prof['model_args'].split() + BASE_ARGS.split()
    dups = duplicate_flags(cmd)
    if dups:
        print(f'⚠️ 训练命令里重复的参数 {dups}（argparse 取最后一个！）', file=sys.stderr)
    return cmd


def git_commit():
    try:
        r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                           cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() or 'NA'
    except Exception:
        return 'NA'


# ---------------------------------------------------------------- GPU telemetry（E0-2）
# nvidia-smi 的时间戳形如 `2026/09/12 10:30:45.123`（--format=csv 下第一列带引号）；
# 无小数秒时是 `2026/09/12 10:30:45`。用 strptime（带 %f 保住毫秒）→ 本地时区 epoch，
# 与 step CSV 的 time_unix（time.time()）同一坐标系。
GPU_TS_FORMATS = ('%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S')


def parse_gpu_ts(text):
    """nvidia-smi 时间戳 → epoch 秒；解析失败返回 None（坏行跳过而不崩）。"""
    s = (text or '').strip().strip('"').strip()
    if not s:
        return None
    for fmt in GPU_TS_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    try:                                    # 兜底：ISO8601（含带时区）
        return datetime.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def read_gpu_rows(path):
    """nvidia-smi 采样 CSV → [(epoch|None, [util, mem, power, temp, clock]), ...]。

    第一列能解析成时间戳时当作 timestamp 列并用其结果算 epoch；否则整行当旧 5 列格式
    （time=None）。列数不足 / 数值解析失败的行**整行跳过**（不崩）。
    """
    rows = []
    with open(path, newline='') as f:
        for cells in csv.reader(f):
            cells = [c.strip() for c in cells if c is not None]
            if not cells or not any(cells):
                continue
            t = parse_gpu_ts(cells[0])
            vals = cells[1:] if t is not None else cells
            if len(vals) < 5:
                continue
            try:
                nums = [float(v.strip().strip('"') or 'nan') for v in vals[:5]]
            except ValueError:
                continue
            rows.append((t, nums))
    return rows


def parse_gpu_csv(path, t_start=None, t_end=None):
    """nvidia-smi 采样 CSV → (peak_vram, util_avg/p50/p90, power_avg)。

    t_start/t_end（epoch 秒）非 None 时**只统计窗口内样本**：mean/p50/p90 与峰值显存
    都限定窗口。理由（E0-2）：采样器在训练进程**启动前**就开、结束后才停，
    不过滤会把 python 启动、tokenizer/cache 加载、torch.compile 编译期算进均值 →
    avg 被严重低估（历史 B6：avg 65.5 但 p50 98）。
    兼容：旧 CSV 无 timestamp 列（或窗口内无样本）→ 退回全段统计 = 旧行为。
    """
    import numpy as np
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    rows = read_gpu_rows(path)
    if not rows:
        return None
    ts = [r[0] for r in rows]
    nums = [r[1] for r in rows]
    keep = list(range(len(rows)))
    if t_start is not None or t_end is not None:
        stamped = [i for i, t in enumerate(ts) if t is not None]
        if stamped:
            keep = [i for i in stamped
                    if (t_start is None or ts[i] >= t_start)
                    and (t_end is None or ts[i] <= t_end)]
            if not keep:
                print(f'⚠️ GPU 采样窗口 [{t_start}, {t_end}] 内无样本 → 退回全段统计: {path}',
                      file=sys.stderr)
                keep = stamped
        # 无任何时间戳（旧 5 列 CSV）→ keep 保持全段，行为与旧版完全一致
    a = np.array([nums[i] for i in keep], dtype=float)
    return dict(peak_vram=float(np.nanmax(a[:, 1])) if a[:, 1].size else 'NA',
                util_avg=float(np.nanmean(a[:, 0])),
                util_p50=float(np.nanpercentile(a[:, 0], 50)),
                util_p90=float(np.nanpercentile(a[:, 0], 90)),
                power_avg=float(np.nanmean(a[:, 2])),
                n_samples=len(keep))


# ---------------------------------------------------------------- 吞吐（E0-1/E0-2）
def has_time_unix(step_csv):
    """step CSV 是否带 E0-1 的 time_unix 列（历史 8 列 CSV → False）。"""
    if not os.path.exists(step_csv):
        return False
    with open(step_csv, newline='') as f:
        head = next(csv.reader(f), [])
    return 'time_unix' in [h.strip() for h in head]


def read_step_samples(step_csv):
    """step_history CSV → [(step, t_epoch), ...]（坏行跳过）。

    优先 `time_unix`（E0-1，毫秒精度，行生成时刻取样）；旧 8 列 CSV 无该列（或某行
    该列为空）→ 退回秒级 `time` 列（旧行为，量化误差 ≈1.2%）。
    """
    samples = []
    if not os.path.exists(step_csv):
        return samples
    with open(step_csv, newline='') as f:
        rd = csv.DictReader(f)
        has_unix = 'time_unix' in (rd.fieldnames or [])
        for row in rd:
            try:
                step = int(row['step'])
            except (TypeError, ValueError, KeyError):
                continue
            t = None
            if has_unix:
                try:
                    t = float(row['time_unix'])
                except (TypeError, ValueError):
                    t = None
            if t is None:
                try:
                    t = time.mktime(time.strptime(row['time'], '%Y-%m-%d %H:%M:%S'))
                except (TypeError, ValueError, KeyError):
                    t = None
            if t is None:
                continue
            samples.append((step, t))
    return samples


def compute_throughput(step_csv, toks_per_step, seg=10):
    """从 step_history CSV（log_every 采样行）算稳定段吞吐。

    取最后 seg+1 个采样行：Δstep/Δtime → step/s、tokens/s。
    返回 (step_per_sec, tokens_per_sec, measured_steps) 或 ('NA','NA',0)。
    """
    samples = read_step_samples(step_csv)
    if len(samples) < seg + 1:
        return 'NA', 'NA', 0
    ds = samples[-1][0] - samples[-1 - seg][0]
    dt = samples[-1][1] - samples[-1 - seg][1]
    if dt <= 0:
        return 'NA', 'NA', 0
    sps = ds / dt
    return round(sps, 3), round(sps * toks_per_step, 0), ds


def stable_segment_window(step_csv, seg=10):
    """稳定段（最后 seg+1 个采样行）的首尾时刻 → (t_start, t_end)。

    只对带 `time_unix` 的新 CSV 给窗口：旧 8 列 CSV 的秒级 `time` 精度不足，
    返回 (None, None) → parse_gpu_csv 不做窗口过滤，行为与旧版一致（E0-2 向后兼容）。
    """
    if not has_time_unix(step_csv):
        return None, None
    samples = read_step_samples(step_csv)
    if len(samples) < seg + 1:
        return None, None
    t0, t1 = samples[-1 - seg][1], samples[-1][1]
    return (t0, t1) if t1 > t0 else (None, None)


def gpu_csv_path(name, rep):
    """本次重复的 GPU 采样文件：**带 _r<rep>** → 绝不覆盖历史的 gpu_metrics_<variant>.csv。"""
    return os.path.join(BENCH, f'gpu_metrics_{name}_r{rep}.csv')


def parse_model_params(log_text):
    """从训练日志解析 `GPT 参数量: 101,457,280` → int；解析不到 → 'NA'。

    不写死参数量（旧代码写死 51411840，等于 relu-FFN 时代的 50M 值，跑 100M 会记错）。
    解析不到时返回 'NA'：CSV 里一眼可见缺值，胜过静默填一个错误数字；
    调用处还会打一条 stderr 警告，避免整组结果被误读。
    """
    m = re.search(r'GPT 参数量:\s*([\d,]+)', log_text or '')
    if not m:
        return 'NA'
    try:
        return int(m.group(1).replace(',', ''))
    except ValueError:
        return 'NA'


# ---------------------------------------------------------------- 重复测量汇总（E0-4）
def _stats(values):
    """数值列 → (median, min, max)；忽略 'NA'/None/非数（全无效 → (None,)*3）。"""
    vals = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:                          # 排除 nan
            vals.append(f)
    if not vals:
        return None, None, None
    return statistics.median(vals), min(vals), max(vals)


def summarize_repeats(rows, name, profile_name):
    """N 次重复行 → 汇总 dict（median/min/max）。'NA' 行忽略；全 NA → 'NA'。"""
    out = {'variant': name, 'model_profile': profile_name, 'repeats': len(rows)}
    for key, nd in (('tokens_per_sec', 0), ('step_per_sec', 3), ('gpu_util_avg', 1)):
        med, lo, hi = _stats([r.get(key) for r in rows])
        out[f'{key}_median'] = 'NA' if med is None else round(med, nd)
        out[f'{key}_min'] = 'NA' if lo is None else round(lo, nd)
        out[f'{key}_max'] = 'NA' if hi is None else round(hi, nd)
    return out


# ---------------------------------------------------------------- 输出
def _is_header(rec):
    return len(rec) > 1 and rec[0] == 'variant' and rec[1] == 'commit'


def normalize_results_csv(path=None, fields=None):
    """把 performance_results.csv 补齐成**统一宽度**；返回是否发生了改动。

    ⚠️ `path` 的默认值**不能**写成 `path=RESULTS_CSV`：Python 默认参数在**函数定义时**
    求值，那样会绑死 import 时刻的路径，使运行时改写 / monkeypatch `RESULTS_CSV` 全部失效
    （2026-09-12 被 test_perf_bench 当场抓出：测试规范化了真实文件而不是临时文件）。

    为什么需要：E0-4 当时要求"只追加、不重写历史文件"，但那会留下
    「表头 29 列 / 历史行 28 列」的 ragged 文件，`pandas.read_csv` 直接抛
    `ParserError: Error tokenizing data. C error: Expected 28 fields in line N, saw 29`
    （2026-09-12 实测）。仓库文档与作图脚本里都有 pandas 读 CSV 的用法，
    所以这是个会绊倒后来者的真实缺陷，不只是"请用别的读取函数"就能了事。

    做法：**只给短行补空字段**（缺 `repeat` 即补空，语义上就是"该次运行早于
    repeat 特性、重复次数未知"），**原有字段值逐字不动**；表头统一为 CSV_FIELDS。
    改前先备份 `*.bak`，写入走临时文件 + `os.replace`，避免中途失败留下半个文件。
    """
    fields = list(fields or CSV_FIELDS.split(','))
    path = RESULTS_CSV if path is None else path
    n = len(fields)
    if not os.path.exists(path):
        return False
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.reader(f) if r]

    fixed, changed = [], False
    for rec in rows:
        if _is_header(rec):
            if rec != fields:
                fixed.append(list(fields))
                changed = True
            else:
                fixed.append(rec)
        elif len(rec) != n:
            fixed.append(rec + [''] * (n - len(rec)) if len(rec) < n else rec[:n])
            changed = True
        else:
            fixed.append(rec)
    if not changed:
        return False

    shutil.copy2(path, path + '.bak')
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        # ⚠️ 必须显式 `lineterminator='\n'`：csv 模块在 Windows 上默认写 `\r\n`，
        #    而本仓库 `.gitattributes` 明确规定 `*.csv text eol=lf`（跨平台、diff 干净）。
        #    2026-09-12 用默认值写出过 CRLF，被 git 警告才发现。
        csv.writer(f, lineterminator='\n').writerows(fixed)
    os.replace(tmp, path)
    return True


def write_row(row):
    """追加一行到 performance_results.csv，并**保证文件结构始终一致**。

    写入前若发现任一行字段数与 CSV_FIELDS 不符（历史 28 列文件、或表头漏 repeat），
    先 `normalize_results_csv()` 补齐再追加 → 任何时刻 `pandas.read_csv` 都能直接读。
    """
    fields = CSV_FIELDS.split(',')
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
            w.writeheader()
            w.writerow(row)
        return
    if normalize_results_csv():
        print(f'ℹ️ performance_results.csv 存在字段数不一致的行（历史行缺 repeat 列）→ '
              f'已补齐为统一 {len(fields)} 列（短行补空 repeat，原有字段值逐字未改），'
              f'原文件备份为 .bak；现在 pandas.read_csv 可直接读取。', file=sys.stderr)
    with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=fields, lineterminator='\n').writerow(row)


def write_summary_row(row):
    """追加一行到 performance_summary.csv（新文件才写表头）。"""
    new = not os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS.split(','),
                           lineterminator='\n')
        if new:
            w.writeheader()
        w.writerow(row)


def read_results_rows(path=RESULTS_CSV, fields=None):
    """容错读取 performance_results.csv。

    历史文件表头少 `repeat`（E0-4 要求不改写历史文件），所以不依赖表头，
    直接按 CSV_FIELDS 的列序 zip：老行缺的字段为 None，新行的 repeat 正常读出。
    表头行自动跳过。
    """
    fields = list(fields or CSV_FIELDS.split(','))
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for rec in csv.reader(f):
            if not rec:
                continue
            if len(rec) > 1 and rec[0] == 'variant' and rec[1] == 'commit':
                continue                    # 表头
            rows.append({k: (rec[i] if i < len(rec) else None)
                         for i, k in enumerate(fields)})
    return rows


# ---------------------------------------------------------------- 单次运行
def run_once(name, cfg, profile_name, steps_total, rep):
    """跑一次 train.py 并把该次结果写成一行（repeat=rep，1-based）。"""
    prof = MODEL_PROFILES[profile_name]
    env = profile_env(profile_name)
    out_dir = os.path.join(OUT_ROOT, name)
    log_dir = os.path.join(out_dir, 'log')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # tokenizer pkl（train.py 在 out_dir 找）
    if env['tok_src'] and os.path.exists(env['tok_src']):
        shutil.copy(env['tok_src'], os.path.join(out_dir, os.path.basename(env['tok_src'])))
    step_csv = step_csv_path(log_dir, env['train_txt'])
    if os.path.exists(step_csv):
        os.remove(step_csv)

    log_every = cfg.get('log_every', LOG_EVERY)
    batch = cfg['batch']
    toks_per_step = batch * prof['block_size']
    cmd = build_train_cmd(cfg, profile_name, steps_total, out_dir, log_dir, env)

    # GPU 采样（后台）：**第一列 timestamp** 供 E0-2 的窗口过滤；文件名带 _r<rep>，
    # 绝不覆盖历史 gpu_metrics_<variant>.csv
    gpu_csv = gpu_csv_path(name, rep)
    gpu_proc = None
    try:
        if shutil.which('nvidia-smi'):
            with open(gpu_csv, 'w') as gf:
                gpu_proc = subprocess.Popen(
                    ['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,'
                     'power.draw,temperature.gpu,clocks.sm',
                     '--format=csv,noheader,nounits', '-l', '1'],
                    stdout=gf, stderr=subprocess.DEVNULL)
    except Exception:
        gpu_proc = None

    t0 = time.time()
    print(f'=== [{name} r{rep}/{steps_total}步] 启动: {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if gpu_proc is not None:
        gpu_proc.terminate()
        try:
            gpu_proc.wait(5)
        except Exception:
            pass

    log_text = r.stdout + r.stderr
    loss_start = loss_end = 'NA'
    amp_skipped = 0
    m = re.findall(r'AMP skipped (\d+) 步', log_text)
    if m:
        amp_skipped = int(m[-1])
    m = re.search(r'epoch 0 平均 train loss（每 \d+ 步采样）: ([\d.]+)', log_text)
    if m:
        loss_end = float(m.group(1))
    model_params = parse_model_params(log_text)
    if model_params == 'NA':
        print(f'⚠️ [{name}] 日志里没解析到 "GPT 参数量: N" → model_params 记 NA'
              f'（不写死，避免 100M 档被记成 50M 的数）', file=sys.stderr)

    # step_history 稳定段计时（最后 10 个采样行）+ GPU 采样窗口（E0-2）
    step_per_sec, tokens_per_sec, measured = compute_throughput(
        step_csv, toks_per_step)
    t_start, t_end = stable_segment_window(step_csv)
    if loss_start == 'NA':
        # loss_start: 首个采样行 train_loss
        if os.path.exists(step_csv):
            with open(step_csv) as f:
                for row in csv.DictReader(f):
                    try:
                        loss_start = float(row['train_loss'])
                        break
                    except (TypeError, ValueError):
                        continue

    g = parse_gpu_csv(gpu_csv, t_start, t_end) if os.path.exists(gpu_csv) else None
    nan_count = log_text.count('nan') + log_text.count('NaN')
    row = {
        'variant': name, 'commit': git_commit(),
        'model_params': model_params, 'batch_size': batch,
        'block_size': prof['block_size'],
        'precision': cfg['precision'], 'optimizer': 'AdamW',
        'fused': int(cfg['fused']), 'compile': int(cfg['compile']),
        'num_workers': cfg['workers'], 'log_every': log_every,
        'warmup_steps_measured': steps_total - measured,
        'measured_steps': measured, 'elapsed_seconds': round(elapsed, 1),
        'step_per_sec': round(step_per_sec, 3) if step_per_sec != 'NA' else 'NA',
        'tokens_per_sec': round(tokens_per_sec, 0) if tokens_per_sec != 'NA' else 'NA',
        'peak_vram_mb': int(g['peak_vram']) if g else 'NA',
        'gpu_util_avg': round(g['util_avg'], 1) if g else 'NA',
        'gpu_util_p50': round(g['util_p50'], 1) if g else 'NA',
        'gpu_util_p90': round(g['util_p90'], 1) if g else 'NA',
        'gpu_power_avg': round(g['power_avg'], 1) if g else 'NA',
        'cpu_util_avg': 'NA', 'ram_peak_mb': 'NA',
        'loss_start': loss_start, 'loss_end': loss_end,
        'nan_count': nan_count, 'amp_skipped': amp_skipped,
        'notes': cfg.get('notes', ''),
        'repeat': rep,
    }
    write_row(row)
    win = f', gpu窗口 {t_end - t_start:.1f}s/{g["n_samples"]}样本' if (g and t_start) else ''
    print(f'=== [{name} r{rep}] {row["tokens_per_sec"]} tok/s '
          f'(measured {measured} steps, {elapsed:.0f}s{win})', flush=True)
    return row


def run_variant(name, cfg, profile_name=None, steps_total=STEPS_TOTAL, repeat=1):
    """同一变体重复 `repeat` 次（E0-4）→ 每次一行 + 一行 median/min/max 汇总。"""
    if name not in VARIANTS:
        raise KeyError(f'未知 variant: {name}')
    prof_name = resolve_profile(name, cfg, profile_name)
    rows = [run_once(name, cfg, prof_name, steps_total, rep)
            for rep in range(1, repeat + 1)]
    s = summarize_repeats(rows, name, prof_name)
    write_summary_row(s)
    print(f'=== [{name}] summary: tokens/s median={s["tokens_per_sec_median"]} '
          f'min={s["tokens_per_sec_min"]} max={s["tokens_per_sec_max"]} '
          f'({s["repeats"]} 次, {prof_name})', flush=True)
    return rows


# ---------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--repeat', type=int, default=3,
                    help='同一变体重复次数（默认 3，E0-4）；CSV 末列 repeat 为 1-based')
    ap.add_argument('--model-profile', choices=sorted(MODEL_PROFILES), default=None,
                    help='配置档（缺省按变体绑定：B* 用 50m，C* 用 100m）')
    ap.add_argument('--steps', type=int, default=STEPS_TOTAL)
    ap.add_argument('--train-txt', default=None, help='覆盖配置档的 train txt')
    ap.add_argument('--val-txt', default=None, help='覆盖配置档的 val txt')
    ap.add_argument('--cache-dir', default=None, help='覆盖配置档的 tokenize cache 目录')
    ap.add_argument('--tok-src', default=None, help='覆盖配置档的 tokenizer pkl')
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    for key in ('train_txt', 'val_txt', 'cache_dir', 'tok_src'):
        if getattr(args, key):
            PATH_OVERRIDES[key] = getattr(args, key)
    os.makedirs(OUT_ROOT, exist_ok=True)

    if args.all:
        prof = args.model_profile or DEFAULT_PROFILE
        for name in ALL_ORDER[prof]:
            run_variant(name, VARIANTS[name], profile_name=prof,
                        steps_total=args.steps, repeat=args.repeat)
    elif args.variant:
        if args.variant not in VARIANTS:
            print(f'未知 variant: {args.variant}，可选: {list(VARIANTS)}')
            sys.exit(1)
        cfg = VARIANTS[args.variant]
        prof = resolve_profile(args.variant, cfg, args.model_profile)
        run_variant(args.variant, cfg, profile_name=prof,
                    steps_total=args.steps, repeat=args.repeat)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
