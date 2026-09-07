# -*- coding: utf-8 -*-
"""TRAINING_PERFORMANCE_AUDIT 消融 benchmark 执行器（云端 4090 运行）。

用法（云端 /root，代码在 /root/train + /root/data）：
  python benchmark/perf_bench.py --variant B1          # 单组
  python benchmark/perf_bench.py --all                 # B0-B8 全部

每组流程：
  1. out/log/cache 指向 benchmark/perf_out/<variant>（tokenizer pkl 自动从 --tok-src 拷贝）
  2. 后台 nvidia-smi 1s 采样 → benchmark/gpu_metrics_<variant>.csv
  3. 前台跑 train/train.py（--max-steps N --val-every 0 --log-every 50）
  4. 聚合：step_history 稳定段（最后 10 个采样行线性斜率）→ step/s、tokens/s
  5. 解析 GPU 指标（mean/p50/p90）→ 一行写入 benchmark/performance_results.csv

参数与变体矩阵见 VARIANTS。train.py 真实参数在 variant 里配置。
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, 'benchmark')
OUT_ROOT = os.path.join(BENCH, 'perf_out')
RESULTS_CSV = os.path.join(BENCH, 'performance_results.csv')

CSV_FIELDS = ('variant,commit,model_params,batch_size,block_size,precision,'
              'optimizer,fused,compile,num_workers,log_every,'
              'warmup_steps_measured,measured_steps,elapsed_seconds,'
              'step_per_sec,tokens_per_sec,peak_vram_mb,gpu_util_avg,'
              'gpu_util_p50,gpu_util_p90,gpu_power_avg,cpu_util_avg,'
              'ram_peak_mb,loss_start,loss_end,nan_count,amp_skipped,notes')

# ---------- 云端真实路径（可被 --overrides 覆盖） ----------
TRAIN_TXT = '/root/autodl-tmp/data/train_webnovel_v2.txt'
VAL_TXT = '/root/autodl-tmp/data/val_webnovel_v2.txt'
CACHE_DIR = '/root/autodl-tmp/data'
TOK_SRC = '/root/result_50m/tokenizer_v6144_s4000000.pkl'
TRAIN_PY = os.path.join(ROOT, 'train', 'train.py')  # 云端 /root/train/train.py

MODEL_ARGS = ('--n-layer 12 --n-head 9 --n-embd 576 --block-size 512 '
              '--vocab-size 6144 --tie-embeddings --sample-mode pack '
              '--bpe-trainer fast --cache-format shards')
BASE_ARGS = (f'--lr 8e-4 --min-lr-ratio 0.0625 --warmup-steps 1000 '
             f'--weight-decay 0.05 --epochs 1 --patience 2 --eval-batches 0 '
             f'--val-every 0 --num-workers 8 --encode-workers 16')

STEPS_TOTAL = 900          # 每组总步（含 burn）；稳定段 = 最后 ~500 步
LOG_EVERY = 50             # step CSV 采样间隔（计时用）

# ---------- 变体矩阵（B0-B8 消融） ----------
VARIANTS = {
    # B0 基线：当前最新代码 + 默认参数（data pipeline 修复已在代码里，B0 即含 P0 修复的基线）
    'B0_current_baseline': dict(batch=64, precision='fp16', fused=False,
                                compile=False, workers=8, notes='P0 fixes included, logging every-step-equivalent(default 20)'),
    # B1 与 B0 相同（P0 修复已并入代码；本组用于确认 B0 可复现）
    'B1_logging_reduced': dict(batch=64, precision='fp16', fused=False,
                               compile=False, workers=8, log_every=50,
                               notes='log_every=50 采样（每步 item 消除）'),
    'B2_zerograd_h2d': dict(batch=64, precision='fp16', fused=False,
                            compile=False, workers=8, log_every=50,
                            notes='B1 + zero_grad set_to_none + non_blocking(已在代码，代码级)'),
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
}


def git_commit():
    try:
        r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                           cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() or 'NA'
    except Exception:
        return 'NA'


def parse_gpu_csv(path):
    """nvidia-smi 采样 CSV → (peak_vram, util_avg/p50/p90, power_avg)"""
    import numpy as np
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().strip('"').split(',')
            if len(parts) >= 5:
                try:
                    rows.append([float(p.strip().strip('"') or 'nan')
                                 for p in parts[:5]])
                except ValueError:
                    continue
    if not rows:
        return None
    a = np.array(rows)
    return dict(peak_vram=float(np.nanmax(a[:, 1])) if a[:, 1].size else 'NA',
                util_avg=float(np.nanmean(a[:, 0])),
                util_p50=float(np.nanpercentile(a[:, 0], 50)),
                util_p90=float(np.nanpercentile(a[:, 0], 90)),
                power_avg=float(np.nanmean(a[:, 2])))


def compute_throughput(step_csv, toks_per_step, seg=10):
    """从 step_history CSV（log_every 采样行）算稳定段吞吐。

    取最后 seg+1 个采样行：Δstep/Δtime → step/s、tokens/s。
    返回 (step_per_sec, tokens_per_sec, measured_steps) 或 ('NA','NA',0)。
    """
    if not os.path.exists(step_csv):
        return 'NA', 'NA', 0
    steps, times = [], []
    with open(step_csv) as f:
        for row in csv.DictReader(f):
            try:
                steps.append(int(row['step']))
                times.append(time.mktime(time.strptime(row['time'],
                                                       '%Y-%m-%d %H:%M:%S')))
            except (ValueError, KeyError):
                continue
    if len(times) < seg + 1:
        return 'NA', 'NA', 0
    ds = steps[-1] - steps[-1 - seg]
    dt = times[-1] - times[-1 - seg]
    if dt <= 0:
        return 'NA', 'NA', 0
    sps = ds / dt
    return round(sps, 3), round(sps * toks_per_step, 0), ds


def run_variant(name, cfg, steps_total=STEPS_TOTAL):
    out_dir = os.path.join(OUT_ROOT, name)
    log_dir = os.path.join(out_dir, 'log')
    cache_dir = os.path.join(out_dir, 'cache')   # cache 命中真实目录（传 CACHE_DIR）
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # tokenizer pkl（train.py 在 out_dir 找）
    if TOK_SRC and os.path.exists(TOK_SRC):
        shutil.copy(TOK_SRC, os.path.join(out_dir, os.path.basename(TOK_SRC)))
    step_csv = os.path.join(log_dir, 'step_history_train_webnovel_v2.csv')
    if os.path.exists(step_csv):
        os.remove(step_csv)

    log_every = cfg.get('log_every', LOG_EVERY)
    batch = cfg['batch']
    toks_per_step = batch * 512
    cmd = [sys.executable, '-u', TRAIN_PY,
           '--train-txt', TRAIN_TXT, '--val-txt', VAL_TXT,
           '--cache-dir', CACHE_DIR, '--out-dir', out_dir, '--log-dir', log_dir,
           '--batch-size', str(batch), '--precision', cfg['precision'],
           '--num-workers', str(cfg['workers']), '--log-every', str(log_every),
           '--max-steps', str(steps_total)]
    if cfg['fused']:
        cmd.append('--fused-adamw')
    if cfg['compile']:
        cmd.append('--compile')
    cmd += MODEL_ARGS.split() + BASE_ARGS.split()

    # GPU 采样（后台）
    gpu_csv = os.path.join(BENCH, f'gpu_metrics_{name}.csv')
    gpu_proc = None
    try:
        if shutil.which('nvidia-smi'):
            with open(gpu_csv, 'w') as gf:
                gpu_proc = subprocess.Popen(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,'
                     'power.draw,temperature.gpu,clocks.sm',
                     '--format=csv,noheader,nounits', '-l', '1'],
                    stdout=gf, stderr=subprocess.DEVNULL)
    except Exception:
        gpu_proc = None

    t0 = time.time()
    print(f'=== [{name}] 启动: {" ".join(cmd)}', flush=True)
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
    import re
    m = re.findall(r'AMP skipped (\d+) 步', log_text)
    if m:
        amp_skipped = int(m[-1])
    m = re.search(r'epoch 0 平均 train loss（每 \d+ 步采样）: ([\d.]+)', log_text)
    if m:
        loss_end = float(m.group(1))

    # step_history 稳定段计时（最后 10 个采样行）
    step_per_sec, tokens_per_sec, measured = compute_throughput(
        step_csv, toks_per_step)
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

    g = parse_gpu_csv(gpu_csv) if os.path.exists(gpu_csv) else None
    nan_count = log_text.count('nan') + log_text.count('NaN')
    row = {
        'variant': name, 'commit': git_commit(),
        'model_params': 51411840, 'batch_size': batch, 'block_size': 512,
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
    }
    write_row(row)
    print(f'=== [{name}] {row["tokens_per_sec"]} tok/s '
          f'(measured {measured} steps, {elapsed:.0f}s)', flush=True)
    return row


def write_row(row):
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS.split(','))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    global TRAIN_TXT, VAL_TXT, CACHE_DIR, TOK_SRC
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--steps', type=int, default=STEPS_TOTAL)
    ap.add_argument('--train-txt', default=TRAIN_TXT)
    ap.add_argument('--val-txt', default=VAL_TXT)
    ap.add_argument('--cache-dir', default=CACHE_DIR)
    ap.add_argument('--tok-src', default=TOK_SRC)
    args = ap.parse_args()

    TRAIN_TXT, VAL_TXT, CACHE_DIR, TOK_SRC = (args.train_txt, args.val_txt,
                                              args.cache_dir, args.tok_src)
    os.makedirs(OUT_ROOT, exist_ok=True)

    if args.all:
        order = ['B0_current_baseline', 'B1_logging_reduced',
                 'B2_zerograd_h2d', 'B3_dl_w0', 'B3_dl_w2', 'B3_dl_w4',
                 'B3_dl_w8', 'B3_dl_w12', 'B5_fused_adamw', 'B6_compile',
                 'B7_bf16', 'B8_batch80', 'B8_batch96']
        for name in order:
            run_variant(name, VARIANTS[name], steps_total=args.steps)
    elif args.variant:
        if args.variant not in VARIANTS:
            print(f'未知 variant: {args.variant}，可选: {list(VARIANTS)}')
            sys.exit(1)
        run_variant(args.variant, VARIANTS[args.variant], steps_total=args.steps)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
