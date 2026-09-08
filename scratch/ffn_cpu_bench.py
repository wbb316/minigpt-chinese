"""CPU 小规模 FFN 变体对照（本地验证用，2026-09-08）

作用：在没有 GPU 的本地机器上，用**小模型 + 小语料**把三个 FFN 变体跑通并对比，
证明代码路径正确、对照流程闭环（参数对齐 / 起点一致 / 日志字段齐全），
**不用于得出 50M 结论**（那是云端 4090 的任务）。

对照条件（三者完全一致）：
  同一份语料切片 / seed 42 / batch 16 / block 64 / 4L-128d-4H
  同一 lr 恒温 8e-4、dropout 0.1、pack 模式
  起点：先跑 relu 得到 checkpoint，gelu 直接加载（同参数名），
        swiglu 由 ffn_adapter 把 fc1/fc2 映射到 gate/up/down

用法：python scratch/ffn_cpu_bench.py [--steps 400] [--parallel 2]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
OUT = os.path.join(ROOT, 'scratch', '_ffn_cpu')
CORPUS = os.path.join(OUT, 'corpus_slice.txt')
N_CHARS = 1_500_000
N_STEPS_DEFAULT = 400

COMMON = [
    '--sample-mode', 'pack', '--batch-size', '16', '--block-size', '64',
    '--n-layer', '4', '--n-head', '4', '--n-embd', '128',
    '--vocab-size', '2000', '--tokens-sample', '400000', '--bpe-trainer', 'fast',
    '--tie-embeddings', '--dropout', '0.1', '--position-encoding', 'rope',
    '--lr', '8e-4', '--lr-scheme', 'const', '--min-lr-ratio', '1.0',
    '--warmup-steps', '0', '--weight-decay', '0.05', '--seed', '42',
    '--val-every', '200', '--eval-batches', '10', '--log-every', '10',
    '--num-workers', '0', '--encode-workers', '1',
    '--cache-dir', os.path.join(OUT, 'cache'),
]


def make_corpus():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(CORPUS) and os.path.getsize(CORPUS) > N_CHARS * 0.9:
        return CORPUS
    src = os.path.join(ROOT, 'data', 'train_lightnovel.txt')
    with open(src, encoding='utf-8') as f:
        text = f.read(N_CHARS)
    with open(CORPUS, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'语料切片: {len(text):,} 字符 → {CORPUS}')
    return CORPUS


def run(ff, steps, init_from=None, tag=None, threads=4):
    tag = tag or ff
    out_dir = os.path.join(OUT, tag)
    log_dir = os.path.join(OUT, 'log', tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(OUT, f'run_{tag}.log')
    cmd = [PY, '-u', os.path.join(ROOT, 'train', 'train.py'),
           '--train-txt', CORPUS, '--val-txt', CORPUS,
           *COMMON, '--max-steps', str(steps),
           '--ff-type', ff,
           '--out-dir', out_dir, '--log-dir', log_dir]
    if init_from:
        cmd += ['--init-from', init_from]
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = str(threads)
    env['MKL_NUM_THREADS'] = str(threads)
    t0 = time.perf_counter()
    with open(log_path, 'w', encoding='utf-8') as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    return {'tag': tag, 'ff': ff, 'log': log_path,
            'wall': time.perf_counter() - t0,
            'ckpt': os.path.join(out_dir, 'checkpoint_best.pt'),
            'step_csv': os.path.join(log_dir, 'step_history_corpus_slice.csv')}


def parse_run(res):
    txt = open(res['log'], encoding='utf-8', errors='replace').read()
    def g(pat, cast=float, default=None):
        m = re.search(pat, txt)
        return cast(m.group(1)) if m else default
    res['params'] = g(r'GPT 参数量: ([\d,]+)', lambda s: int(s.replace(',', '')))
    res['ff_layer_params'] = g(r'层参数 ([\d,]+)', lambda s: int(s.replace(',', '')))
    res['ff_hidden'] = g(r'中间维 h=(\d+)', int) or g(r'中间维 (\d+)', int)
    res['tokens_per_s'] = g(r'([\d,]+) tokens/s', lambda s: int(s.replace(',', '')))
    res['step_per_s'] = g(r'([\d.]+) step/s')
    res['train_loss'] = g(r'平均 train loss（每 \d+ 步采样）: ([\d.]+)')
    res['ema'] = g(r'EMA ([\d.]+)')
    res['val'] = g(r'val ([\d.]+)')
    res['train_eval'] = g(r'train_eval\(无dropout\) ([\d.]+)')
    res['gap'] = g(r'gap ([+-][\d.]+)')
    res['init_ok'] = ('已从' in txt) or ('missing=[]' in txt)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=N_STEPS_DEFAULT)
    ap.add_argument('--parallel', type=int, default=2)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()

    make_corpus()
    os.makedirs(os.path.join(OUT, 'cache'), exist_ok=True)

    print(f'=== [1/2] relu 基线（{args.steps} 步，同时建立 token 缓存）===', flush=True)
    base = parse_run(run('relu', args.steps, threads=args.threads))
    print(f"    relu 完成: {base['wall']:.0f}s  EMA {base['ema']}", flush=True)

    print('=== [2/2] relu_sameinit + gelu + swiglu（同一 checkpoint 起点）===', flush=True)
    jobs = [('relu', 'relu_sameinit'), ('gelu', 'gelu'), ('swiglu', 'swiglu')]
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(run, ff, args.steps, base['ckpt'], tag, args.threads)
                for ff, tag in jobs]
        others = [parse_run(f.result()) for f in futs]

    rows = [base] + others
    print('\n=========== CPU 小规模 FFN 对照（仅验证代码路径，非 50M 结论）===========')
    hdr = (f"{'variant':13s} {'ff_h':>6s} {'ff_params':>10s} {'total':>10s} "
           f"{'train_EMA':>9s} {'val':>7s} {'train_ev':>8s} {'gap':>7s} "
           f"{'step/s':>7s} {'tok/s':>7s} {'init':>5s}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['tag']:13s} {str(r['ff_hidden'] or 0):>6s} "
              f"{r['ff_layer_params'] or 0:>10,} {r['params'] or 0:>10,} "
              f"{(r['ema'] or float('nan')):>9.4f} "
              f"{(r['val'] or float('nan')):>7.4f} "
              f"{(r['train_eval'] or float('nan')):>8.4f} "
              f"{(r['gap'] or float('nan')):>+7.4f} "
              f"{(r['step_per_s'] or 0):>7.2f} {(r['tokens_per_s'] or 0):>7,} "
              f"{'ok' if r['init_ok'] else 'N/A':>5s}")
    print('\n注：relu 行 = 从零随机初始化（跑 400 步建 checkpoint）；'
          'relu_sameinit / gelu / swiglu = 从**同一** checkpoint 出发，'
          '这三行才是同起点对照。')
    print(f"\n日志: {OUT}/run_*.log ｜ step CSV: {OUT}/log/*/step_history_corpus_slice.csv")


if __name__ == '__main__':
    main()
