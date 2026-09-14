# -*- coding: utf-8 -*-
"""benchmark/perf_bench.py 的纯函数测试（E0-2 / E0-3 / E0-4 / E0-5）。

本地没有 GPU、没有云端 shard12 数据 → 这里只覆盖**可离线验证**的部分：
GPU CSV 解析与窗口过滤、nvidia-smi 时间戳解析、吞吐口径（新旧 step CSV）、
重复汇总（median/min/max）、配置档与变体矩阵、训练命令拼装、CLI。
真实性能数字（C0 在云端跑通、B6 回归量级）必须云端实测，本文件不假装覆盖。
"""
import csv
import datetime
import os
import re
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'benchmark'))
import perf_bench as pb  # noqa: E402

OLD_STEP_HEADER = 'step,epoch,tokens_seen,train_loss,val_loss,lr,tokens_per_sec,time'
NEW_STEP_HEADER = OLD_STEP_HEADER + ',time_unix'
# 历史 50m 配置（保护项）：12L/9H/576d / vocab6144 / shard0（与改动前逐字一致）
HIST_MODEL_ARGS_50M = ('--n-layer 12 --n-head 9 --n-embd 576 --block-size 512 '
                       '--vocab-size 6144 --tie-embeddings --sample-mode pack '
                       '--bpe-trainer fast --cache-format shards')
B_SERIES_50M = {            # variant -> (batch, precision, fused, compile, workers)
    'B0_current_baseline': (64, 'fp16', False, False, 8),
    'B1_logging_reduced': (64, 'fp16', False, False, 8),
    'B3_dl_w0': (64, 'fp16', False, False, 0),
    'B3_dl_w2': (64, 'fp16', False, False, 2),
    'B3_dl_w4': (64, 'fp16', False, False, 4),
    'B3_dl_w8': (64, 'fp16', False, False, 8),
    'B3_dl_w12': (64, 'fp16', False, False, 12),
    'B5_fused_adamw': (64, 'fp16', True, False, 8),
    'B6_compile': (64, 'fp16', False, True, 8),
    'B7_bf16': (64, 'bf16', False, False, 8),
    'B8_batch80': (80, 'fp16', False, False, 8),
    'B8_batch96': (96, 'fp16', False, False, 8),
    'B9_compile_batch80': (80, 'fp16', False, True, 8),
    'B9_compile_batch64_rep': (64, 'fp16', False, True, 8),
}
# 旧 5 列 GPU 采样（无 timestamp）：util 83/99/10/0，mem 18500 峰值，power 均值 178.325
OLD_GPU = ['83, 17263, 293.3, 55, 2100',
           '99, 18500, 300.0, 60, 2110',
           '10, 9000, 100.0, 40, 1000',
           '0, 100, 20.0, 30, 200']
# 新 6 列（timestamp + 原 5 列），1s 一行、2s 一个采样
TS_GPU = ['"2026/09/12 10:00:00.000", 10, 1000, 50.0, 40, 1000',
          '"2026/09/12 10:00:02.000", 30, 2000, 60.0, 41, 1010',
          '"2026/09/12 10:00:04.000", 90, 3000, 70.0, 42, 1020',
          '"2026/09/12 10:00:06.000", 100, 4000, 80.0, 43, 1030']


def dump(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(p)


def make_step_rows_new(n=11, step_every=10, dt=2.0, t0=1000.0, time_col_dt=1.0):
    """新 9 列 step CSV：`time` 列按 time_col_dt 走、`time_unix` 按 dt 走（用于验证优先级）。"""
    rows = [NEW_STEP_HEADER]
    for k in range(n):
        step = k * step_every
        wall = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0 + k * time_col_dt))
        rows.append(f'{step},0.001,{step * 64 * 512},5.0000,,8.00e-04,8000,'
                    f'{wall},{t0 + k * dt:.3f}')
    return rows


# ================================================================ E0-2: GPU telemetry
def test_parse_gpu_csv_old_format_no_window(tmp_path):
    """旧 5 列 CSV（无 timestamp）：行为与改动前完全一致（全部手算核对）。"""
    p = dump(tmp_path, 'gpu_old.csv', OLD_GPU)
    g = pb.parse_gpu_csv(p)
    assert g['n_samples'] == 4
    assert g['util_avg'] == pytest.approx(48.0)             # (83+99+10+0)/4
    assert g['util_p50'] == pytest.approx(46.5)             # 排序 [0,10,83,99] 线性插值
    assert g['util_p90'] == pytest.approx(94.2)
    assert g['peak_vram'] == pytest.approx(18500)
    assert g['power_avg'] == pytest.approx(178.325)


def test_parse_gpu_csv_window_keeps_only_inside_samples(tmp_path):
    """带窗口时 mean/p50/p90 **与峰值显存**都只统计窗口内样本。"""
    p = dump(tmp_path, 'gpu_ts.csv', TS_GPU)
    ts = [pb.parse_gpu_ts(line.split(',')[0]) for line in TS_GPU]
    all_g = pb.parse_gpu_csv(p)
    assert all_g['n_samples'] == 4 and all_g['peak_vram'] == 4000
    assert all_g['util_avg'] == pytest.approx(57.5)

    g = pb.parse_gpu_csv(p, ts[1], ts[2])                   # 只剩 30 与 90 两个样本
    assert g['n_samples'] == 2
    assert g['util_avg'] == pytest.approx(60.0)
    assert g['util_p50'] == pytest.approx(60.0)
    assert g['util_p90'] == pytest.approx(84.0)             # 30 + 0.9*(90-30)
    assert g['peak_vram'] == pytest.approx(3000)            # 峰值显存也限定窗口
    assert g['power_avg'] == pytest.approx(65.0)


def test_parse_gpu_csv_window_open_ended(tmp_path):
    p = dump(tmp_path, 'gpu_ts.csv', TS_GPU)
    ts = [pb.parse_gpu_ts(line.split(',')[0]) for line in TS_GPU]
    assert pb.parse_gpu_csv(p, t_start=ts[2])['util_avg'] == pytest.approx(95.0)   # 90,100
    assert pb.parse_gpu_csv(p, t_end=ts[1])['util_avg'] == pytest.approx(20.0)     # 10,30


def test_parse_gpu_csv_window_on_old_csv_is_noop(tmp_path):
    """旧 CSV 无 timestamp → 给窗口也不过滤（E0-2 向后兼容）。"""
    p = dump(tmp_path, 'gpu_old.csv', OLD_GPU)
    assert pb.parse_gpu_csv(p, t_start=0.0, t_end=1.0) == pb.parse_gpu_csv(p)


def test_parse_gpu_csv_window_with_no_sample_falls_back(tmp_path):
    """窗口里一个样本都没有 → 退回全段（打警告），而不是返回 None。"""
    p = dump(tmp_path, 'gpu_ts.csv', TS_GPU)
    ts = [pb.parse_gpu_ts(line.split(',')[0]) for line in TS_GPU]
    g = pb.parse_gpu_csv(p, ts[-1] + 3600, ts[-1] + 7200)
    assert g['n_samples'] == 4 and g['peak_vram'] == 4000


def test_parse_gpu_csv_bad_lines_skipped(tmp_path):
    """坏行（列数不足 / 数值非数 / nvidia-smi 报错文本）整行跳过，不崩。"""
    lines = ['83, 17263, 293.3, 55, 2100',
             'nvidia-smi: command not found',
             '',
             '1,2,3',                                       # 列数不足
             '7, 8, 9, 10, 11',
             '42, 17263, notanumber, 55, 2100',             # 数值坏
             '"2026/09/12 10:00:00.000", 90, 3000, 70.0, 42, 1020']
    p = dump(tmp_path, 'gpu_mix.csv', lines)
    g = pb.parse_gpu_csv(p)
    assert g['n_samples'] == 3
    assert g['util_avg'] == pytest.approx((83 + 7 + 90) / 3)
    assert g['peak_vram'] == pytest.approx(17263)
    # 有 timestamp 的样本存在时，给窗口 → 只有能定位到时间里的样本入选
    t = pb.parse_gpu_ts('"2026/09/12 10:00:00.000"')
    g2 = pb.parse_gpu_csv(p, t - 1, t + 1)
    assert g2['n_samples'] == 1 and g2['util_avg'] == pytest.approx(90.0)


def test_parse_gpu_csv_empty_or_missing(tmp_path):
    p = dump(tmp_path, 'gpu_empty.csv', [''])
    assert pb.parse_gpu_csv(p) is None
    assert pb.parse_gpu_csv(str(tmp_path / 'nope.csv')) is None


def test_parse_gpu_ts_formats():
    """nvidia-smi timestamp：带/不带小数秒、带引号；毫秒必须保住；坏行返回 None。"""
    exp_ms = datetime.datetime(2026, 9, 12, 10, 30, 45, 123000).timestamp()
    exp_s = datetime.datetime(2026, 9, 12, 10, 30, 45).timestamp()
    assert pb.parse_gpu_ts('2026/09/12 10:30:45.123') == pytest.approx(exp_ms, abs=1e-6)
    assert pb.parse_gpu_ts('"2026/09/12 10:30:45.123"') == pytest.approx(exp_ms, abs=1e-6)
    assert pb.parse_gpu_ts('"2026/09/12 10:30:45"') == pytest.approx(exp_s, abs=1e-6)
    assert pb.parse_gpu_ts(' 2026/09/12 10:30:45 ') == pytest.approx(exp_s, abs=1e-6)
    # 毫秒精度：strptime 丢 %f 的实现会在这里挂掉
    assert (pb.parse_gpu_ts('2026/09/12 10:30:45.999')
            - pb.parse_gpu_ts('2026/09/12 10:30:45.000')) == pytest.approx(0.999, abs=1e-3)
    assert pb.parse_gpu_ts('2026-09-12T10:30:45.123') == pytest.approx(exp_ms, abs=1e-6)
    for bad in ('', '   ', 'garbage', '83', 'NA', 'nvidia-smi: not found', '2026/13/45 99:99:99'):
        assert pb.parse_gpu_ts(bad) is None


# ================================================================ E0-1/E0-2: 吞吐与窗口
def test_compute_throughput_new_9col_uses_time_unix(tmp_path):
    """新 9 列 CSV：用 time_unix（毫秒精度的 Δt），而不是秒级 time 列。"""
    p = dump(tmp_path, 'step_new.csv', make_step_rows_new(n=11, dt=2.0, time_col_dt=1.0))
    sps, tps, measured = pb.compute_throughput(p, 64 * 512, seg=10)
    assert measured == 100                                  # 10 行 × 10 step
    assert sps == pytest.approx(5.0)                        # 100 step / (10×2s)
    assert tps == pytest.approx(5.0 * 64 * 512)             # 163840


def test_compute_throughput_old_8col_falls_back_to_time(tmp_path):
    """旧 8 列 CSV：退回 `time` 列（秒级），结果与改动前一致。"""
    t0 = time.mktime(time.strptime('2026-09-12 10:00:00', '%Y-%m-%d %H:%M:%S'))
    rows = [OLD_STEP_HEADER]
    for k in range(11):
        wall = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0 + k * 3))
        rows.append(f'{k * 10},0.001,{k * 10 * 64 * 512},5.0000,,8.00e-04,8000,{wall}')
    p = dump(tmp_path, 'step_old.csv', rows)
    sps, tps, measured = pb.compute_throughput(p, 64 * 512, seg=10)
    assert measured == 100
    assert sps == pytest.approx(round(100 / 30.0, 3))       # 返回值按 round(...,3) 口径
    assert tps == pytest.approx(round(100 / 30.0 * 64 * 512, 0))


def test_compute_throughput_na_when_too_few_rows(tmp_path):
    p = dump(tmp_path, 'step_short.csv', make_step_rows_new(n=5))
    assert pb.compute_throughput(p, 64 * 512, seg=10) == ('NA', 'NA', 0)
    assert pb.compute_throughput(str(tmp_path / 'none.csv'), 64 * 512) == ('NA', 'NA', 0)


def test_read_step_samples_skips_bad_rows(tmp_path):
    rows = [NEW_STEP_HEADER,
            'abc,0,0,5,,8e-04,0,2026-09-12 10:00:00,1.0',     # step 坏 → 跳过
            '10,0,0,5,,8e-04,0,2026-09-12 10:00:02,1001.5',
            '20,0,0,5,,8e-04,0,bad-time,1003.5']             # time 坏但 time_unix 有效 → 保留
    p = dump(tmp_path, 'step_mix.csv', rows)
    assert pb.read_step_samples(p) == [(10, pytest.approx(1001.5)),
                                       (20, pytest.approx(1003.5))]


def test_read_step_samples_per_row_fallback(tmp_path):
    """time_unix 为空的行退回 time 列（不整行丢弃）。"""
    rows = [NEW_STEP_HEADER,
            '10,0,0,5,,8e-04,0,2026-09-12 10:00:00,',
            '20,0,0,5,,8e-04,0,2026-09-12 10:00:05,1005.0']
    p = dump(tmp_path, 'step_partial.csv', rows)
    s = pb.read_step_samples(p)
    assert s[1] == (20, pytest.approx(1005.0))
    assert s[0][1] == pytest.approx(time.mktime(
        time.strptime('2026-09-12 10:00:00', '%Y-%m-%d %H:%M:%S')))


def test_stable_segment_window_new_csv(tmp_path):
    p = dump(tmp_path, 'step_new.csv', make_step_rows_new(n=11, dt=2.0))
    t0, t1 = pb.stable_segment_window(p)
    assert (t0, t1) == (1000.0, 1020.0)
    assert pb.has_time_unix(p)


def test_stable_segment_window_old_csv_is_none(tmp_path):
    """旧 8 列 CSV → 不给窗口（E0-2 要求旧文件行为不变）。"""
    rows = [OLD_STEP_HEADER] + [f'{k * 10},0,0,5,,8e-04,0,2026-09-12 10:00:00'
                                for k in range(11)]
    p = dump(tmp_path, 'step_old.csv', rows)
    assert not pb.has_time_unix(p)
    assert pb.stable_segment_window(p) == (None, None)


def test_stable_segment_window_too_few_rows(tmp_path):
    p = dump(tmp_path, 'step_short.csv', make_step_rows_new(n=5))
    assert pb.stable_segment_window(p) == (None, None)


# 仓库里的真实历史 step CSV（7/8 列，无 time_unix）：50M v3_alpha 与 100M shard12
REAL_STEP_CSVS = [
    os.path.join(ROOT, 'log', '50M参数_v3_alpha+998Mtokens',
                 'step_history_train_webnovel_v2.csv'),
    os.path.join(ROOT, 'log', '100M参数_v3_2Btokens',
                 'step_history_train_webnovel_shard12.csv'),
]


@pytest.mark.parametrize('path', REAL_STEP_CSVS)
def test_real_archived_step_csv_backcompat(path):
    """真实历史 CSV（7/8 列，无 time_unix）→ 旧 `time` 口径仍能算，窗口给 None（E0-2 兼容）。"""
    if not os.path.exists(path):
        pytest.skip('归档 CSV 不在仓库里')
    samples = pb.read_step_samples(path)
    assert len(samples) > 100
    assert not pb.has_time_unix(path)
    assert pb.stable_segment_window(path) == (None, None)
    sps, tps, measured = pb.compute_throughput(path, 64 * 512)
    assert sps > 0 and tps > 0 and measured > 0


def test_gpu_csv_path_never_overwrites_history():
    """GPU 采样文件名带 _r<n>：历史的 gpu_metrics_<variant>.csv 绝不被写。"""
    p = pb.gpu_csv_path('B6_compile', 1)
    assert os.path.basename(p) == 'gpu_metrics_B6_compile_r1.csv'
    assert p != os.path.join(pb.BENCH, 'gpu_metrics_B6_compile.csv')


# ================================================================ E0-4: 重复测量
def test_summarize_repeats_median_min_max():
    rows = [{'tokens_per_sec': 100.0, 'step_per_sec': 1.0, 'gpu_util_avg': 95.0},
            {'tokens_per_sec': 200.0, 'step_per_sec': 2.0, 'gpu_util_avg': 98.0},
            {'tokens_per_sec': 300.0, 'step_per_sec': 3.0, 'gpu_util_avg': 97.0}]
    s = pb.summarize_repeats(rows, 'B6_compile', '50m')
    assert (s['tokens_per_sec_median'], s['tokens_per_sec_min'],
            s['tokens_per_sec_max']) == (200, 100, 300)
    assert (s['step_per_sec_median'], s['step_per_sec_min'], s['step_per_sec_max']) == (2.0, 1.0, 3.0)
    assert (s['gpu_util_avg_median'], s['gpu_util_avg_min'], s['gpu_util_avg_max']) == (97.0, 95.0, 98.0)
    assert s['variant'] == 'B6_compile' and s['model_profile'] == '50m' and s['repeats'] == 3


def test_summarize_repeats_ignores_na():
    rows = [{'tokens_per_sec': 'NA', 'step_per_sec': 'NA', 'gpu_util_avg': 'NA'},
            {'tokens_per_sec': 100.0, 'step_per_sec': 1.0, 'gpu_util_avg': 90.0},
            {'tokens_per_sec': 200.0, 'step_per_sec': 2.0, 'gpu_util_avg': 92.0}]
    s = pb.summarize_repeats(rows, 'B8_batch96', '50m')
    assert s['tokens_per_sec_median'] == 150                # 2 个有效值 → 中位取平均
    assert s['repeats'] == 3                                # 仍是 3 次（含失败那次）
    s_all_na = pb.summarize_repeats(
        [{'tokens_per_sec': 'NA', 'step_per_sec': 'NA', 'gpu_util_avg': 'NA'}], 'x', '50m')
    assert s_all_na['tokens_per_sec_median'] == 'NA'
    assert s_all_na['gpu_util_avg_max'] == 'NA'


def test_summary_fields_and_repeat_column():
    fields = pb.CSV_FIELDS.split(',')
    assert fields[-1] == 'repeat' and fields[-2] == 'notes'  # 新列追加在末尾
    assert len(fields) == 29                                 # 历史 28 列 + repeat
    sf = pb.SUMMARY_FIELDS.split(',')
    for key in ('tokens_per_sec_median', 'tokens_per_sec_min', 'tokens_per_sec_max',
                'step_per_sec_median', 'step_per_sec_min', 'step_per_sec_max',
                'gpu_util_avg_median', 'gpu_util_avg_min', 'gpu_util_avg_max'):
        assert key in sf
    assert sf[0] == 'variant'


def test_write_row_normalizes_ragged_csv(tmp_path, monkeypatch, capsys):
    """历史 28 列文件在写入前被补齐为统一宽度并备份；原有字段值逐字不变。

    ⚠️ 2026-09-12 契约变更：E0-4 原契约是「只追加、从不重写历史文件」，
    但那会留下「表头 29 列 / 历史行 28 列」的 ragged 文件，`pandas.read_csv`
    直接抛 `Expected 28 fields in line N, saw 29`（实测）。故改为写入时自动补齐
    —— 只给短行补**空的 repeat**，原有字段值不动。
    """
    old_fields = pb.CSV_FIELDS.split(',')[:-1]
    n = len(pb.CSV_FIELDS.split(','))
    old_row = ['B0'] + ['1'] * (len(old_fields) - 1)
    p = tmp_path / 'results.csv'
    p.write_text(','.join(old_fields) + '\n' + ','.join(old_row) + '\n', encoding='utf-8')
    monkeypatch.setattr(pb, 'RESULTS_CSV', str(p))

    row = {f: '1' for f in old_fields}
    row.update(variant='B6_compile', commit='abc', repeat=2)
    pb.write_row(row)

    with open(p, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    assert all(len(r) == n for r in rows)              # 全文件统一宽度 → pandas 可读
    assert rows[0] == pb.CSV_FIELDS.split(',')         # 表头被补齐
    assert rows[1][:n - 1] == old_row                  # 历史字段值逐字未改
    assert rows[1][-1] == ''                           # 只补了空 repeat
    assert rows[-1][0] == 'B6_compile' and rows[-1][-1] == '2'
    assert (tmp_path / 'results.csv.bak').exists()     # 改写前有备份
    assert 'repeat' in capsys.readouterr().err

    # 幂等：文件已一致时再写一次，不应再次改写（.bak 不变，只追加）
    before2 = p.read_text(encoding='utf-8')
    bak_before = (tmp_path / 'results.csv.bak').read_text(encoding='utf-8')
    pb.write_row(dict(row, variant='B7_bf16'))
    assert (tmp_path / 'results.csv.bak').read_text(encoding='utf-8') == bak_before
    assert p.read_text(encoding='utf-8').startswith(before2)


def test_write_row_new_file_has_new_header(tmp_path, monkeypatch):
    monkeypatch.setattr(pb, 'RESULTS_CSV', str(tmp_path / 'new.csv'))
    row = {f: '' for f in pb.CSV_FIELDS.split(',')}
    row.update(variant='C0_100m_baseline', commit='NA', repeat=1)
    pb.write_row(row)
    with open(tmp_path / 'new.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == pb.CSV_FIELDS.split(',')
    assert rows[0]['repeat'] == '1'


def test_read_results_rows_tolerates_old_header_and_short_rows(tmp_path):
    """老行（28 列，无 repeat）读出 repeat=None；新行（29 列）按位置读出 repeat。"""
    old_fields = pb.CSV_FIELDS.split(',')[:-1]
    p = tmp_path / 'results.csv'
    p.write_text(','.join(old_fields) + '\n'
                 + ','.join(['B0'] + ['1'] * (len(old_fields) - 1)) + '\n'
                 + ','.join(['B6'] + ['1'] * (len(old_fields) - 1) + ['2']) + '\n',
                 encoding='utf-8')
    rows = pb.read_results_rows(str(p))
    assert [r['variant'] for r in rows] == ['B0', 'B6']       # 表头行被跳过
    assert rows[0]['repeat'] is None
    assert rows[1]['repeat'] == '2'
    assert rows[1]['notes'] == '1'


def test_write_summary_row_creates_header(tmp_path, monkeypatch):
    monkeypatch.setattr(pb, 'SUMMARY_CSV', str(tmp_path / 'summary.csv'))
    s = pb.summarize_repeats([{'tokens_per_sec': 100.0, 'step_per_sec': 1.0,
                               'gpu_util_avg': 90.0}], 'C0_100m_baseline', '100m')
    pb.write_summary_row(s)
    with open(tmp_path / 'summary.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == pb.SUMMARY_FIELDS.split(',')
    assert rows[0]['variant'] == 'C0_100m_baseline' and rows[0]['model_profile'] == '100m'
    assert rows[0]['tokens_per_sec_median'] == '100.0'


# ================================================================ E0-5: 配置档
def test_50m_profile_is_identical_to_history():
    """回归保护：50m 档的**模型配置**必须与改动前逐字一致。

    注意区分两类字段：`model_args` / `block_size` 属**模型配置**（一变就破坏历史可比性，
    必须锁死）；`train_txt` / `val_txt` 属**环境路径** —— 2026-09-12 修正为系统盘
    `/root/data/`（原写成数据盘 `/root/autodl-tmp/data/`，与实际不符，
    导致每次跑都得手工 `--train-txt/--val-txt` 覆盖）。
    """
    p50 = pb.MODEL_PROFILES['50m']
    assert p50['model_args'] == HIST_MODEL_ARGS_50M
    assert p50['block_size'] == 512
    assert p50['train_txt'] == '/root/data/train_webnovel_v2.txt'
    assert p50['val_txt'] == '/root/data/val_webnovel_v2.txt'
    assert p50['cache_dir'] == '/root/autodl-tmp/data'
    assert p50['tok_src'] == '/root/result_50m/tokenizer_v6144_s4000000.pkl'
    assert pb.DEFAULT_PROFILE == '50m'


def test_100m_profile_arch_and_paths():
    p = pb.MODEL_PROFILES['100m']
    for tok in ('--n-layer 16', '--n-head 11', '--n-embd 704', '--block-size 512',
                '--vocab-size 8192'):
        assert tok in p['model_args']
    assert p['block_size'] == 512
    assert p['train_txt'] == '/root/data/train_webnovel_shard12.txt'
    assert p['val_txt'] == '/root/data/val_webnovel_shard12.txt'
    assert os.path.basename(p['train_txt']) == 'train_webnovel_shard12.txt'
    assert os.path.basename(p['val_txt']) == 'val_webnovel_shard12.txt'
    assert os.path.basename(p['tok_src']) == 'tokenizer_v8192_s4000000.pkl'
    # 2026-09-12：语料位置与 tokenizer 均已在云端**实测确认**（见下方注释里的实测记录），
    # 所以 `TODO(云端确认)` 未验证标记应已移除。原先本测试断言 "TODO in src"，
    # 那是把「未验证」当成了永久契约 —— 验证完成后它必然失败。
    src = open(os.path.join(ROOT, 'benchmark', 'perf_bench.py'), encoding='utf-8').read()
    assert 'TODO(云端确认)' not in src


@pytest.mark.parametrize('prof_name', ['50m', '100m'])
def test_profile_block_size_matches_model_args(prof_name):
    prof = pb.MODEL_PROFILES[prof_name]
    assert str(prof['block_size']) == re.search(r'--block-size\s+(\d+)',
                                                prof['model_args']).group(1)


@pytest.mark.parametrize('prof_name,expect', [('50m', 51421056), ('100m', 101457280)])
def test_profile_param_count_by_local_torch(prof_name, expect):
    """用本地 torch(CPU) 按 profile 建模型数参数 —— C0 验收数字 101,457,280 的离线证据。

    只证明「配置 → 参数量」这一环；云端真跑通仍需 GPU。
    """
    pytest.importorskip('torch')
    from model.gpt import GPT
    s = pb.MODEL_PROFILES[prof_name]['model_args']
    arch = {k: int(re.search(rf'{flag}\s+(\d+)', s).group(1)) for k, flag in
            (('n_layer', '--n-layer'), ('n_head', '--n-head'), ('n_embd', '--n-embd'),
             ('vocab_size', '--vocab-size'), ('block_size', '--block-size'))}
    m = GPT(dropout=0.1, tie_embeddings=True, **arch)
    assert m.get_num_params() == expect


def test_100m_profile_matches_archived_production_run_config():
    """C 系列必须对齐已归档的生产 run（log/100M参数_v3_2Btokens/run_config.txt）。"""
    txt = open(os.path.join(ROOT, 'log', '100M参数_v3_2Btokens', 'run_config.txt'),
               encoding='utf-8', errors='ignore').read()
    assert re.search(r'n_layer:\s+16', txt)
    assert re.search(r'n_head:\s+11', txt)
    assert re.search(r'n_embd:\s+704', txt)
    assert re.search(r'vocab_size:\s+8192', txt)
    assert re.search(r'block_size:\s+512', txt)
    assert re.search(r'parameters:\s+101,457,280', txt)
    c0 = pb.VARIANTS['C0_100m_baseline']
    assert (c0['batch'], c0['precision'], c0['compile'], c0['workers']) == (64, 'fp16', True, 8)
    assert c0['log_every'] == int(re.search(r'log_every:\s+(\d+)', txt).group(1))


def test_step_csv_path_follows_train_txt():
    """step_csv 不能再硬编码 v2（100M 是 shard12）。"""
    assert pb.step_csv_path('/log', pb.MODEL_PROFILES['50m']['train_txt']) == \
        os.path.join('/log', 'step_history_train_webnovel_v2.csv')
    assert pb.step_csv_path('/log', pb.MODEL_PROFILES['100m']['train_txt']) == \
        os.path.join('/log', 'step_history_train_webnovel_shard12.csv')


def test_profile_env_cli_override(monkeypatch):
    monkeypatch.setitem(pb.PATH_OVERRIDES, 'train_txt', '/tmp/my.txt')
    env = pb.profile_env('50m')
    assert env['train_txt'] == '/tmp/my.txt'
    assert env['val_txt'] == pb.MODEL_PROFILES['50m']['val_txt']
    assert env['tok_src'] == pb.MODEL_PROFILES['50m']['tok_src']


def test_resolve_profile_binding_and_override(capsys):
    assert pb.resolve_profile('C0_100m_baseline', pb.VARIANTS['C0_100m_baseline']) == '100m'
    assert pb.resolve_profile('B6_compile', pb.VARIANTS['B6_compile']) == '50m'
    assert pb.resolve_profile('B6_compile', pb.VARIANTS['B6_compile'], '100m') == '100m'
    assert '覆盖' in capsys.readouterr().err


# ================================================================ E0-5: model_params 解析
def test_parse_model_params():
    assert pb.parse_model_params('GPT 参数量: 101,457,280（含 tie_embeddings，已去重）') == 101457280
    assert pb.parse_model_params('GPT 参数量: 51,421,056') == 51421056
    assert pb.parse_model_params('') == 'NA'
    assert pb.parse_model_params('位置编码: rope\nFFN: swiglu') == 'NA'
    assert pb.parse_model_params('GPT 参数量: abc') == 'NA'


def test_parse_model_params_on_real_archived_log():
    """真实训练日志（仓库归档）→ 必须解析出 101,457,280。"""
    log = os.path.join(ROOT, 'log', '100M参数_v3_2Btokens', 'train_100m.log')
    with open(log, encoding='utf-8', errors='ignore') as f:
        head = f.read(200000)
    assert pb.parse_model_params(head) == 101457280


# ================================================================ E0-3: 变体矩阵
def test_b2_empty_variant_removed():
    assert 'B2_zerograd_h2d' not in pb.VARIANTS
    assert 'B2_zerograd_h2d' not in pb.ALL_ORDER['50m']


def test_b0_uses_real_train_default_log_every():
    b0 = pb.VARIANTS['B0_current_baseline']
    assert b0['log_every'] == 20                              # train.py --log-every 默认 20
    assert 'every-step-equivalent' not in b0['notes']         # 旧 notes 是错的
    assert pb.VARIANTS['B1_logging_reduced']['log_every'] == 50
    # B0/B1 必须是真正不同的配置（改动前两者都是 50 → 同一配置跑两遍）
    assert b0 != pb.VARIANTS['B1_logging_reduced']


def test_b9_repeat_added_to_all_order():
    assert 'B9_compile_batch64_rep' in pb.ALL_ORDER['50m']
    assert 'B9_compile_batch64_rep' in pb.VARIANTS


def test_all_order_names_exist_and_profile_consistent():
    for prof, order in pb.ALL_ORDER.items():
        assert order, prof
        assert len(order) == len(set(order))
        for name in order:
            assert name in pb.VARIANTS, name
            assert (pb.VARIANTS[name].get('profile') or '50m') == prof
    assert set(pb.ALL_ORDER['100m']) == {'C0_100m_baseline', 'C1_100m_nocompile'}
    assert pb.VARIANTS['C0_100m_baseline']['compile'] is True
    assert pb.VARIANTS['C1_100m_nocompile']['compile'] is False
    # C1 与 C0 只差 compile（notes 只是文案，可不同）
    c0, c1 = pb.VARIANTS['C0_100m_baseline'], pb.VARIANTS['C1_100m_nocompile']
    skip = ('compile', 'notes')
    assert {k: v for k, v in c0.items() if k not in skip} == \
           {k: v for k, v in c1.items() if k not in skip}


def test_b_series_50m_configs_unchanged():
    """保护项：B0–B9 的 50M 配置（batch/precision/fused/compile/workers）一律不动。"""
    for name, expect in B_SERIES_50M.items():
        cfg = pb.VARIANTS[name]
        got = (cfg['batch'], cfg['precision'], cfg['fused'], cfg['compile'], cfg['workers'])
        assert got == expect, name
        assert 'profile' not in cfg, name                     # B 系列全部走 50m 档


def test_build_train_cmd_has_no_duplicate_flags():
    """重复 --flag 会被 argparse 静默覆盖 → 制造"空转变体"（历史 B3_dl_w* 即如此）。"""
    for name, cfg in pb.VARIANTS.items():
        prof = cfg.get('profile') or '50m'
        cmd = pb.build_train_cmd(cfg, prof, 900, '/out', '/log')
        assert pb.duplicate_flags(cmd) == [], name
        assert cmd[cmd.index('--num-workers') + 1] == str(cfg['workers']), name
        assert cmd[cmd.index('--log-every') + 1] == str(cfg.get('log_every', pb.LOG_EVERY)), name
    assert '--num-workers' not in pb.BASE_ARGS.split()        # 别放回 BASE_ARGS


def test_build_train_cmd_uses_profile_paths_and_arch():
    c0 = pb.build_train_cmd(pb.VARIANTS['C0_100m_baseline'], '100m', 900, '/o', '/l')
    assert c0[c0.index('--vocab-size') + 1] == '8192'
    assert c0[c0.index('--n-embd') + 1] == '704'
    assert c0[c0.index('--train-txt') + 1] == pb.MODEL_PROFILES['100m']['train_txt']
    assert '--compile' in c0
    c1 = pb.build_train_cmd(pb.VARIANTS['C1_100m_nocompile'], '100m', 900, '/o', '/l')
    assert '--compile' not in c1
    assert c1[c1.index('--cache-dir') + 1] == pb.MODEL_PROFILES['100m']['cache_dir']
    b6 = pb.build_train_cmd(pb.VARIANTS['B6_compile'], '50m', 900, '/o', '/l')
    assert b6[b6.index('--vocab-size') + 1] == '6144'
    assert b6[b6.index('--train-txt') + 1] == pb.MODEL_PROFILES['50m']['train_txt']
    assert b6[b6.index('--block-size') + 1] == '512'
    assert f'{pb.TRAIN_PY}' in b6


# ================================================================ CLI
def test_cli_help_prints():
    """help 文本里的裸 % 会让 argparse 炸 —— 必须能正常打印。"""
    script = os.path.join(ROOT, 'benchmark', 'perf_bench.py')
    r = subprocess.run([sys.executable, script, '--help'],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    for flag in ('--variant', '--all', '--repeat', '--model-profile', '--steps',
                 '--train-txt', '--val-txt', '--cache-dir', '--tok-src'):
        assert flag in r.stdout


def test_cli_defaults():
    a = pb.build_parser().parse_args([])
    assert a.repeat == 3                                     # 默认重复 3 次
    assert a.model_profile is None                           # 缺省按变体绑定
    assert a.steps == pb.STEPS_TOTAL
    assert (a.train_txt, a.val_txt, a.cache_dir, a.tok_src) == (None, None, None, None)
    assert pb.build_parser().parse_args(['--model-profile', '100m']).model_profile == '100m'
    assert pb.build_parser().parse_args(['--variant', 'C0_100m_baseline']).variant == 'C0_100m_baseline'
