# -*- coding: utf-8 -*-
"""汇总 performance_results.csv 关键列（本地查看用）。"""
import csv

with open('benchmark/performance_results.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

hdr = f"{'variant':24s} {'tok/s':>8s} {'step/s':>7s} {'loss_s':>8s} {'loss_e':>8s} {'vram':>6s} {'nan':>3s} {'notes'}"
print(hdr)
for r in rows:
    print(f"{r['variant']:24s} {r['tokens_per_sec']:>8s} {r['step_per_sec']:>7s} "
          f"{r['loss_start']:>8s} {r['loss_end']:>8s} {r['peak_vram_mb']:>6s} "
          f"{r['nan_count']:>3s}  {r['notes']}")
