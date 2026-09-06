# -*- coding: utf-8 -*-
"""50M scaling 实验：从 step_history / val_history 提取曲线与统计（只读，不训练不改代码）。"""
import csv, json, math, os
from datetime import datetime

BASE = r"D:\WBB_Python\pytorch"
F50_STEP = os.path.join(BASE, r"log\50M参数+998Mtokens\step_history_train_webnovel_v2.csv")
F35_STEP = os.path.join(BASE, r"log\35M参数+998Mtokens\step_history_train_webnovel_v2.csv")
F50_VAL  = os.path.join(BASE, r"log\50M参数+998Mtokens\val_history_train_webnovel_v2.csv")
OUT      = os.path.join(BASE, r"docs\report_output\v2_50M+1B\chart_data.json")


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            d["step"] = int(d["step"])
            d["tokens_seen"] = int(d["tokens_seen"]) if d["tokens_seen"] else None
            d["train_loss"] = float(d["train_loss"]) if d["train_loss"] else None
            d["val_loss"] = float(d["val_loss"]) if d["val_loss"] else None
            d["lr"] = float(d["lr"]) if d["lr"] else None
            rows.append(d)
    return rows


def split_runs(rows):
    """在 step 回绕处切 run。"""
    runs, cur = [], []
    prev = -1
    for d in rows:
        if d["step"] < prev:
            runs.append(cur); cur = []
        cur.append(d); prev = d["step"]
    if cur:
        runs.append(cur)
    return runs


def ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def bin_series(run, bin_size=250, warm_cut=50):
    """每 bin_size 步一个平均点（train_loss 均值、lr 取末值），跳过 nan（AMP overflow 步）。"""
    bins = {}
    for d in run:
        b = max(0, (d["step"] - 1)) // bin_size
        bins.setdefault(b, []).append(d)
    out = []
    for b in sorted(bins):
        seg = bins[b]
        ls = [x["train_loss"] for x in seg if ok(x["train_loss"])]
        lrs = [x["lr"] for x in seg if ok(x["lr"])]
        if not ls:
            continue
        out.append({
            "step": seg[-1]["step"],
            "train_loss": round(sum(ls) / len(ls), 4),
            "lr": lrs[-1],
        })
    return out


def at_step(run, targets):
    idx = {d["step"]: d for d in run}
    res = {}
    for t in targets:
        if t in idx:
            res[t] = {"train_loss": idx[t]["train_loss"], "lr": idx[t]["lr"]}
    return res


def ema(run, alpha=0.01):
    e = None; last = None
    for d in run:
        if d["train_loss"] is None:
            continue
        e = d["train_loss"] if e is None else alpha * d["train_loss"] + (1 - alpha) * e
        last = e
    return last


# ---------- step history ----------
rows50 = load_rows(F50_STEP)
runs50 = split_runs(rows50)
run_info = [(len(r), r[0]["step"], r[-1]["step"], r[0]["time"], r[-1]["time"]) for r in runs50]
run50 = runs50[-1]                       # 最后一个 run = 50M
run35_in50file = runs50[0]               # 35M E1+E2 连续段

rows35 = load_rows(F35_STEP)
runs35 = split_runs(rows35)
# 35M 文件里 E1 为 run0（step 1..30463），E2 续接
run35e1 = runs35[0]

t0 = datetime.strptime(run50[0]["time"], "%Y-%m-%d %H:%M:%S")
t1 = datetime.strptime(run50[-1]["time"], "%Y-%m-%d %H:%M:%S")
dur = (t1 - t0).total_seconds()

probe = at_step(run50, [1, 2, 10, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 30454, 30463])

result = {
    "run_split_in_50m_file": run_info,
    "run50": {
        "n_rows": len(run50),
        "first_step": run50[0]["step"], "last_step": run50[-1]["step"],
        "t_start": run50[0]["time"], "t_end": run50[-1]["time"],
        "duration_sec": dur,
        "tokens_seen_last": run50[-1]["tokens_seen"],
        "train_min": round(min(d["train_loss"] for d in run50 if d["train_loss"] is not None), 4),
        "train_last": run50[-1]["train_loss"],
        "ema_last_alpha0.01": round(ema(run50), 4),
        "probe": probe,
        "bins250": bin_series(run50, 250),
    },
    "run35e1_bins250": bin_series(run35e1, 250),
}

# ---------- val history: 50M 段（step 回绕 0 之后） ----------
vrows = []
with open(F50_VAL, "r", encoding="utf-8") as f:
    for d in csv.DictReader(f):
        vrows.append(d)
# 找最后一个 step=0 的位置
zero_idx = [i for i, d in enumerate(vrows) if d["step"] == "0"]
cut = zero_idx[-1]
val50 = vrows[cut:]
val35 = vrows[:cut]
result["val50"] = [{
    "step": int(d["step"]), "where": d["where"],
    "val": float(d["val"]), "train_eval": float(d["train_eval"]),
    "gap": float(d["gap"]), "lr": float(d["lr"]),
    "is_best": int(d["is_best"]), "wall_time": d["wall_time"],
} for d in val50]
result["val35"] = [{
    "step": int(d["step"]), "where": d["where"],
    "val": float(d["val"]), "train_eval": float(d["train_eval"]),
    "gap": float(d["gap"]), "lr": float(d["lr"]),
} for d in val35]

# ---------- 关键对比计算 ----------
v35e1 = 3.7076; v35e2 = 3.6372; v50 = 3.6154
p35, p50 = 34676736, 51411840
calc = {
    "params_delta_abs": p50 - p35,
    "params_delta_pct": round((p50 - p35) / p35 * 100, 2),
    "param_gain_val": round(v35e1 - v50, 4),
    "repeat_gain_val": round(v35e1 - v35e2, 4),
    "param_vs_repeat_excess": round((v35e1 - v50) - (v35e1 - v35e2), 4),
    "param_gain_rel_pct": round((math.exp(v35e1) - math.exp(v50)) / math.exp(v35e1) * 100, 2),
    "repeat_gain_rel_pct": round((math.exp(v35e1) - math.exp(v35e2)) / math.exp(v35e1) * 100, 2),
    "ppl_35e1": round(math.exp(v35e1), 3),
    "ppl_35e2": round(math.exp(v35e2), 3),
    "ppl_50": round(math.exp(v50), 3),
    "gap_35e1": 0.028910777568817103,
    "gap_35e2": 0.030953211784362722,
    "gap_50": 0.03781000852584837,
    "gap_delta_50_vs_35e1": round(0.03781000852584837 - 0.028910777568817103, 4),
    "same_step_lead": {},
}
v35map = {d["step"]: d["val"] for d in result["val35"]}
for d in result["val50"]:
    if d["step"] in v35map:
        calc["same_step_lead"][d["step"]] = round(v35map[d["step"]] - d["val"], 4)
result["calc"] = calc

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=1)

# 控制台摘要
print("runs in 50m step file (n, first, last, t0, t1):")
for x in run_info:
    print(" ", x)
print("\n50M run:", result["run50"]["first_step"], "->", result["run50"]["last_step"],
      "| rows", len(run50), "| dur(s)", dur, "| tokens", run50[-1]["tokens_seen"])
print("train min", result["run50"]["train_min"], "last", result["run50"]["train_last"],
      "ema", result["run50"]["ema_last_alpha0.01"])
print("\nprobe:")
for k, v in probe.items():
    print("  step", k, v)
print("\nval50 points:", [(d["step"], d["val"]) for d in result["val50"]])
print("\ncalc:", json.dumps(calc, ensure_ascii=False, indent=1))
