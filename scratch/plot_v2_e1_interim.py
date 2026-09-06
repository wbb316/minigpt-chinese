# -*- coding: utf-8 -*-
"""
MiniGPT-Chinese v2 35M/1B WebNovel — Epoch 1 Interim Report 绘图脚本
数据来源：val_history_train_webnovel_v2.csv / step_history_train_webnovel_v2.csv
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"D:\WBB_Python\pytorch\log\35M参数+998Mtokens"
OUT = r"D:\WBB_Python\pytorch\docs\report_output"
os.makedirs(OUT, exist_ok=True)

# ---------- 读数据 ----------
val = pd.read_csv(os.path.join(BASE, "val_history_train_webnovel_v2.csv"))
step = pd.read_csv(os.path.join(BASE, "step_history_train_webnovel_v2.csv"))

# val_history 里没有 tokens_seen 列，按 step * 32768 推算（batch=64, block=512）
val = val.copy()
val["tokens_seen"] = val["step"] * 32768
val = val.sort_values("step").reset_index(drop=True)

# 验证最后一步的 tokens_seen 是否与日志一致（998,211,584）
print("tokens_seen @ final step:", val["tokens_seen"].iloc[-1])
print("expected: 998,211,584")

# ---------- 统一风格 ----------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
C_VAL = "#C0392B"      # val
C_TRAIN = "#2471A3"    # train_eval
C_GAP = "#D35400"      # gap
C_LR = "#1E8449"       # lr
C_BAR = "#5D6D7E"      # 改善柱

XTICKS = [0, 2e8, 4e8, 6e8, 8e8, 1e9]

def xfmt(v, pos):
    return f"{v/1e8:.1f}"

# 末两点（30000 / 30463）x 极近，标注上下错位避免重叠
def annotate_series(ax, xs, ys, fmt, color, above=True):
    n = len(xs)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == n - 2:  # 倒数第二点与末点错位
            dy = 11 if above else -16
        elif i == n - 1:
            dy = -16 if above else 11
        else:
            dy = 11 if above else -16
        ax.annotate(f"{y:{fmt}}", (x, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=9, color=color)

# ---------- Figure 1: Val loss vs tokens_seen ----------
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.plot(val["tokens_seen"], val["val"], marker="o", ms=7, lw=2, color=C_VAL,
        label="Validation loss")
annotate_series(ax, val["tokens_seen"], val["val"], ".4f", C_VAL, above=True)
ax.set_xlabel("Tokens seen (×10⁸)")
ax.set_ylabel("Val loss (nats)")
ax.set_title("Figure 1 · Validation loss vs tokens seen (Epoch 1)")
ax.set_xticks(XTICKS)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xfmt))
ax.set_ylim(val["val"].min() - 0.12, val["val"].max() + 0.12)
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v2_35M_E1_fig1_val_loss.png"), dpi=160)
plt.close(fig)

# ---------- Figure 2: Train_eval vs Val loss ----------
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.plot(val["tokens_seen"], val["val"], marker="o", ms=7, lw=2, color=C_VAL, label="Val loss")
ax.plot(val["tokens_seen"], val["train_eval"], marker="s", ms=7, lw=2, color=C_TRAIN,
        label="Train eval (no dropout)")
annotate_series(ax, val["tokens_seen"], val["train_eval"], ".4f", C_TRAIN, above=False)
ax.set_xlabel("Tokens seen (×10⁸)")
ax.set_ylabel("Loss (nats)")
ax.set_title("Figure 2 · Val loss vs train_eval (Epoch 1)")
ax.set_xticks(XTICKS)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xfmt))
ax.set_ylim(val["train_eval"].min() - 0.16, val["val"].max() + 0.12)
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v2_35M_E1_fig2_train_vs_val.png"), dpi=160)
plt.close(fig)

# ---------- Figure 3: Generalization gap ----------
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.plot(val["tokens_seen"], val["gap"], marker="o", ms=7, lw=2, color=C_GAP,
        label="Gap = val − train_eval")
annotate_series(ax, val["tokens_seen"], val["gap"], ".4f", C_GAP, above=True)
ax.set_xlabel("Tokens seen (×10⁸)")
ax.set_ylabel("Gap (nats)")
ax.set_title("Figure 3 · Generalization gap vs tokens seen (Epoch 1)")
ax.set_xticks(XTICKS)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xfmt))
ax.set_ylim(0.004, 0.038)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v2_35M_E1_fig3_gap.png"), dpi=160)
plt.close(fig)

# ---------- Figure 4: Learning Rate (per-step 实际曲线) ----------
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.plot(step["tokens_seen"], step["lr"], lw=1.6, color=C_LR, label="LR (per step, actual)")
ax.plot(val["tokens_seen"], val["lr"], marker="o", ms=7, ls="none", color=C_VAL,
        label="LR @ validation")
ax.annotate("max_lr 8.0e-4", (0.03e9, 8.2e-4), fontsize=9, color=C_LR)
ax.annotate("min_lr 5.0e-5", (9.85e8, 1.2e-4), fontsize=9, color=C_LR)
ax.set_xlabel("Tokens seen (×10⁸)")
ax.set_ylabel("Learning rate")
ax.set_title("Figure 4 · Learning rate schedule vs tokens seen (Epoch 1)")
ax.set_xticks(XTICKS)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xfmt))
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v2_35M_E1_fig4_lr.png"), dpi=160)
plt.close(fig)

# ---------- Figure 5: Val improvement per 1,000 steps ----------
d = val["val"].to_numpy()
st = val["step"].to_numpy()
impr = -(d[1:] - d[:-1])
step_len = st[1:] - st[:-1]                      # 区间步长
impr_per_1k = impr / step_len * 1000.0           # 标准化：每 1000 步的改善量
lbl = [f"{st[i]}→{st[i+1]}" for i in range(len(st) - 1)]
lbl[-1] = "30000→30463\n(463 steps, partial)"
fig, ax = plt.subplots(figsize=(8.6, 5.2))
bars = ax.bar(range(len(impr_per_1k)), impr_per_1k, color=C_BAR, width=0.62)
for i, v in enumerate(impr_per_1k):
    ax.text(i, v + 0.0015, f"{v:.4f}", ha="center", fontsize=10)
ax.set_xticks(range(len(impr_per_1k)))
ax.set_xticklabels(lbl, fontsize=8.5)
ax.set_ylabel("Δ Val loss per 1,000 steps (nats)")
ax.set_title("Figure 5 · Val improvement per 1,000 steps by interval (Epoch 1)")
ax.set_ylim(0, max(impr_per_1k) * 1.16)
ax.text(0.99, 0.95, "First 5 intervals: 5,000 steps each\nLast interval: 463 steps (partial)\nNormalized per 1,000 steps for comparability",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#555555")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v2_35M_E1_fig5_improvement.png"), dpi=160)
plt.close(fig)

# ---------- 校验输出 ----------
print("\nFigures saved to:", OUT)
for f in sorted(os.listdir(OUT)):
    if f.startswith("v2_35M_E1"):
        print(" -", f)

# ---------- 关键数值核对 ----------
print("\n=== 校验 ===")
print("improvements:", [f"{v:.4f}" for v in impr])
print("final gap:", f"{val['gap'].iloc[-1]:.4f}")
print("best val:", val["val"].iloc[-1], "| is_best 全部:", val["is_best"].tolist(),
      "| no_improve 全部:", val["no_improve"].tolist())
print("step rows:", len(step), "| 首步 lr:", step["lr"].iloc[0], "| 末步 lr:", step["lr"].iloc[-1])
print("tokens/step:", 32768, "| 5.72 step/s ->", round(32768*5.72/1000), "k tokens/s")
