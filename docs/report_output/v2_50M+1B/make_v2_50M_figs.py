# -*- coding: utf-8 -*-
"""
MiniGPT-Chinese v2 50M / 1B WebNovel — 单轮训练总结 核心图（仿 35M E1 五图风格）
数据来源：log/50M参数+998Mtokens/{val_history,step_history}_train_webnovel_v2.csv
  ⚠️ CSV 混有 35M(E1/E2) 行 —— 50M 行在文件末尾（追加顺序），val 取最后 8 行；
     step_history 的 lr 用 dict 按 step 覆盖（末尾 50M 覆盖 ≤30454 的 step）。
输出 5 张图到 docs/report_output/v2_50M+1B/：
  fig1_val_loss.png   Val loss vs step/tokens_seen（单轮，无 E1/E2 分段）
  fig2_train_vs_val.png  Train_eval vs Val
  fig3_gap.png   Generalization gap
  fig4_lr.png    LR（cosine 8e-4 → 5e-5, log 轴）
  fig5_improvement.png  Val improvement per 1,000 steps by interval
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"D:\WBB_Python\pytorch\log\50M参数+998Mtokens"
OUT = r"D:\WBB_Python\pytorch\docs\report_output\v2_50M+1B"
os.makedirs(OUT, exist_ok=True)

C = "#4C72B0"        # 主线色
C2 = "#55A868"       # train_eval 色

# ---------- val_history：50M 行 = 文件末尾 8 行（step 0..30454） ----------
all_val = []
with open(BASE + r"\val_history_train_webnovel_v2.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        all_val.append({
            "step": int(r["step"]),
            "val": float(r["val"]),
            "train_eval": float(r["train_eval"]),
            "gap": float(r["gap"]),
            "lr": float(r["lr"]),
            "wall": r["wall_time"],
        })
v50 = all_val[-8:]                     # 50M 的 8 个验证点
v50 = [r for r in v50 if r["step"] > 0]  # 去掉 step 0 初始随机验证(368, 假信号)
FINAL_STEP = max(r["step"] for r in v50)
FINAL_VAL = min(r["val"] for r in v50)
assert FINAL_STEP == 30454, FINAL_STEP
print(f"50M val points: {len(v50)}（step {v50[0]['step']}..{FINAL_STEP}），final val = {FINAL_VAL}")

# ---------- step_history：lr（dict 按 step 覆盖 → ≤30454 即 50M 值） ----------
lr_map = {}
with open(BASE + r"\step_history_train_webnovel_v2.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        lr_map[int(r["step"])] = float(r["lr"])
steps_lr = sorted((s, lr) for s, lr in lr_map.items() if s <= FINAL_STEP)
x_lr = [s for s, _ in steps_lr]
y_lr = [v for _, v in steps_lr]

# ---------- 通用 ----------
def new_ax(figsize=(10.5, 5.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    return fig, ax

def add_token_axis(ax):
    """顶部 tokens_seen 轴（tokens/step = 32768）"""
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = [0, 5000, 10000, 15000, 20000, 25000, 30000, FINAL_STEP]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{t * 32768 / 1e9:.2f}" for t in ticks])
    ax2.set_xlabel("tokens_seen (B)")
    return ax2

xs = [r["step"] for r in v50]
ys = [r["val"] for r in v50]
yt = [r["train_eval"] for r in v50]
yg = [r["gap"] for r in v50]

# ========== Figure 1 · Val loss ==========
fig, ax = new_ax()
ax.plot(xs, ys, "-o", color=C, lw=2, ms=6, label="val loss", zorder=3)
ax.annotate(f"{FINAL_VAL}", xy=(FINAL_STEP, FINAL_VAL), xytext=(FINAL_STEP - 6500, FINAL_VAL + 0.30),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9), fontsize=10, color="#333333")
ax.set_xlabel("global step")
ax.set_ylabel("val loss (nats)")
ax.set_title("Figure 1 · Validation loss trajectory — v2 50M (12L/576d/9H) single epoch 1B", fontsize=11.5)
add_token_axis(ax)
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(OUT + r"\v2_50M_fig1_val_loss.png", bbox_inches="tight")
plt.close(fig)

# ========== Figure 2 · Train_eval vs Val ==========
fig, ax = new_ax()
ax.plot(xs, yt, "-s", color=C2, lw=2, ms=6, label="train_eval (无 dropout)", zorder=3)
ax.plot(xs, ys, "--o", color=C, lw=1.8, ms=5, label="val", zorder=3)
ax.annotate(f"train_eval {v50[-1]['train_eval']:.4f}", xy=(FINAL_STEP, v50[-1]["train_eval"]),
            xytext=(8000, 3.62), arrowprops=dict(arrowstyle="->", color=C2, lw=0.9), fontsize=9.5, color=C2)
ax.annotate(f"val {FINAL_VAL}", xy=(FINAL_STEP, FINAL_VAL),
            xytext=(8000, 3.74), arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9), fontsize=9.5, color="#333333")
ax.set_xlabel("global step")
ax.set_ylabel("loss (nats, eval 模式无 dropout)")
ax.set_title("Figure 2 · Train-evaluation vs validation loss — v2 50M single epoch", fontsize=11.5)
ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT + r"\v2_50M_fig2_train_vs_val.png", bbox_inches="tight")
plt.close(fig)

# ========== Figure 3 · Generalization gap ==========
fig, ax = new_ax()
ax.plot(xs, yg, "-o", color=C, lw=2, ms=6, label="gap = val − train_eval", zorder=3)
g0, g1 = v50[0]["gap"], v50[-1]["gap"]
ax.annotate(f"开始 {g0:.4f}", xy=(xs[0], g0), xytext=(xs[0] + 700, g0 - 0.006),
            fontsize=9, color="#333333")
ax.annotate(f"结束 {g1:.4f}", xy=(FINAL_STEP, g1), xytext=(FINAL_STEP - 9500, g1 + 0.0025),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9), fontsize=9.5, color="#333333")
ax.set_xlabel("global step")
ax.set_ylabel("gap = val − train_eval (nats)")
ax.set_title("Figure 3 · Generalization gap evolution — v2 50M single epoch", fontsize=11.5)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(OUT + r"\v2_50M_fig3_gap.png", bbox_inches="tight")
plt.close(fig)

# ========== Figure 4 · Learning Rate ==========
fig, ax = new_ax()
ax.plot(x_lr, y_lr, color="#333333", lw=1.1, label="step lr", zorder=2)
ax.set_yscale("log")
ax.set_ylim(1e-6, 2e-3)
ax.set_xlabel("global step")
ax.set_ylabel("learning rate (log scale)")
ax.set_title("Figure 4 · Learning Rate — cosine 8e-4 → 5e-5 (warmup 1000)", fontsize=11.5)
ax.annotate("warmup 1000 步 → 峰值 8e-4", xy=(1000, 8e-4), xytext=(4500, 6.5e-4),
            arrowprops=dict(arrowstyle="->", color=C, lw=1.0), fontsize=10, color=C)
ax.annotate("cosine → 5e-5", xy=(FINAL_STEP, 5e-5), xytext=(FINAL_STEP - 7000, 2.2e-4),
            arrowprops=dict(arrowstyle="->", color=C, lw=1.0), fontsize=10, color=C)
ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT + r"\v2_50M_fig4_lr.png", bbox_inches="tight")
plt.close(fig)

# ========== Figure 5 · Val improvement per 1,000 steps by interval ==========
intervals = []   # (label, Δval / 千步)
for i in range(len(xs) - 1):
    s0, s1, v0, v1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    length = s1 - s0
    intervals.append((f"{s0//1000}k-{s1//1000 if s1 % 1000 else s1//1000}k"
                      if s1 < 10000 else f"{s0//1000}k-{s1//1000}k"
                      if s1 % 1000 else f"{s0//1000}k-{s1//1000}k",
                      (v0 - v1) / length * 1000))
labels = []
for i, (s0, s1) in enumerate(zip(xs, xs[1:])):
    if s0 >= 10000 and s1 < 10000:
        pass
    if s1 % 1000 == 0:
        labels.append(f"{s0//1000}k-{s1//1000}k")
    else:
        labels.append(f"{s0//1000}k-{s1//1000:.1f}k" if s0 % 1000 == 0 else f"{s0//1000}k-{s1//1000}k*")
imps = [(v0 - v1) / (s1 - s0) * 1000 for s0, s1, v0, v1 in zip(xs, xs[1:], ys, ys[1:])]
fig, ax = new_ax()
bars = ax.bar(range(len(imps)), imps, color=C, alpha=0.85, width=0.6)
for i, (b, imp) in enumerate(zip(bars, imps)):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
            f"{imp:.4f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(imps)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlabel("interval (global step)")
ax.set_ylabel("val improvement per 1,000 steps (nats)")
ax.set_title("Figure 5 · Val improvement per 1,000 steps by interval（末区间为 partial，已按步长标准化）",
             fontsize=11)
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
fig.tight_layout()
fig.savefig(OUT + r"\v2_50M_fig5_improvement.png", bbox_inches="tight")
plt.close(fig)

print("figures saved to:", OUT)
print("improvement/1k:", [f"{x:.4f}" for x in imps])
