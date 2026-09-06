# -*- coding: utf-8 -*-
"""
MiniGPT-Chinese v2 35M / 1B WebNovel — Epoch 1+2 阶段训练总结 核心图
数据来源：log/35M参数+998Mtokens/{val_history,step_history}_train_webnovel_v2.csv（全部实际日志）
输出 4 张图：
  fig1_val_loss.png  E1+E2 Val loss vs step/tokens_seen（E1/E2 边界竖线）
  fig2_train_vs_val.png  Train_eval vs Val
  fig3_gap.png  完整 Generalization gap 曲线
  fig4_lr.png   LR + Val loss（E1 cosine decay → E2 const 5e-5）
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"D:\WBB_Python\pytorch\log\35M参数+998Mtokens"
OUT = r"D:\WBB_Python\pytorch\docs\report_output\v2_35M+1B\二阶段"

# ---------- 读取 val_history（权威验证点） ----------
val_rows = []
with open(BASE + r"\val_history_train_webnovel_v2.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        val_rows.append({
            "step": int(r["step"]),
            "val": float(r["val"]),
            "train_eval": float(r["train_eval"]),
            "gap": float(r["gap"]),
            "lr": float(r["lr"]),
        })
E1_END = 30463  # E1 终点 / E2 起点边界

# ---------- 读取 step_history（LR 曲线，去重验证点重复行） ----------
lr_map = {}
with open(BASE + r"\step_history_train_webnovel_v2.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        lr_map[int(r["step"])] = float(r["lr"])
steps_lr = sorted(lr_map.items())          # (step, lr) 全 60,915 步
x_lr = [s for s, _ in steps_lr]
y_lr = [v for _, v in steps_lr]

BOUND = 30463
E1_COLOR, E2_COLOR = "#4C72B0", "#DD8452"

# ---------- 通用画布与轴 ----------
def new_ax(figsize=(10.5, 5.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    return fig, ax

def draw_boundary(ax):
    ax.axvline(BOUND, color="grey", linestyle=":", linewidth=1.4, zorder=1)
    ax.annotate("E1 END / E2 START", xy=(BOUND, ax.get_ylim()[1]),
                xytext=(BOUND - 8, ax.get_ylim()[1]),
                ha="right", va="top", fontsize=9.5, color="#555555")

def draw_epoch_shade(ax, ymin, ymax):
    ax.axvspan(0, BOUND, color=E1_COLOR, alpha=0.06, zorder=0)
    ax.axvspan(BOUND, 62000, color=E2_COLOR, alpha=0.06, zorder=0)
    ax.text(1000, ymax - 0.015, "Epoch 1", fontsize=11, color=E1_COLOR, fontweight="bold", va="top")
    ax.text(61000, ymax - 0.015, "Epoch 2", fontsize=11, color=E2_COLOR, fontweight="bold", va="top", ha="right")

# ========== Figure 1 · Val loss vs step / tokens_seen ==========
fig, ax = new_ax()
xs = [r["step"] for r in val_rows]
ys = [r["val"] for r in val_rows]
ax.plot(xs[:7], ys[:7], "-o", color=E1_COLOR, label="Epoch 1 val", lw=2, ms=6, zorder=3)
ax.plot(xs[6:], ys[6:], "-o", color=E2_COLOR, label="Epoch 2 val", lw=2, ms=6, zorder=3)
# 端点标注
ax.annotate("3.7076", xy=(30463, 3.7076), xytext=(28200, 3.90),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9), fontsize=9.5, color="#333333")
ax.annotate("3.6372", xy=(60915, 3.6372), xytext=(50000, 3.78),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9), fontsize=9.5, color="#333333")
draw_epoch_shade(ax, ax.get_ylim()[0], ax.get_ylim()[1])
draw_boundary(ax)
ax.set_xlabel("global step")
ax.set_ylabel("val loss (nats)")
ax.set_title("Figure 1 · Validation loss trajectory during Epoch 1 initialization and Epoch 2 continuation", fontsize=11.5)
# 顶部 tokens_seen 轴
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
def step_to_tok(x): return x * 32768
ax2.set_xlim(ax.get_xlim())
ticks_step = [0, 15232, 30463, 45695, 60915]
ax2.set_xticks(ticks_step)
ax2.set_xticklabels([f"{step_to_tok(t)/1e9:.2f}" for t in ticks_step])
ax2.set_xlabel("tokens_seen (B)")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(OUT + r"\v2_35M_E1E2_fig1_val_loss.png", bbox_inches="tight")
plt.close(fig)

# ========== Figure 2 · Train_eval vs Val ==========
fig, ax = new_ax()
xt = [r["step"] for r in val_rows]
yt = [r["train_eval"] for r in val_rows]
ax.plot(xt[:7], yt[:7], "-s", color="#55A868", lw=2, ms=6, label="Epoch 1 train_eval", zorder=3)
ax.plot(xt[6:], yt[6:], "-s", color="#C44E52", lw=2, ms=6, label="Epoch 2 train_eval", zorder=3)
ax.plot(xs[:7], ys[:7], "--o", color=E1_COLOR, lw=1.8, ms=5, label="Epoch 1 val", zorder=3)
ax.plot(xs[6:], ys[6:], "--o", color=E2_COLOR, lw=1.8, ms=5, label="Epoch 2 val", zorder=3)
ax.annotate("train_eval 3.6063", xy=(60915, 3.6062694501876833), xytext=(45500, 3.535),
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.9), fontsize=9.5, color="#C44E52")
ax.annotate("val 3.6372", xy=(60915, 3.6372), xytext=(45500, 3.665),
            arrowprops=dict(arrowstyle="->", color="#DD8452", lw=0.9), fontsize=9.5, color="#DD8452")
draw_epoch_shade(ax, ax.get_ylim()[0], ax.get_ylim()[1])
draw_boundary(ax)
ax.set_xlabel("global step")
ax.set_ylabel("loss (nats, eval 模式无 dropout)")
ax.set_title("Figure 2 · Train-evaluation and validation loss during continuation training", fontsize=11.5)
ax.legend(loc="upper right", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT + r"\v2_35M_E1E2_fig2_train_vs_val.png", bbox_inches="tight")
plt.close(fig)

# ========== Figure 3 · Generalization gap ==========
fig, ax = new_ax()
xg = [r["step"] for r in val_rows]
yg = [r["gap"] for r in val_rows]
ax.plot(xg[:7], yg[:7], "-o", color=E1_COLOR, lw=2, ms=6, label="Epoch 1 gap", zorder=3)
ax.plot(xg[6:], yg[6:], "-o", color=E2_COLOR, lw=2, ms=6, label="Epoch 2 gap", zorder=3)
ax.axhline(0.031, color="#999999", linestyle="--", lw=1.0)
ax.text(62000, 0.031, "0.031 参考线", fontsize=8.5, color="#999999", ha="right", va="bottom")
ax.annotate("E1 结束 0.0289", xy=(30463, 0.028910777568817103), xytext=(18000, 0.0365),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9), fontsize=9.5, color="#333333")
ax.annotate("E2 早期收窄至 0.0269", xy=(35000, 0.02687199592590339), xytext=(36000, 0.0235),
            arrowprops=dict(arrowstyle="->", color="#DD8452", lw=0.9), fontsize=9.5, color="#DD8452")
ax.annotate("E2 结束 0.03095", xy=(60915, 0.030953211784362722), xytext=(50500, 0.0368),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9), fontsize=9.5, color="#333333")
draw_epoch_shade(ax, ax.get_ylim()[0], ax.get_ylim()[1])
draw_boundary(ax)
ax.set_xlabel("global step")
ax.set_ylabel("gap = val − train_eval (nats)")
ax.set_title("Figure 3 · Generalization gap evolution during two-pass training", fontsize=11.5)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(OUT + r"\v2_35M_E1E2_fig3_gap.png", bbox_inches="tight")
plt.close(fig)

# ========== Figure 4 · Learning Rate + Val loss ==========
fig, ax = new_ax(figsize=(10.5, 5.6))
ax.plot(x_lr, y_lr, color="#333333", lw=1.1, label="step lr（全 60,915 步）", zorder=2)
ax.set_yscale("log")
ax.set_ylim(1e-6, 2e-3)
ax.set_xlabel("global step")
ax.set_ylabel("learning rate (log scale)")
ax.set_title("Figure 4 · Learning Rate + Val loss（E1 cosine decay → E2 const 5e-5）", fontsize=11.5)
ax.annotate("E1: cosine 8e-4 → 5e-5", xy=(15000, 4.5e-4), xytext=(3500, 1.1e-3),
            arrowprops=dict(arrowstyle="->", color=E1_COLOR, lw=1.0), fontsize=10, color=E1_COLOR)
ax.annotate("E2: const 5e-5（恒温续训）", xy=(48000, 5e-5), xytext=(42000, 2.2e-4),
            arrowprops=dict(arrowstyle="->", color=E2_COLOR, lw=1.0), fontsize=10, color=E2_COLOR)
ax.axvline(BOUND, color="grey", linestyle=":", linewidth=1.4, zorder=1)
ax.annotate("E1 END / E2 START", xy=(BOUND, 1e-6), xytext=(BOUND - 9.5, 3.2e-6),
            ha="right", fontsize=9.5, color="#555555")
ax.axvspan(0, BOUND, color=E1_COLOR, alpha=0.06)
ax.axvspan(BOUND, 62000, color=E2_COLOR, alpha=0.06)
ax.text(1200, 1.25e-3, "Epoch 1", fontsize=11, color=E1_COLOR, fontweight="bold")
ax.text(60000, 1.25e-3, "Epoch 2", fontsize=11, color=E2_COLOR, fontweight="bold", ha="right")
# 右轴 val loss
axr = ax.twinx()
xsr = [r["step"] for r in val_rows]
ysr = [r["val"] for r in val_rows]
axr.plot(xsr[:7], ysr[:7], "--o", color=E1_COLOR, lw=1.8, ms=5, label="Epoch 1 val", zorder=3)
axr.plot(xsr[6:], ysr[6:], "-o", color=E2_COLOR, lw=1.8, ms=5, label="Epoch 2 val", zorder=3)
axr.set_ylabel("val loss (nats)")
axr.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = axr.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT + r"\v2_35M_E1E2_fig4_lr.png", bbox_inches="tight")
plt.close(fig)

print("figures saved to:", OUT)
