"""画训练 loss 曲线：支持从 train.log 或 CSV 读取数据。"""
import re
import csv
import matplotlib.pyplot as plt


def from_log(log_path):
    """从 train.log 提取（正则匹配 'epoch X 平均 loss: Y'）。"""
    with open(log_path, encoding='utf-8') as f:
        lines = f.readlines()
    epochs, losses = [], []
    for line in lines:
        m = re.search(r'epoch (\d+) 平均 loss: ([\d.]+)', line)
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return epochs, losses


def from_csv(csv_path):
    """从 loss_history CSV 提取。"""
    epochs, losses = [], []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            losses.append(float(row['train_loss']))
    return epochs, losses


def plot(epochs, losses, title, save_path):
    """画图并保存。"""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'o-', color='#4C72B0', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"已保存: {save_path}")
    print(f"起始 loss: {losses[0]:.3f} → 最终 loss: {losses[-1]:.3f}")
    plt.close()


if __name__ == '__main__':
    # 画基线（A组）
    e, l = from_log('log/train.log')
    plot(e, l, 'MiniGPT-Chinese Training Loss (Baseline A: 5.5M/505万)', 'result/training_loss_curve.png')

    # 画 B 组（小模型）
    e, l = from_csv('result/loss_history_b.csv')
    plot(e, l, 'Experiment B: Small Model (1.2M/200万)', 'result/training_loss_curve_b.png')
