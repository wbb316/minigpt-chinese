"""从 train.log 提取 loss 数据并画训练曲线。"""
import re
import matplotlib.pyplot as plt

# 读取 train.log 提取每个 epoch 的平均 loss
with open('log/train.log', encoding='utf-8') as f:
    lines = f.readlines()

epochs = []
losses = []
for line in lines:
    m = re.search(r'epoch (\d+) 平均 loss: ([\d.]+)', line)
    if m:
        epochs.append(int(m.group(1)))
        losses.append(float(m.group(2)))

print(f"提取到 {len(epochs)} 个 epoch 的 loss 数据")

# 画图
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, 'o-', color='#4C72B0', linewidth=2, markersize=6)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Loss', fontsize=12)
plt.title('MiniGPT-Chinese Training Loss (5.5M params, 505万字符语料)', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存到 result 目录
plt.savefig('result/training_loss_curve.png', dpi=150)
print("已保存: result/training_loss_curve.png")

# 也显示一下数据
print(f"起始 loss: {losses[0]:.3f} → 最终 loss: {losses[-1]:.3f}")
print(f"共下降: {losses[0]-losses[-1]:.3f}")
