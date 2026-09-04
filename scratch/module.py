import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch
from torch import nn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])



if __name__ == '__main__':
    train_data = torchvision.datasets.MNIST(
        root="./scratch/data",  # 下载到哪
        train=True,  # 训练集
        download=True,  # 没下载就自动下
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    test_data = torchvision.datasets.MNIST(
        root="./scratch/data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)
    images, labels = next(iter(train_loader))
    print("一个 batch 的形状:", images.shape)  # [64, 1, 28, 28]
    print("对应标签:", labels[:10])

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1=nn.Conv2d(1,16,3,padding=1)
            self.conv2=nn.Conv2d(16,32,3,padding=1)
            self.pool = nn.MaxPool2d(2,2)
            self.fc1 = nn.Linear(32*7*7,128)
            self.fc2 = nn.Linear(128,10)
        def forward(self, x):
            x=self.pool(F.relu(self.conv1(x)))
            x=self.pool(F.relu(self.conv2(x)))
            x=torch.flatten(x,1)
            x=self.fc1(x)
            x=self.fc2(x)
            return x
    net = Net()
    # print("网络参数量:", sum(p.numel() for p in net.parameters()))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #net.train()
    # for epoch in range(3):
    #     running_loss = 0.0
    #     for images, labels in train_loader:
    #         optimizer.zero_grad()
    #         outputs = net(images)
    #         loss = loss_fn(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #     print(f"epoch {epoch + 1}, 平均损失: {running_loss / len(train_loader):.4f}")
    PATH = 'result/minst_net.pt'
    # torch.save(net.state_dict(), PATH)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct+=(predicted == labels).sum().item()
        print(f"测试集准确率: {100 * correct / total:.2f}%")

