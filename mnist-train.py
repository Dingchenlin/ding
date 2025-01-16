import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from CNN import CNN
import torchvision
import torchvision.transforms as transforms

class Dataset(Dataset):
    def __init__(self, data):
        self.len = len(data)
        self.x_data = torch.from_numpy(np.array(list(map(lambda x: x[0], data)), dtype=np.float32))
        self.y_data = torch.from_numpy(np.array(list(map(lambda x: x[-1], data)))).squeeze().long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# 加载训练数据集
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
train_dataset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)

# 加载测试数据集
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
test_dataset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)


# 将数据集处理成Dataloder
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
testloader = DataLoader(test_dataset, shuffle=True, batch_size=32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型和损失函数，并将模型和损失函数移到GPU上
model = CNN(in_channels=1, classes=10)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
min_loss = np.inf

for i in range(30):
    correct = 0
    total = 0
    loss1 = 0
    for data, label in trainloader:
        train_data_value, train_data_label = data.to(device), label.to(device)
        train_data_label_pred = model(train_data_value)
        loss = criterion(train_data_label_pred, train_data_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(train_data_label_pred, 1)
        total += train_data_label.size(0)
        correct += (predicted == train_data_label).sum().item()
        loss1 = loss1 + loss.item()

    accuracy = 100 * correct / total
    loss1 = loss1 / len(trainloader)
    if loss1 < min_loss:
        min_loss = loss1
        torch.save(model, 'save1.pt')
    print(f'Epoch {i + 1}, Accuracy: {accuracy:.2f}%, Loss: {loss.item()}')
