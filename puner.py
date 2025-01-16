import torch
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from CNN import CNN
import torchvision
import torchvision.transforms as transforms
def prune_weights(model, pruning_rate):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 计算绝对值
            weight_abs = param.abs()
            # 计算剪枝阈值
            threshold = torch.quantile(weight_abs, pruning_rate)
            # 剪枝权重
            param.data[weight_abs < threshold] = 0
    return model

# 加载测试数据集
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
test_dataset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
# 将数据集处理成Dataloder
testloader = DataLoader(test_dataset, shuffle=True, batch_size=32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('save1.pt')
model = model.to('cuda:0')

for i in range(12):
    r = 0.25 + i * 0.05
    model = prune_weights(model, pruning_rate=r)
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for testdata in testloader:
            test_data_value, test_data_label = testdata
            test_data_value, test_data_label = test_data_value.to(device), test_data_label.to(device)
            test_data_label_pred = model(test_data_value)
            _, test_predicted = torch.max(test_data_label_pred.data, dim=1)
            test_total += test_data_label.size(0)
            test_correct += (test_predicted == test_data_label).sum().item()
    test_acc = round(100 * test_correct / test_total, 3)
    print(f'pruning_rate = {r:.2f}, Test Accuracy: {test_acc:.2f}%')
