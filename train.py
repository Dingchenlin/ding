import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from init_clients import clients_dataloader
import random
from model import CNN
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import math
from Add_noise import PM
#参数初始化
alpha = 0.0000055
lamd = 13.909
beta = 0.986
omega = 1
C = 0.3     #参与训练的客户端的比例
E = 10      #每个客户端本地训练的轮数
B = 16      #每个客户的BatchSize大小
R_t = 1.5   #服务器支付初始化
client_num = 20    #客户端数量

# Test_dataset class
class Dataset(Dataset):
    def __init__(self, data):
        self.len = len(data)
        self.x_data = torch.from_numpy(np.array(list(map(lambda x: x[0], data)), dtype=np.float32))
        self.y_data = torch.from_numpy(np.array(list(map(lambda x: x[-1], data)))).squeeze().long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# 设置随机种子
def set_random_seed(seed_value=100):
    random.seed(seed_value)         # Fixed Python built-in random generator
    np.random.seed(seed_value)      # Fixed NumPy random generator
    torch.manual_seed(seed_value)   # Fixed PyTorch random generator

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # If multiple GPUs are used

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(100)

# 初始化客户端dataloader 用10个客户端的数据生成Dataloader
client_dataloader_list = clients_dataloader(B)

# 加载测试数据集
data_test = np.load('Test_dataset/MNIST_test_data.npy')
label_test = np.load('Test_dataset/MNIST_test_label.npy')
test_dataset = [[i, j] for i, j in zip(data_test, label_test)]


# 将测试集数据处理成Dataloder
Test_dataset = Dataset(test_dataset)
testloader = DataLoader(Test_dataset, shuffle=True, batch_size=256)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型和损失函数，并将模型和损失函数移到GPU上
model = CNN(in_channels=1, classes=10)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)

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

class Client():
    def __init__(self, client_id, p_cm, epsilon, compression_ratio):
        self.client_id = client_id
        self.p_cm = p_cm
        self.epsilon = epsilon
        self.compression_ratio = compression_ratio
e = np.array([10.47256213, 10.51902829, 10.38731396, 10.86236131, 10.31080512, 10.72477872, 10.05770234, 10.5853274, 10.72781808, 10.38871681, 10.87705194, 10.70393851,
 10.3546538, 10.25110603, 10.95984959, 10.08561442, 10.28752421, 10.09938183, 10.21014868, 10.8285576 ])
clients = []
for i in range(client_num):
    #model = CNN()
    #client_dataloader = client_dataloader_list[i]
    p_cm = 20
    epsilon = e[i]
    compression_ratio = 0.5
    client = Client(i, p_cm, epsilon, compression_ratio)
    clients.append(client)

def client_update(client_id, E, model_parameter):
    '''
    客户端更新参数
    :param client_num: 选择到的客户端的编号
    :param E: 本地训练的epoch轮数
    :param model_parameter: 中心服务器发给客户端的模型参数
    :return: 在该客户端上训练好的模型参数
    '''

    dataloader = client_dataloader_list[client_id]     # 获取dataloder
    client_model = CNN().to(device)                     # 加载一个空模型
    client_model.load_state_dict(model_parameter)       # 加载中心服务器发的模型参数
    client_model.train()                                # 将模型变为训练模型

    # optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(client_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    for i in range(E):
        correct = 0
        total = 0

        for data, label in dataloader:
            train_data_value, train_data_label = data.to(device), label.to(device)
            train_data_label_pred = client_model(train_data_value)

            loss = criterion(train_data_label_pred, train_data_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(train_data_label_pred, 1)
            total += train_data_label.size(0)
            correct += (predicted == train_data_label).sum().item()

        accuracy = 100 * correct / total
        #print(f'Client:{client_num},Epoch {i+1}/{E}, Accuracy: {accuracy:.2f}%, Loss: {loss.item()}')
    s = 0
    for i in range(client_num):
        s = s + clients[i].p_cm / clients[i].epsilon
    m = ((client_num - 1) * R_t * ((client_num - 1) * clients[client_id].p_cm - clients[client_id].epsilon * s))
    n = omega * clients[client_id].epsilon * clients[client_id].epsilon * s
    clients[client_id].compression_ratio = m / n + 1
    #print(clients[client_id].compression_ratio)
    if clients[client_id].compression_ratio < 0.5:
        clients[client_id].compression_ratio = 0.5
    if clients[client_id].compression_ratio > 0.8:
        clients[client_id].compression_ratio = 0.8
    client_model = prune_weights(client_model, clients[client_id].compression_ratio)
    client_model = PM(client_model, clients[client_id].epsilon)

    # 返回客户端训练模型的模型参数
    return client_model.state_dict()

def func1(R):
    sum = 1
    for i in range(client_num):
        a = (clients[i].compression_ratio - 1) / R
        sum = sum + a * alpha * lamd * math.exp(lamd * clients[i].compression_ratio)
    return sum

def func2(R):
    sum = 0
    for i in range(client_num):
        a = (clients[i].compression_ratio - 1) / R
        sum = sum + a * a * alpha * lamd * lamd * math.exp(lamd * clients[i].compression_ratio)
    #print(f"a = {a}")
    return sum
def calcu(R):
    while abs(func1(R)) > 0.13:
        #print(f"func1(R) = {func1(R)}     func2(R) = {func2(R)}")
        R = R - func1(R) / func2(R)
    return R

def train(client_num, C, E):
    model.train()                               # 将服务器模型变为训练模型
    send_model_parameter = model.state_dict()   # 然后获取即将分发给各个客户端的模型的权重

    random_numbers = random.sample(range(client_num), int(client_num*C))        # 按照比例随机选择出用于本轮训练的模型的索引编号
    client_return_parameter_list = []                                           # 初始化一个列表用与保存每个客户端本地训练好之后返回的模型参数
    sum = 0
    for client_id in random_numbers:                                               # 遍历所有的选择到的客户端的编号(索引号)
        model_parameter = client_update(client_id, E=E, model_parameter=send_model_parameter)      # 返回每个客户端训练好的模型权重
        sum = sum + clients[client_id].compression_ratio
        client_return_parameter_list.append(model_parameter)                                    # 将每一个客户端的模型权重加载到列表中
    avg = sum / (client_num*C)
    # 先生成一个参数都为0的模型参数的参数字典，用于之后将客户端返回的模型参数都加到改字典上
    aggregated_model_parameter = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in send_model_parameter.items()}

    # 将所有客户端模型的权重都加权求和
    for client_param in client_return_parameter_list:
        for key in aggregated_model_parameter:
            aggregated_model_parameter[key] += client_param[key] * (1 / int(client_num*C))

    # 将求和好的权重加载给中心服务器模型，用于下一轮的发送
    model.load_state_dict(aggregated_model_parameter)
    return model, avg

def test():
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
    print(f'Test Accuracy: {test_acc:.2f}%')
    return test_acc

if __name__ == '__main__':
    test_accuracies = []
    avg_compression_ratio = []
    epochs = 30
    for i in range(epochs):
        R_t = calcu(R_t)
        #print(R_t)
        print(f"Epoch:{i}——", end='')
        _, avg = train(client_num, C, E)
        avg_compression_ratio.append(avg)
        test_acc = test()
        test_accuracies.append(test_acc)
        for j in range(client_num):
            print(clients[j].compression_ratio)

    # Plotting the test accuracy curve
    print(test_accuracies)
    print(avg_compression_ratio)
    np.save('test_accuracies.npy', test_accuracies)
    np.save('avg_compression_ratio.npy', avg_compression_ratio)
    plt.plot(range(1, epochs + 1), test_accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Test Accuracy vs. Epoch (C={C}, E={E}, B={B})')
    plt.grid(True)
    plt.show()

    np.save(rf'Train_result/C{C}B{B}E{E}.npy', test_accuracies)

