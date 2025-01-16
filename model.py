import torch
class CNN(torch.nn.Module):
    def __init__(self, in_channels=1, classes=10):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=5)
        self.max_pool1 = torch.nn.MaxPool2d(2, 2)
        self.norm1 = torch.nn.BatchNorm2d(32)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.max_pool2 = torch.nn.MaxPool2d(2, 2)
        self.norm2 = torch.nn.BatchNorm2d(64)
        self.linear = torch.nn.Linear(64 * 4 * 4, classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        # x = self.norm1(x)

        x = self.conv2(x)
        x = self.max_pool2(x)
        # x = self.norm2(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


if __name__ == '__main__':
    model = CNN(3, 10)
    print(model)
    for name, parameters in model.named_parameters():
        if "bias" in name or "norm" in name:
            continue
        else:
            print(name)
