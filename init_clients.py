import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.len = len(data)
        self.x_data = torch.from_numpy(np.array(list(map(lambda x: x[0], data)), dtype=np.float32))
        self.y_data = torch.from_numpy(np.array(list(map(lambda x: x[-1], data)))).squeeze().long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def clients_dataloader(batch_size=16):
    dataloader_list = []

    dir_path = os.listdir("Client_datasets")
    for dir in dir_path:
        data_path = os.path.join("Client_datasets", dir, "data.npy")
        label_path = os.path.join("Client_datasets", dir, "label.npy")

        data = np.load(data_path)
        label = np.load(label_path)

        dataset = [[i, j] for i, j in zip(data, label)]
        dataloader = DataLoader(CustomDataset(dataset), shuffle=True, batch_size=batch_size)
        dataloader_list.append(dataloader)

    return dataloader_list


if __name__ == '__main__':
    dataloaders = clients_dataloader()
    if dataloaders:
        print(f"Loaded {len(dataloaders)} client dataloaders.")
    else:
        print("No dataloaders loaded.")
