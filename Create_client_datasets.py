import os
import numpy as np
import cv2

# 创建100个客户端的文件夹
# for i in range(1, 101):
#     os.makedirs(os.path.join('Client_datasets', f'client_{i}'), exist_ok=True)
A = 10
# 获取每个类别数据的图像数据
number_0 = [[cv2.imread(os.path.join("mnist_train/0", i), 0) / 255] for i in os.listdir("mnist_train/0")]
number_1 = [[cv2.imread(os.path.join("mnist_train/1", i), 0) / 255] for i in os.listdir("mnist_train/1")]
number_2 = [[cv2.imread(os.path.join("mnist_train/2", i), 0) / 255] for i in os.listdir("mnist_train/2")]
number_3 = [[cv2.imread(os.path.join("mnist_train/3", i), 0) / 255] for i in os.listdir("mnist_train/3")]
number_4 = [[cv2.imread(os.path.join("mnist_train/4", i), 0) / 255] for i in os.listdir("mnist_train/4")]
number_5 = [[cv2.imread(os.path.join("mnist_train/5", i), 0) / 255] for i in os.listdir("mnist_train/5")]
number_6 = [[cv2.imread(os.path.join("mnist_train/6", i), 0) / 255] for i in os.listdir("mnist_train/6")]
number_7 = [[cv2.imread(os.path.join("mnist_train/7", i), 0) / 255] for i in os.listdir("mnist_train/7")]
number_8 = [[cv2.imread(os.path.join("mnist_train/8", i), 0) / 255] for i in os.listdir("mnist_train/8")]
number_9 = [[cv2.imread(os.path.join("mnist_train/9", i), 0) / 255] for i in os.listdir("mnist_train/9")]

# 每个类别的样本总数除以100个客户端
first_round_number = [len(number_0) // A, len(number_1) // A, len(number_2) // A, len(number_3) // A,
                      len(number_4) // A, len(number_5) // A, len(number_6) // A, len(number_7) // A,
                      len(number_8) // A, len(number_9) // A]

# 每个类别剩余的样本数量
remain_number = [len(number_0) % A, len(number_1) % A,
                 len(number_2) % A, len(number_3) % A,
                 len(number_4) % A, len(number_5) % A,
                 len(number_6) % A, len(number_7) % A,
                 len(number_8) % A, len(number_9) % A]

# 获得所有客户端数据集构成的列表
number_list = [number_0, number_1, number_2, number_3, number_4, number_5, number_6, number_7, number_8, number_9]

# 处理剩余数据
remain_data = (number_0[-remain_number[0]:] + number_1[-remain_number[1]:] + number_2[-remain_number[2]:] +
               number_3[-remain_number[3]:] + number_4[-remain_number[4]:] + number_5[-remain_number[5]:] +
               number_6[-remain_number[6]:] + number_7[-remain_number[7]:] + number_8[-remain_number[8]:] +
               number_9[-remain_number[9]:])

# 剩余数据的标签
remain_label = ([0] * remain_number[0] + [1] * remain_number[1] + [2] * remain_number[2] + [3] * remain_number[3] +
                [4] * remain_number[4] + [5] * remain_number[5] + [6] * remain_number[6] + [7] * remain_number[7] +
                [8] * remain_number[8] + [9] * remain_number[9])

# 开始构建A个客户端的数据
for i in range(A):
    data = []
    label = []
    for index, j in enumerate(number_list):
        data += j[i*first_round_number[index]:(i+1)*first_round_number[index]]
        label += [index] * first_round_number[index]

    # 将剩余的数据再补充分配到每个数据集当中
    data += remain_data[i*len(remain_data)//A:(i+1)*len(remain_data)//A]
    label += remain_label[i*len(remain_label)//A:(i+1)*len(remain_label)//A]

    # 缓存每个数据集
    print(i+1, np.shape(data),np.shape(label))
    np.save(rf"C:\Users\Administrator\Desktop\Federated\Client_datasets\client_{i+1}\data.npy", data, allow_pickle=True)
    np.save(rf"C:\Users\Administrator\Desktop\Federated\Client_datasets\client_{i+1}\label.npy", label, allow_pickle=True)

