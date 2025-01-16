import os
import cv2
import numpy as np


MNIST_dir_test = r'C:\Users\Administrator\Desktop\Federated\mnist_test'
MNIST_test_list = os.listdir(MNIST_dir_test)

data_list = []
label_list = []

for label in MNIST_test_list:
    label_path = os.path.join(MNIST_dir_test, label)
    print(label, len(os.listdir(label_path)))
    for image in os.listdir(label_path):
        image_path = os.path.join(label_path, image)
        image = cv2.imread(image_path, 0)/255
        data_list.append([image])
        label_list.append(int(label))

np.save('Test_dataset\MNIST_test_data.npy', data_list)
np.save('Test_dataset\MNIST_test_label.npy', label_list)
