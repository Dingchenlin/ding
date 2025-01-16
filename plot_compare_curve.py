import matplotlib.pyplot as plt
import numpy as np

line1 = np.load("Train_result/C0.1B10E5.npy")
line2 = np.load("Train_result/C0.1B50E5.npy")
line3 = np.load("Train_result/C0.1B100E5.npy")
line4 = np.load("Train_result/C0.1B600E1.npy")
line5 = np.load("Train_result/C0.1B600E5.npy")
line6 = np.load("Train_result/C0.2B600E5.npy")
line7 = np.load("Train_result/C1B600E5.npy")


plt.plot(line1, linewidth=3.0,label='C=0.1,B=10,E=5')
plt.plot(line2, linewidth=3.0,label='C=0.1,B=50,E=5')
plt.plot(line3, linewidth=3.0,label='C=0.1,B100,E=5')
plt.plot(line4, linewidth=3.0,label='C=0.1,B=∞,E=1')
plt.plot(line5, linewidth=3.0,label='C=0.1,B=∞,E=5')
plt.plot(line6, linewidth=3.0,label='C=0.2,B=∞,E=5')
plt.plot(line7, linewidth=3.0,label='C=1,B=∞,E=5')

plt.grid()
plt.legend()

plt.xlabel("Tran epoch")
plt.ylabel("Accuracy")
plt.title("Federated Learning")

plt.show()
