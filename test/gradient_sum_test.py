import sys, os
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs


if __name__ == '__main__':
    root = fcs.get_parent_path(lvl=1)
    folder1 = 'online_training'
    path = os.path.join(root, 'data', folder1)
    folders = '20240829_154717'
    file = 'data_gradient_sum'

    path_file = os.path.join(path, folders, file)
    with open(path_file, 'rb') as file:
        data = pickle.load(file)

    sum_list = []
    for i in range(len(data)):
        sum1 = 0
        for ii in range(len(data[i])):
            sum1 += data[i][ii]
        sum1 /= len(data[i])
        sum_list.append(np.linalg.norm(sum1))

    file = 'loss_marker_loss'
    path_file = os.path.join(path, folders, file)
    with open(path_file, 'rb') as file:
        loss = pickle.load(file)

    x = [i * 20 for i in range(len(loss))]
    
    plt.semilogy(sum_list)
    plt.semilogy(x, loss)
    plt.show()
    print(np.min(sum_list))