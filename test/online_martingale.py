"""Visualization test
"""
import os, sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tikzplotlib as tp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

def read_data(file):
    with open(file, 'rb') as file:
            data = pickle.load(file)
    return data[0][::10], data[1][::10]
    
def plot_tikz(trajs_data, trajs_marker, trajs_martingale, path_save, 
              if_data=True, if_marker=True, if_martingale=True):
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fcs.set_axes_format(ax, r'Epoch', r'Loss')

    if if_data is True:
        for i in range(len(trajs_data)):
            traj_data = trajs_data[i]
            x = [i * 1 for i in range(len(traj_data))]
            ax.plot(x, traj_data, linewidth=0.5, linestyle='-')

    if if_marker is True:
        for i in range(len(trajs_marker)):
            traj_marker = trajs_marker[i]
            x = [i * 20 for i in range(len(traj_marker))]
            ax.plot(x, traj_marker, linewidth=1.0, linestyle='-')

    if if_martingale is True:
        for i in range(len(trajs_martingale)):
            traj_martingale = trajs_martingale[i]
            x = [i * 1 for i in range(len(traj_martingale))]
            ax.plot(x, traj_martingale, linewidth=1.5, linestyle='-')
    # ax.legend(fontsize=14)
    # tp.save(path_save)
    plt.show()

def average(lst):
    return sum(lst) / len(lst)

def _get_martingale(data, mode, rolling):
    num = len(data)
    rolling_list = []
    martingale_list = []
    for i in range(num):
        rolling_list.append(data[i])

        if mode == 'rolling':
            if len(rolling_list) > rolling:
                rolling_list.pop(0)
        
        sum_value = sum(rolling_list)
        martingale_list.append(np.linalg.norm(sum_value/len(rolling_list)))
    return martingale_list

def get_martingale(data_list, mode, rolling):
    num = len(data_list)
    martingale_list = []
    for i in range(num):
        martingale_list.append(_get_martingale(data_list[i], mode, rolling))
    return martingale_list
    
if __name__ == '__main__':
    folder1s = ['martingale_test2']
    folders = ['1.0_0.1_0.1']

    root = fcs.get_parent_path(lvl=1)
    path = os.path.join(root, 'data')

    data_loss = []
    marker_loss = []
    name_list = []
    gradient_list = []
    
    for (folder1, folder) in zip(folder1s, folders):
        path_gradient = os.path.join(path, folder1, folder, 'data_gradient')
        path_data = os.path.join(path, folder1, folder, 'data_loss')
        path_marker = os.path.join(path, folder1, folder, 'loss_marker_loss')

        if os.path.isfile(path_data) and os.path.isfile(path_marker):
            name_list.append(folder)

            with open(path_gradient, 'rb') as file:
                gradient = pickle.load(file)
            gradient_list.append(gradient)

            with open(path_data, 'rb') as file:
                loss = pickle.load(file)
            data_loss.append(loss)

            with open(path_marker, 'rb') as file:
                loss = pickle.load(file)

            marker_loss.append(loss)
        else:
            print('No file!')

    martingale_list = get_martingale(gradient_list, mode='rolling', rolling=100)
    path_fig = os.path.join(root, 'figure', 'tikz', 'martingale_gd_shift.tex')
    plot_tikz(data_loss, marker_loss, martingale_list, path_fig, if_data=True, if_marker=True, if_martingale=True)


