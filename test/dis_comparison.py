import os, sys
import matplotlib.pyplot as plt
import tikzplotlib as tp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from trajectory import TRAJ
import utils as fcs

def get_traj(num_traj, distribution):
    traj_generator = TRAJ(distribution=distribution)
    trajs = []
    for i in range(num_traj):
        traj, _ = traj_generator.get_traj()
        trajs.append(traj.flatten())
    return trajs

def plot_dis(trajs_1, trajs_2, step=5):
    root = fcs.get_parent_path(lvl=1)
    path_save = os.path.join(root, 'figure', 'tikz', 'dis_comparison.tex')

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fcs.set_axes_format(ax, r'Index', r'$y$')
    for i in range(len(trajs_1)):
        x = [i * step for i in range(len(trajs_1[i][::step]))]
        ax.plot(x, trajs_1[i][::step], linewidth=1.0, linestyle='-', alpha=0.05, color='gray')

    for i in range(len(trajs_2)):
        x = [i * step for i in range(len(trajs_2[i][::step]))]
        ax.plot(x, trajs_2[i][::step], linewidth=1.0, linestyle='-', alpha=0.05, color='blue')
    # ax.legend(fontsize=14)
    # tp.save(path_save)
    plt.show()

def plot_area(data):
    colors = ['gray', 'blue', 'red', 'green']
    root = fcs.get_parent_path(lvl=1)
    path_save = os.path.join(root, 'figure', 'tikz', 'dis_contour_shift.tex')

    num = len(data)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fcs.set_axes_format(ax, r'Index', r'$y$')
    
    for i in range(num):
        contours = data[i][0]
        upper = contours[0]
        lower = contours[1]
        mean = data[i][1]
        x = [i*1 for i in range(len(upper))]
        ax.plot(x, upper, linewidth=1.0, linestyle='-', alpha=1, color=colors[i])
        ax.plot(x, lower, linewidth=1.0, linestyle='-', alpha=1, color=colors[i])
        ax.plot(x, mean, linewidth=2.0, linestyle='-', alpha=1, color=colors[i])

    # ax.legend(fontsize=14)
    # tp.save(path_save)
    plt.show()

def get_contour(trajs):
    trajs_matrix = np.array(trajs)
    upper = np.max(trajs_matrix, axis=0)
    lower = np.min(trajs_matrix, axis=0)
    mean = np.mean(trajs_matrix, axis=0)
    return (upper, lower), mean

if __name__ == '__main__':
    num_traj = 10000
    trajs_original = get_traj(num_traj, 'original')
    trajs_tmp = get_traj(num_traj, 'tmp')
    trajs_shift = get_traj(num_traj, 'shift')

    contour_original, mean_original = get_contour(trajs_original)
    contour_tmp, mean_tmp = get_contour(trajs_tmp)
    contour_shift, mean_shift = get_contour(trajs_shift)

    plot_area(((contour_original, mean_original), (contour_tmp, mean_tmp), (contour_shift, mean_shift)))

    # plot_dis(trajs_original, trajs_shift)
