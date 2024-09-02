import os, sys
import matplotlib.pyplot as plt
import tikzplotlib as tp

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

def plot_dis(trajs_1, trajs_2, step=20):
    root = fcs.get_parent_path(lvl=1)
    path_save = os.path.join(root, 'figure', 'tikz', 'dis_comparison.tex')

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fcs.set_axes_format(ax, r'Index', r'$y$')
    for i in range(len(trajs_1)):
        x = [i * step for i in range(len(trajs_1[i][::step]))]
        ax.plot(x, trajs_1[i][::step], linewidth=1.0, linestyle='-', alpha=0.1, color='gray')

    for i in range(len(trajs_2)):
        x = [i * step for i in range(len(trajs_2[i][::step]))]
        ax.plot(x, trajs_2[i][::step], linewidth=1.0, linestyle='-', alpha=0.1, color='blue')
    # ax.legend(fontsize=14)
    tp.save(path_save)
    # plt.show()

if __name__ == '__main__':
    num_traj = 300
    trajs_original = get_traj(num_traj, 'original')
    trajs_shift = get_traj(num_traj, 'shift')
    plot_dis(trajs_original, trajs_shift)
