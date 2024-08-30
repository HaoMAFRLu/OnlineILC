import matplotlib.pyplot as plt
import tikzplotlib as tp

from trajectory import TRAJ
import utils as fcs

num = 50
traj_list = []

path_planner = TRAJ()
for i in range(num):
    traj, _ = path_planner.get_traj()
    traj_list.append(traj.flatten())

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
fcs.set_axes_format(ax, r'Epoch', r'Loss')

for traj in traj_list:
    ax.plot(traj, linewidth=0.5, linestyle='-', alpha=0.5, color='gray')
# ax.legend(fontsize=14)
# tp.save(path_save)
plt.show()