import pickle
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

def list_files(directory):
    items = os.listdir(directory)
    files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
    return files


def test():
    root = "/home/hao/Desktop/MPI/Online_Convex_Optimization/OnlineILC/data/online_training"
    file = "20240805_122459"
    path = os.path.join(root, file)
    path_data = os.path.join(path, 'data')
    path_figure = os.path.join(path, 'figure')
    fcs.mkdir(path_figure)

    files = list_files(path_data)
    for i in range(len(files)):
        path_file = os.path.join(path_data, str(i))
        with open(path_file, 'rb') as file:
            data = pickle.load(file)
        
        yout = data["yout"].flatten()
        d = data["d"].flatten()
        yref = data["yref"].flatten()[1:]
        u = data["u"].flatten()[1:]

        fig, axs = plt.subplots(3, 1, figsize=(20, 20))
        ax = axs[0]
        fcs.set_axes_format(ax, r'indefdfsafd', r'Displacement')
        ax.plot(yref, linewidth=1.0, linestyle='--', label=r'reference')
        ax.plot(yout, linewidth=1.0, linestyle='-', label=r'outputs')
        ax.legend(fontsize=14)

        ax = axs[1]
        fcs.set_axes_format(ax, r'Time index', r'Input')
        ax.plot(u, linewidth=1.0, linestyle='-')

        ax = axs[2]
        fcs.set_axes_format(ax, r'Time index', r'disturbance')
        ax.plot(d, linewidth=1.0, linestyle='-')

        # if self.is_save is True:
        #     plt.savefig(os.path.join(self.path_figure,str(i)+'.pdf'))
        #     plt.close()
        # else:
        plt.show()



if __name__ == '__main__':
    test()