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
    is_save = True

    root = "/home/hao/Desktop/MPI/Online_Convex_Optimization/OnlineILC/data/online_gradient_shift_distribution"
    file = "1.0_0.1_0.1"
    path = os.path.join(root, file)
    path_data = os.path.join(path, 'data')
    path_figure = os.path.join(path, 'figure')
    fcs.mkdir(path_figure)

    files = list_files(path_data)
    s_list = []

    path_marker = os.path.join(path, 'loss_marker')
    path_marker_fig = os.path.join(path_figure, 'marker')
    fcs.mkdir(path_marker_fig)

    markers = list_files(path_marker)
    loss_list = []
    for i in range(len(markers)):
        path_marker_file = os.path.join(path_marker, str(i+1))
        with open(path_marker_file, 'rb') as file:
            yref = pickle.load(file)
            yout = pickle.load(file)
            d = pickle.load(file)
            u = pickle.load(file)
            loss = pickle.load(file)
        loss_list.append(loss)

        fig, axs = plt.subplots(3, 1, figsize=(20, 20))
        ax = axs[0]
        fcs.set_axes_format(ax, r'Time index', r'Displacement')
        ax.plot(yref.flatten()[1:], linewidth=1.0, linestyle='--', label=r'reference')
        ax.plot(yout.flatten(), linewidth=1.0, linestyle='-', label=r'outputs')
        ax.legend(fontsize=14)

        ax = axs[1]
        fcs.set_axes_format(ax, r'Time index', r'Input')
        ax.plot(u, linewidth=1.0, linestyle='-')

        ax = axs[2]
        fcs.set_axes_format(ax, r'Time index', r'disturbance')
        ax.plot(d, linewidth=1.0, linestyle='-')
        if is_save is True:
            plt.savefig(os.path.join(path_marker_fig, str(i)+'.pdf'))
            plt.close()
        else:
            plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    fcs.set_axes_format(ax, r'Iteration', r'Loss')
    ax.plot(loss_list, linewidth=1, linestyle='-')
    if is_save is True:
        plt.savefig(os.path.join(path_marker_fig,'loss.pdf'))
        plt.close()
    else:
        plt.show()

    loss_list = []
    path_train_fig = os.path.join(path_figure, 'train')
    fcs.mkdir(path_train_fig)

    for i in range(len(files)):
        path_file = os.path.join(path_data, str(i))
        with open(path_file, 'rb') as file:
            data = pickle.load(file)
        
        yout = data["yout"].flatten()
        d = data["d"].flatten()
        yref = data["yref"].flatten()[1:]
        u = data["u"].flatten()
        # s = data["hidden_states"].flatten()
        loss = data["loss"]
        loss_list.append(loss)
        
        # s_list.append(s)

        if i%50 == 0:
            fig, axs = plt.subplots(4, 1, figsize=(20, 40))
            ax = axs[0]
            fcs.set_axes_format(ax, r'Time index', r'Displacement')
            ax.plot(yref, linewidth=1.0, linestyle='--', label=r'reference')
            ax.plot(yout, linewidth=1.0, linestyle='-', label=r'outputs')
            ax.legend(fontsize=14)

            ax = axs[1]
            fcs.set_axes_format(ax, r'Time index', r'Input')
            ax.plot(u, linewidth=1.0, linestyle='-')

            ax = axs[2]
            fcs.set_axes_format(ax, r'Time index', r'disturbance')
            ax.plot(d, linewidth=1.0, linestyle='-')

            # ax = axs[3]
            # fcs.set_axes_format(ax, r'Index', r'Hidden states')
            # ax.plot(s, linewidth=1.0, linestyle='-')

            if is_save is True:
                plt.savefig(os.path.join(path_train_fig,str(i)+'.pdf'))
                plt.close()
            else:
                plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    # fcs.set_axes_format(ax, r'Index', r'Hidden state')
    # for i in range(len(s_list)):
    #     if i%50 == 0:
    #         s = s_list[i]
    #         ax.plot(s, linewidth=0.5, linestyle='-')
    # if is_save is True:
    #     plt.savefig(os.path.join(path_train_fig,'hidden_state.pdf'))
    #     plt.close()
    # else:
    #     plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    fcs.set_axes_format(ax, r'Iteration', r'Loss')
    ax.plot(loss_list, linewidth=1, linestyle='-')
    if is_save is True:
        plt.savefig(os.path.join(path_train_fig,'loss.pdf'))
        plt.close()
    else:
        plt.show()

    

if __name__ == '__main__':
    test()