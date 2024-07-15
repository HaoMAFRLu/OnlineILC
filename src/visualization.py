"""Classes used for showing the training results
"""
from pathlib import Path
import torch
import torch.nn
from typing import Any, List, Tuple
import os
import matplotlib.pyplot as plt
import pickle

import utils as fcs
from mytypes import Array, Array2D, Array3D

class Visual():
    """
    """
    def __init__(self, PARAMS: dict) -> None:
        self.is_save = PARAMS["is_save"]

        self.root = fcs.get_parent_path(lvl=1)
        self.folder = self.get_path_params(PARAMS['paths'])
        
        self.path_params = os.path.join(self.root, 'data', self.folder, PARAMS['checkpoint']+'.pth')
        self.path_loss = os.path.join(self.root, 'data', self.folder, 'loss')
        self.path_figure = os.path.join(self.root, 'figure', self.folder, PARAMS['data'], PARAMS['checkpoint'])
        fcs.mkdir(self.path_figure)

    @staticmethod
    def _get_path_params(paths: List[str]) -> Path:
        """Recursively generate paths
        """
        path_params = paths[0]
        if len(paths) > 1:
            for i in range(1, len(paths)):
                path_params = os.path.join(path_params, paths[i])
        return path_params

    def get_path_params(self, paths: List[str]) -> Path:
        if isinstance(paths, str):
            return paths
        elif isinstance(paths, list):
            return os.path.join(self._get_path_params(paths))

    def load_model(self, path_params: Path=None,
                   **kwargs: Any) -> None:
        """Specify the model structure and 
        load the pre-trained model parameters
        
        parameters:
        -----------
        params_path: path to the pre-trained parameters
        model: the model structure
        optimizer: the optimizer
        """
        if path_params is None:
            path_params = self.path_params

        checkpoint = torch.load(path_params)
        if 'model' in kwargs:
            kwargs['model'].load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer' in kwargs:
            kwargs['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
    
    @staticmethod
    def data_flatten(data: torch.tensor) -> Array2D:
        """Return flatten data, and transfer to cpu
        """
        batch_size = data.shape[0]
        return data.view(batch_size, -1).cpu().detach().numpy()
    
    def load_loss(self, path: Path) -> Tuple[list]:
        """Load the loss
        """
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def plot_loss(self, data: Tuple[list]) -> None:
        """Plot the loss

        parameters:
        -----------
        data: including training loss and evaluation loss
        """
        train_loss = data[0]
        eval_loss = data[1]

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        fcs.set_axes_format(ax, r'Epoch', r'Loss')
        ax.plot(train_loss, linewidth=1.0, linestyle='--', label=r'Training Loss')
        ax.plot(eval_loss, linewidth=1.0, linestyle='-', label=r'Eval Loss')
        ax.legend(fontsize=14)
        if self.is_save is True:
            plt.savefig(os.path.join(self.path_figure, 'loss.pdf'))
            plt.close()
        else:
            plt.show()

    def _visualize_result(self, label: Array2D, 
                          outputs: Array2D,
                          inputs: Array2D) -> None:
        """
        """
        num_data = label.shape[0]
        for i in range(num_data):
            uref = label[i, :]
            uout = outputs[i, :]
            yref = inputs[i, :]

            fig, axs = plt.subplots(2, 1, figsize=(15, 20))
            ax = axs[0]
            fcs.set_axes_format(ax, r'Time index', r'Displacement')
            ax.plot(uref/1000, linewidth=1.0, linestyle='--', label=r'reference')
            ax.plot(uout/1000, linewidth=1.0, linestyle='-', label=r'outputs')
            ax.legend(fontsize=14)

            ax = axs[1]
            fcs.set_axes_format(ax, r'Time index', r'Input')
            ax.plot(yref, linewidth=1.0, linestyle='-', label=r'reference')
            ax.legend(fontsize=14)

            if self.is_save is True:
                plt.savefig(os.path.join(self.path_figure,str(i)+'.pdf'))
                plt.close()
            else:
                plt.show()

    def plot_results(self, NN: torch.nn,
                    inputs: List[torch.tensor],
                    outputs: List[torch.tensor]) -> None:
        """Visualize the comparison between the ouputs of 
        the neural network and the labels

        parameters:
        -----------
        NN: the neural network
        inputs: the input data
        outputs: the output label
        """
        num_data = len(inputs)
        for i in range(num_data):
            data = inputs[i]
            label = outputs[i]
            output = NN(data.float())
            
            label_flatten = self.data_flatten(label)
            output_flatten = self.data_flatten(output)
            data_flatten = self.data_flatten(data)

            self._visualize_result(label_flatten, output_flatten, data_flatten)
        








