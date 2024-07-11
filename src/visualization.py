"""Classes used for showing the training results
"""
from pathlib import Path
import torch
import torch.nn
from typing import Any, List, Tuple
import os
import matplotlib.pyplot as plt

import utils as fcs
from mytypes import Array, Array2D, Array3D

class Visual():
    """
    """
    def __init__(self, PARAMS: dict) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.abspath(os.path.join(current_dir, os.pardir))
        self.folder = self.get_path_params(PARAMS['paths'])
        
        self.path_params = os.path.join(self.root, 'data', self.folder, PARAMS['checkpoint'])
        self.path_figure = os.path.join(self.root, 'figure', self.folder, PARAMS['checkpoint'])
        fcs.mkdir(self.path_figure)

    @staticmethod
    def _get_path_params(paths: List[str]) -> Path:
        """Recursively generate paths
        """
        path_params = []
        for pth in paths:
            os.path.join(path_params, pth)
        return path_params

    def get_path_params(self, paths: List[str], 
                        file_name: str) -> Path:
        if isinstance(paths, str):
            return os.path.join(paths, file_name)
        elif isinstance(paths, list):
            return os.path.join(self._get_path_params(paths), file_name)

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
    
    def plot_results(self, NN: torch.nn,
                     data: dict, 
                     is_save: bool=True) -> None:
        """Plot the results using specified model
        """
        
    
    @staticmethod
    def data_flatten(data: torch.tensor) -> Array2D:
        """Return flatten data, and transfer to cpu
        """
        batch_size = data.shape[0]
        return data.view(batch_size, -1).cpu().detach().numpy()

    def _visualize_result(self, label: Array2D, 
                          outputs: Array2D,
                          inputs: Array2D, 
                          is_save: bool) -> None:
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
            ax.plot(uref, linewidth=0.5, linestyle='--', label=r'reference')
            ax.plot(uout, linewidth=0.5, linestyle='-', label=r'outputs')
            ax = axs[1]
            fcs.set_axes_format(ax, r'Time index', r'Input')
            ax.plot(yref, linewidth=0.5, linestyle='-', label=r'reference')
            if is_save is True:
                plt.savefig(os.path.join(self.path_figure,str(i)+'.pdf'))
                plt.close()
            else:
                plt.show()

    def visualize_result(self, NN: torch.nn,
                         inputs: List[torch.tensor],
                         outputs: List[torch.tensor],
                         loss_train: list,
                         loss_eval: list,
                         is_save: bool) -> None:
        """Visualize the comparison between the ouputs of 
        the neural network and the labels

        parameters:
        -----------
        NN: the neural network
        inputs: the input data
        outputs: the output label
        is_save: if save the plots
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        fcs.set_axes_format(ax, r'Time index', r'Loss')
        ax.semilogy(loss_train, linewidth=1.0, linestyle='--', label=r'Training Loss')
        ax.semilogy(loss_eval, linewidth=1.0, linestyle='-', label=r'Eval Loss')
        if is_save is True:
            plt.savefig(os.path.join(self.path_figure,'loss.pdf'))
            plt.close()
        else:
            plt.show()

        num_data = len(inputs)
        for i in range(num_data):
            data = inputs[i]
            label = outputs[i]
            output = NN(data.float())
            
            label_flatten = self.data_flatten(label)
            output_flatten = self.data_flatten(output)
            data_flatten = self.data_flatten(data)

            self._visualize_result(label_flatten, output_flatten, data_flatten, is_save)
        








