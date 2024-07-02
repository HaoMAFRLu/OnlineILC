"""Classes for offline training using ILC results
"""
import numpy as np
import torch.nn.functional as F
from networks import CNN_SEQUENCE
import torch
import random
from typing import Tuple, List
import matplotlib.pyplot as plt



import utils as fcs
from mytypes import Array, Array2D

class PreTrain():
    """
    """
    def __init__(self, mode: str, **PARAMS: dict) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.PARAMS = PARAMS
        if mode is 'seq2seq':
            self.__class__ = type('DynamicClass', (CNN_SEQUENCE, PreTrain), {})
        else:
            pass
        super(self.__class__, self).__init__(self.device, PARAMS)
        
        self.build_network()
        if self.PARAMS['is_initialization'] is True:
            self.initialize_weight(self.NN)
    
    def import_data(self, data: dict) -> None:
        """Read the data for pretraining

        parameters:
        -----------
        inputs_train: input data for training
        inputs_eval: input data for evaluation
        outputs_train: output data for training
        outputs_eval: output data for evaluation 
        """
        self.inputs_train = data['inputs_train']
        self.inputs_eval = data['inputs_eval']
        self.outputs_train = data['outputs_train']
        self.outputs_eval = data['outputs_eval']

    @staticmethod
    def get_idx(num: int) -> list:
        """Get the index
        """
        return list(range(num))
    
    @staticmethod
    def get_shuffle_idx(idx: list) -> list:
        """Get the shuffle idx
        """
        return random.shuffle(idx)

    def _train(self, NN: torch.nn, 
               optimizer: torch.optim, 
               loss_function: torch.nn.modules.loss, 
               inputs: List[torch.tensor], 
               outputs: List[torch.tensor]) -> Tuple[float, float]:
        """Train the neural network
        """
        total_loss = 0.0
        NN.train()
        idx = self.get_idx(len(inputs))
        self.get_shuffle_idx(idx)
        
        for i in idx:
            data = inputs[i]
            label = outputs[i]
            output = NN(data.float())
            l = output.squeeze().shape[0]
            loss = loss_function(output.squeeze(), label.view(l, -1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss/len(idx)
        return avg_loss


    def _eval(self, NN: torch.nn, 
              loss_function: torch.nn.modules.loss,
              inputs: List[torch.tensor], 
              outputs: List[torch.tensor]) -> Tuple[float, float]:
        """Evaluate the neural network
        """
        total_loss = 0.0 # loss summed over epoch and averaged
        idx = self.get_idx(len(inputs))
        NN.eval()

        for i in idx:
            data = inputs[i]
            label = outputs[i]
            output = NN(data.float())
            l = output.squeeze().shape[0]
            loss = loss_function(output.squeeze(), label.view(l, -1).float())
            total_loss += loss.item()

        avg_loss = total_loss/len(idx)
        return avg_loss
    
    def learn(self, num_epochs: int=100) -> None:
        """Call the training process
        """
        for i in range(num_epochs):
            train_loss = self._train(self.NN,
                                     self.optimizer,
                                     self.loss_function,
                                     self.inputs_train, 
                                     self.outputs_train)
            
            eval_loss = self._eval(self.NN, 
                                   self.loss_function,
                                   self.inputs_eval, 
                                   self.outputs_eval)

            if i == 0:
                loss_train_ini = train_loss
                loss_eval_ini = eval_loss
            
            ptrain = train_loss/loss_train_ini * 100
            peval = eval_loss/loss_eval_ini * 100
            print ('[Epoch {}/{}] TRAIN/VALID loss: {:.6}/{:.6f}||{:.6}%/{:.6f}% '.format(i+1, num_epochs, train_loss, eval_loss, ptrain, peval))

        self.visualize_result(self.NN,
                              self.inputs_eval,
                              self.outputs_eval,
                              is_save=True)

    @staticmethod
    def data_flatten(data: torch.tensor) -> Array2D:
        """Return flatten data, and transfer to cpu
        """
        batch_size = data.shape[0]
        return data.view(batch_size, -1).cpu().detach().numpy()

    def _visualize_result(self, label: Array2D, 
                          outputs: Array2D, 
                          is_save: bool) -> None:
        """
        """
        num_data = label.shape[0]
        for i in range(num_data):
            yref = label[i, :]
            yout = outputs[i, :]
            fig, ax = plt.subplots(1, 1, figsize=(40, 15))
            fcs.set_axes_format(ax, r'Time index', r'Displacement')
            ax.plot(yref, linewidth=0.5, linestyle='--', label=r'reference')
            ax.plot(yout, linewidth=0.5, linestyle='-', label=r'outputs')
            # fcs.set_axes_equal_2d(ax)
            plt.show()

    def visualize_result(self, NN: torch.nn,
                         inputs: List[torch.tensor],
                         outputs: List[torch.tensor],
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
        num_data = len(inputs)
        for i in range(num_data):
            data = inputs[i]
            label = outputs[i]
            output = NN(data.float())
            
            label_flatten = self.data_flatten(label)
            output_flatten = self.data_flatten(output)

            self._visualize_result(label_flatten, output_flatten, is_save)
            


