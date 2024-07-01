"""Classes for offline training using ILC results
"""
import numpy as np
import torch.nn.functional as F
from networks import CNN_SEQUENCE
import torch

class PreTrain():
    """
    """
    def __init__(self, mode: str, **PARAMS: dict) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.PARAMS = PARAMS
        if mode is 'seq2seq':
            self.__class__ = type('DynamicClass', (CNN_SEQUENCE,), {})
        else:
            pass

        super().__init__()
        self.build_network(self.device, self.PARAMS)
        if self.PARAMS['is_initialization'] is True:
            self.initialize_weight(self.NN)
    
    def import_data(self, data: dict) -> None:
        """Get the data for pretraining
        """

    def train(self, nn, optimizer, loss_function, inputs, outputs):
        """Train the neural network
        """
