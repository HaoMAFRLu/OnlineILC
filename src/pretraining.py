"""Classes for offline training using ILC results
"""
import numpy as np
import torch.nn.functional as F
from networks import *
from dataclasses import dataclass


class PreTrain:
    """
    """
    def __init__(self, model_name: str, 
                 NN_PARAMS: dataclass, DATA_PARAMS: dataclass) -> None:
        self.NN_PARAMS = NN_PARAMS
        self.DATA_PARAMS = DATA_PARAMS
        self.build_network()
        self.get_data()
    
    def get_data(self) -> None:
        """Get the data for pretraining
        """     


    def build_network(self) -> None:
        """Create the network
        """
        model = CNN_SEQ(self.NN_PARAMS)

    def train(self, nn, optimizer, loss_function, inputs, outputs):
        """Train the neural network
        """