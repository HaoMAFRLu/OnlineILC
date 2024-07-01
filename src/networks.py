"""Classes for the neural networks
"""
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn
from network.CNN import CNN_SEQ

class CNN_SEQUENCE():
    """The neural network with sequences as input and output
    
    parameters:
    -----------
    NN_PARAMS: hyperparameters for creating the network
        |-- loss_function: type of the loss function
        |-- learning_rate: learning rate of the optimizer
        |-- weight_decay: weight decay of the optimizer
        |-- 
    """
    def __init__(self, device: str, PARAMS: dict) -> None:
        self.device = device
        self.PARAMS = PARAMS
        self.build_network()
        if self.PARAMS['is_initialization'] is True:
            self.initialize_weight(self.NN)
    
    @staticmethod
    def initialize_weight(nn: torch.nn, sparsity: float=0.90, std: float=0.1) -> None:
        """Initialize the weight of the neural network.
        TODO: Check whether the weights of the original network are changed accordingly
        """
        for layer in nn.modules():
            if isinstance(layer, torch.nn.Linear):
                init.sparse_(layer.weight, sparsity=sparsity, std=std)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
            elif isinstance(layer, torch.nn.Conv2d):
                init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias, 0)

    @staticmethod
    def _get_loss_function(name: str) -> torch.nn.functional:
        """Return the loss function of the neural network
        """
        if name == 'Huber':
            return torch.nn.HuberLoss()
        if name == 'L1':
            return torch.nn.L1Loss(reduction='mean')
        if name == 'MSE':
            return torch.nn.MSELoss(reduction='mean')

    @staticmethod
    def _get_optimizer(NN: torch.nn, lr: float, wd: float) -> torch.nn.functional:
        """Return the optimizer of the neural network
        """
        return torch.optim.Adam(NN.parameters(),lr=lr,weight_decay=wd)

    @staticmethod
    def _get_model(PARAMS) -> torch.nn:
        """Create the neural network
        """
        return CNN_SEQ(in_channel=PARAMS['channel'],
                       height=PARAMS['height'],
                       width=PARAMS['width'],
                       filter_size=PARAMS['filter_size'],
                       output_dim=PARAMS['output_dim'])

    def build_network(self) -> None:
        self.NN = self._get_model(PARAMS=self.PARAMS)
        self.NN.to(self.device)
        self.loss_function = self._get_loss_function(self.PARAMS['loss_function'])
        self.loss_function.to(self.device)
        self.optimizer = self._get_optimizer(self.NN, self.PARAMS['learning_rate'], self.PARAMS['weight_decay'])
        
