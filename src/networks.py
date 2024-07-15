"""Classes for the neural networks
"""
import torch.nn.init as init
import torch.nn
from network.CNN import CNN_SEQ, SimplifiedResNet, CustomResNet18, TransformerModel
from network.ResNeXt import ResNeXt, Bottleneck
from custom_loss import CustomLoss
import torch.optim.lr_scheduler as lr_scheduler

class NETWORK_CNN():
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
    def _get_loss_function(mode: str, lambda_regression: float) -> torch.nn.functional:
        """Return the loss function of the neural network
        """
        return CustomLoss(lambda_regression, mode)

    @staticmethod
    def _get_optimizer(NN: torch.nn, lr: float, wd: float) -> torch.nn.functional:
        """Return the optimizer of the neural network
        """
        return torch.optim.Adam(filter(lambda p: p.requires_grad, NN.parameters()), lr=lr,weight_decay=wd)

    @staticmethod
    def _get_scheduler(optimizer: torch.nn.functional,
                       factor: float=0.1,
                       patience: int=500):
        return lr_scheduler.ReduceLROnPlateau(optimizer, 
                                              mode='min', 
                                              factor=factor, 
                                              patience=patience)

    @staticmethod
    def _get_model(PARAMS) -> torch.nn:
        """Create the neural network
        """
        return CNN_SEQ(in_channel=PARAMS['channel'],
                       height=PARAMS['height'],
                       width=PARAMS['width'],
                       filter_size=PARAMS['filter_size'],
                       output_dim=PARAMS['output_dim'])
        # return SimplifiedResNet()
        # return CustomResNet18()
        # return ResNeXt(Bottleneck, [2, 2, 2, 2], cardinality=8, num_classes=550)


    @staticmethod
    def count_parameters(model: torch.nn) -> int:
        """Count the parameters in a neural network
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def print_parameter_details(model: torch.nn) -> None:
        """Print the information of the neural network
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Parameter: {name}, Size: {param.size()}, Number of parameters: {param.numel()}")

    def build_network(self) -> None:
        self.NN = self._get_model(PARAMS=self.PARAMS)
        self.NN.to(self.device)
        self.loss_function = self._get_loss_function(self.PARAMS['loss_function'], self.PARAMS['lambda_regression'])
        self.loss_function.to(self.device)
        self.optimizer = self._get_optimizer(self.NN, self.PARAMS['learning_rate'], self.PARAMS['weight_decay'])
        self.scheduler = self._get_scheduler(self.optimizer)
