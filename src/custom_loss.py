"""Classes for the custom loss functions
"""
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, lambda_regression: float, mode: str) -> None:
        super(CustomLoss, self).__init__()
        self.mode = mode
        self.lambda_regression = lambda_regression

        self.loss_function1 = self._get_loss_function(self.mode)
        
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

    def forward(self, outputs, labels) -> torch.nn.functional:
        """Return the loss function
        """
        loss = self.loss_function1(outputs, labels)
        diff = outputs[:, 1:] - outputs[:, :-1]
        reg_loss = torch.mean(diff ** 2)

        total_loss = loss + self.lambda_regression * reg_loss
        return total_loss
        