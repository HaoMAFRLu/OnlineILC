"""Classes for offline training using ILC results
"""
from networks import NETWORK_CNN
import torch
import random
from typing import Tuple, List

import os
from datetime import datetime

import utils as fcs
from mytypes import Array, Array2D

class PreTrain():
    """
    """
    def __init__(self, mode: str, 
                 PARAMS: dict) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        current_time = datetime.now()
        folder_name = current_time.strftime("%Y%m%d_%H%M%S")

        # path for saving checking points
        self.num_check_points = 1000
        self.path_model = os.path.join(parent_dir, 'data', 'offline_training', folder_name)
        fcs.mkdir(self.path_model)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.PARAMS = PARAMS
        if mode is 'seq2seq':
            self.__class__ = type('DynamicClass', (NETWORK_CNN, PreTrain), {})
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
        avg_loss_train = []
        avg_loss_eval = []

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

            self.scheduler.step(eval_loss)
            avg_loss_train.append(train_loss)
            avg_loss_eval.append(eval_loss)

            loss_train_ini = avg_loss_train[0]
            loss_eval_ini = avg_loss_eval[0]
            
            ptrain = train_loss/loss_train_ini * 100
            peval = eval_loss/loss_eval_ini * 100

            current_lr = self.optimizer.param_groups[0]['lr']
            print('[Epoch {}/{}] LR: {:.6f} | TRAIN/VALID loss: {:.6}/{:.6f}||{:.6}%/{:.6f}% '.format(i+1, num_epochs, current_lr, train_loss, eval_loss, ptrain, peval))

            if (i+1) % self.num_check_points == 0:
                self.save_checkpoint(i+1)
        
        self.save_loss(data=())

    def save_checkpoint(self, num_epoch: int) -> None:
        """Save the checkpoint
        """
        checkpoint = {
            'epoch': num_epoch,
            'model_state_dict': self.NN.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        check_point_path = self.path_model + '/' + f'checkpoint_epoch_{num_epoch}.pth'
        torch.save(checkpoint, check_point_path)

    @staticmethod
    def data_flatten(data: torch.tensor) -> Array2D:
        """Return flatten data, and transfer to cpu
        """
        batch_size = data.shape[0]
        return data.view(batch_size, -1).cpu().detach().numpy()

    
            


