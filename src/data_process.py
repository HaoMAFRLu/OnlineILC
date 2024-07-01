"""Process the data for training, including 
offline training and online training
"""
import numpy as np
import os
from typing import Tuple, List
import pickle
import torch
import math
import random
from dataclasses import dataclass

from mytypes import *

class DataSeq():
    """Generate sequential inputs and outputs

    parameters:
    -----------
    channel: channel dimension
    height: height dimension
    width: width dimension
    """
    def __init__(self, PARAMS: dataclass) -> None:
        self.k = PARAMS['k']
        self.batch_size = PARAMS['batch_size']
        self.channel = PARAMS['channel']
        self.height = PARAMS['height']
        self.width = PARAMS['width']

    def import_data(self, inputs: List[Array], outputs: List[Array]) -> None:
        """Import the raw data, always the first step
        """
        self.inputs = inputs,
        self.outputs = outputs
        self.num_data = len(self.inputs)

    def get_tensor_data(self, data: List[Array]) -> List[torch.tensor]:
        """Convert data to tensor
        
        parameters:
        -----------
        data: the list of array
        """
        tensor_list = [torch.tensor(arr, device=self.device) for arr in data]
        return tensor_list

    @staticmethod
    def select_idx(idx: list, num: int) -> list:
        """Select certain number of elements from the index list

        parameters:
        -----------
        idx: the given index list
        num: the number of to select elements
        """
        return random.sample(idx, num)

    def select_batch_idx(self, idx: list, batch_size: int) -> list:
        """Split the index according to the given batch size

        parameters:
        -----------
        idx: the given index list
        batch_size: the batch size
        """
        batch_idx = []
        rest_idx = idx
        while len(rest_idx) > batch_size:
            _batch_idx = self.select_idx(rest_idx, batch_size)
            batch_idx.append(_batch_idx)
            rest_idx = list(set(rest_idx) - set(_batch_idx))
        
        if len(rest_idx) > 0:
            batch_idx.append(rest_idx)
        return batch_idx

    def generate_split_idx(self, k: float, batch_size: int, num_data: int):
        """Generate the split index for training data and validation data

        parameters:
        -----------
        k: the proportion of data used for training
        batch_size: the batch size
        num_data: the total number of data
        """
        num_train = math.floor(num_data*k)
        all_idx = list(range(num_data))

        train_idx = self.select_idx(all_idx, num_train)
        num_eval = num_data - num_train
        eval_idx = list(set(all_idx) - set(train_idx))
        batch_idx = self.select_batch_idx(train_idx, batch_size)

        SPLIT_IDX = {
            'num_train': num_train,
            'num_eval': num_eval,
            'all_idx': all_idx,
            'train_idx': train_idx,
            'eval_idx': eval_idx,
            'batch_idx': batch_idx
        }
        return SPLIT_IDX

    def split_data(self, data: List[torch.tensor], idx: list):
        """
        """
        return torch.cat([data[i].flatten().unsqueeze(0) for i in idx], dim=0)

    def _split_data(self, data: List[torch.tensor], batch_idx: list, eval_idx: list):
        """
        """
        train = []
        eval = []
        eval.append(self.split_data(data, eval_idx))

        l = len(batch_idx)
        for i in range(l):
            train.append(self.split_data(data, batch_idx[i]))
        
        return train, eval
    
    def generate_data(self):
        """Generate the data for training
        """
        total_inputs_tensor = self.get_tensor_data(self.inputs)
        total_outputs_tensor = self.get_tensor_data(self.outputs)
        SPLIT_IDX = self.generate_split_idx(self.k, self.batch_size, self.num_data)
        inputs_train, inputs_eval = self._split_data(total_inputs_tensor, SPLIT_IDX['batch_idx'], SPLIT_IDX['eval_idx'])
        outputs_train, outputs_eval = self._split_data(total_outputs_tensor, SPLIT_IDX['batch_idx'], SPLIT_IDX['eval_idx'])

class DataProcess():
    """Prepare the inputs and outputs (labels) for the neural network
    """
    def __init__(self, PARAMS: dataclass) -> None:
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.path_data = os.path.join(self.parent_dir, 'data', 'pretraining')
        self.reset_class(PARAMS)

    def reset_class(self, PARAMS: dataclass):
        """Reset the class
        """
        if PARAMS['mode'] is 'seq2seq':
            _data_process = DataSeq(PARAMS)
        else:
            pass

    def _load_keys(self) -> list:
        """Return the list of key words
        """
        path = os.path.join(self.path_data, 'keys')
        with open(path, 'rb') as file:
            keys = pickle.load(file)
        return keys

    def _load_data(self) -> Tuple[List[Array]]:
        """Load the data from file
        """
        path = os.path.join(self.path_data, 'ilc')
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data
    
    def read_raw_data(self, input_name: str, 
                      output_name: str) -> Tuple[List[Array], List[Array]]:
        """Read the raw data from the file

        parameters:
        -----------
        name_input: name of the input
        name_output: name of the output (label)

        returns:
        --------
        raw_inputs: the raw input data
        raw_outputs: the raw outputs data
        """
        keys = self._load_keys()
        data = self._load_data()
        raw_inputs = data[keys.index(input_name)]
        raw_outputs = data[keys.index(output_name)]
        return raw_inputs, raw_outputs
        

    def get_data(self, mode: str, **kwargs):
        """Return the inputs and outputs (labels) for the neural networks
        
        parameters:
        ----------- 
        mode: offline training or online training 
        """
        if mode is 'offline':
            raw_inputs, raw_outputs = self.read_raw_data(kwargs['input_name'], kwargs['output_name'])

        elif mode is 'online':
            return self.get_online_training_data(**kwargs)



