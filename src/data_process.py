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

from mytypes import *
import utils as fcs
class DataWin():
    """Generate windowed data

    parameters:
    -----------
    """
    def __init__(self) -> None:
        pass

class DataSeq():
    """Generate sequential inputs and outputs

    parameters:
    -----------
    channel: channel dimension
    height: height dimension
    width: width dimension
    """
    def __init__(self, PARAMS: dict) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.is_normalization = PARAMS['is_normalization']
        self.input_scale = PARAMS['input_scale']
        self.output_scale = PARAMS['output_scale']

        self.k = PARAMS['k']
        self.batch_size = PARAMS['batch_size']
        self.channel = PARAMS['channel']
        self.height = PARAMS['height']
        self.width = PARAMS['width']

    @staticmethod
    def get_max_value(data: List[Array]) -> float:
        """Return the maximum value
        """
        return np.max(np.concatenate(data))

    @staticmethod
    def get_mean_value(data: List[Array]) -> float:
        """Return the mean value of the data
        """
        return np.mean(np.concatenate(data))

    @staticmethod
    def get_min_value(data: List[Array]) -> float:
        """Return the minimum value
        """
        return np.min(np.concatenate(data))
    
    def _get_min_max_value(self) -> None:
        """Get the minimum and maximum value of the data
        """
        self.max_input = self.get_max_value(self.inputs)
        self.max_output = self.get_max_value(self.outputs)
        self.min_input = self.get_min_value(self.inputs)
        self.min_output = self.get_min_value(self.outputs)

    def import_data(self, inputs: List[Array], outputs: List[Array]) -> None:
        """Import the raw data, always the first step
        """
        self.inputs = inputs.copy()
        self.outputs = outputs.copy()
        self.num_data = len(self.inputs)
        self._get_min_max_value()

    def get_tensor_data(self, data: List[Array]) -> List[torch.tensor]:
        """Convert data to tensor
        
        parameters:
        -----------
        data: the list of array

        returns:
        -------
        tensor_list: a list of tensors, which are in the shape of 1 x channel x height x width
        """
        tensor_list = [torch.tensor(arr, device=self.device).view(1, self.channel, self.height, self.width) for arr in data]
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

        returns:
        --------
        num_train: the number of data used for training
        num_eval: the number of data used for evaluation
        all_idx: the indcies of all data points
        train_idx: the indices of data points used for training
        eval_idx: the indices of data points used for evaluation
        batch_idx: the indices of data in each mini batch
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
        return torch.cat([data[i] for i in idx], dim=0)

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

    def normalize(self, data: List[Array],
                  min_value: float,
                  max_value: float,
                  scale: float=1.0) -> List[Array]:
        """Map the data into [-1, 1]*scale
        """
        num_data = len(data)
        data_norm = [None] * num_data
        for i in range(num_data):
            data_norm[i] = (2*(data[i]-min_value)/(max_value-min_value) - 1) * scale
        
        mean = self.get_mean_value(data_norm)
        return data_norm - mean

    def generate_data(self):
        """Generate the data for training

        parameters:
        -----------
        total_inputs_tensor: the list of all training data
        total_outputs_tensor: the list of all labels
        SPLIT_IDX: the structrue data containing the indices used
                   for separating training data and evaluation data
        is_normalization: if normalize the inputs and outputs
        """
        if self.is_normalization is True:
            _inputs = self.normalize(self.inputs, 
                                     self.min_input, 
                                     self.max_input, 
                                     scale=self.input_scale)
            _outputs = self.normalize(self.outputs, 
                                      self.min_output, 
                                      self.max_output, 
                                      scale=self.output_scale)
        else:
            _inputs = self.inputs.copy()
            _outputs = self.outputs.copy()

        total_inputs_tensor = self.get_tensor_data(_inputs)
        total_outputs_tensor = self.get_tensor_data(_outputs)


        SPLIT_IDX = self.generate_split_idx(self.k, self.batch_size, self.num_data)
        inputs_train, inputs_eval = self._split_data(total_inputs_tensor, SPLIT_IDX['batch_idx'], SPLIT_IDX['eval_idx'])
        outputs_train, outputs_eval = self._split_data(total_outputs_tensor, SPLIT_IDX['batch_idx'], SPLIT_IDX['eval_idx'])
        data = {
            'inputs_train': inputs_train,
            'outputs_train': outputs_train,
            'inputs_eval': inputs_eval,
            'outputs_eval': outputs_eval
        }
        return data

class DataProcess():
    """Prepare the inputs and outputs (labels) for the neural network
    """
    def __init__(self, PARAMS: dict) -> None:
        self.parent_dir = fcs.get_parent_path(lvl=1)
        self.path_data = os.path.join(self.parent_dir, 'data', 'pretraining')
        self.initialization(PARAMS)

    def initialization(self, PARAMS: dict):
        """Reset the class
        """
        self.input_name = PARAMS['input_name']
        self.output_name = PARAMS['output_name']

        if PARAMS['data_format'] == 'seq2seq':
            self._DATA_PROCESS = DataSeq(PARAMS)
        elif PARAMS['data_format'] == 'win2win':
            self._DATA_PROCESS = DataWin(PARAMS)
        else:
            raise ValueError(f'The specified data generation type does not exist!')

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
        
    def get_data(self, mode: str):
        """Return the inputs and outputs (labels) for the neural networks
        
        parameters:
        ----------- 
        mode: get offline training data or online training data 
        """
        if mode == 'offline':
            raw_inputs, raw_outputs = self.read_raw_data(self.input_name, self.output_name)
            self._DATA_PROCESS.import_data(raw_inputs, raw_outputs)
            return self._DATA_PROCESS.generate_data()
        
        elif mode == 'online':
            pass

        else:
            raise ValueError(f"The given data mode does not exist!")





