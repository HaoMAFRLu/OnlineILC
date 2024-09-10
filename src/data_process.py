"""Process the data for training, including 
offline training and online training
"""
import numpy as np
import os
from typing import Tuple, List, Any
import pickle
import torch
import math
import random
from pathlib import Path
import matplotlib.pyplot as plt

from mytypes import *
import utils as fcs

class DataWin():
    """Generate windowed data

    parameters:
    -----------
    """
    def __init__(self, device: str, PARAMS: dict) -> None:
        self.device = device 
        self.channel = PARAMS['channel']
        self.height = PARAMS['height']
        self.width = PARAMS['width']
        self.h1 = PARAMS['h1']
        self.h2 = PARAMS['h2']

        self.l = self.h1 + self.h2 + 1
        self.is_normalization = PARAMS['is_normalization']
        self.is_centerization = PARAMS['is_centerization']
        self.input_scale = PARAMS['input_scale']
        self.output_scale = PARAMS['output_scale']
        

class DataSeq():
    """Generate sequential inputs and outputs

    parameters:
    -----------
    channel: channel dimension
    height: height dimension
    width: width dimension
    """
    def __init__(self, device: str, PARAMS: dict) -> None:
        self.device = device
        self.k = PARAMS['k']
        self.batch_size = PARAMS['batch_size']
        self.channel = PARAMS['channel']
        self.height = PARAMS['height']
        self.width = PARAMS['width']
        self.is_normalization = PARAMS['is_normalization']
        self.is_centerization = PARAMS['is_centerization']
        self.input_scale = PARAMS['input_scale']
        self.output_scale = PARAMS['output_scale']
    
    def preprocess_data(self, data: List[Array], **kwargs) -> List[Array]:
        """Do nothing
        """
        return data
    
    def inverse_preprocess(self, data: List[Array], **kwargs) -> List[Array]:
        """Do nothing
        """
        return data

    def generate_data(self, mode: str, **kwargs: Any):
        """Generate data: 
        offline: input_train, output_train, input_eval, output_eval
        online: just input

        parameters:
        -----------
        mode: offline or online
        """
        if mode == 'offline':
            return self.generate_offline_data(kwargs['inputs'], 
                                              kwargs['outputs'],
                                              kwargs['SPLIT_IDX'])
        elif mode == 'online':
            return self.generate_online_data(kwargs['inputs'],
                                             kwargs['norm_params'])

    @staticmethod
    def inverse_CNS(data: List[Array], preprocess: str, 
                    **kwargs: float) -> List[Array]:
        """inverse [C]enterize/[N]ormalize/[S]calize the input data
        """
        def _inverse_CNS(_data):
            if preprocess == 'C':
                return _data + kwargs['mean']
            elif preprocess == 'N':
                return (_data + 1) * (kwargs['max_value']-kwargs['min_value'])/2 + kwargs['min_value']
            elif preprocess == 'S':
                return _data/kwargs['scale']
        
        if isinstance(data, list):
            num = len(data)
            processed_data = [None]*num
            for i in range(num):
                processed_data[i] = _inverse_CNS(data[i])
        elif isinstance(data, np.ndarray):
            processed_data = _inverse_CNS(data)
        else:
            raise ValueError("Unsupported data type. Expected list or numpy array.")
        return processed_data
    
    def inverse_online_data(self, scale_outputs: Array, norm_params: dict) -> Array:
        """Inverse the normalization process
        """
        norm_outputs = self.inverse_CNS(scale_outputs, 'S', scale=norm_params['output_scale'])

        if self.is_normalization is True:
            center_outputs = self.inverse_CNS(norm_outputs, 'N', 
                                      min_value=norm_params['output_min'], 
                                      max_value=norm_params['output_max']) 
        else:
            center_outputs = norm_outputs.copy()

        if self.is_centerization is True:
            outputs = self.inverse_CNS(center_outputs, 'C', mean=norm_params['input_mean'])
        else:
            outputs = center_outputs.copy()

        return outputs

    def generate_online_data(self, inputs: Array, 
                             norm_params: dict) -> Array:
        """Genearate data for online training
        """
        if self.is_centerization is True:
            center_inputs = self.CNS(inputs, 'C', mean=norm_params['input_mean'])
        else:
            center_inputs = inputs.copy()
        
        if self.is_normalization is True:
            norm_inputs = self.CNS(center_inputs, 'N', 
                                    min_value=norm_params['input_min'], 
                                    max_value=norm_params['input_max']) 
        else:
            norm_inputs = center_inputs.copy()     

        scale_inputs = self.CNS(norm_inputs, 'S', scale=norm_params['input_scale'])

        inputs_tensor = self.get_tensor_data(scale_inputs)
        return inputs_tensor

    def generate_offline_data(self, inputs: List[Array],
                              outputs: List[Array],
                              SPLIT_IDX: dict) -> Tuple:
        """Prepare the data for offline training
        1. get the mean value of training inputs and outputs
        2. centerize all the inputs and outputs
        3. get the min and max values of (centerized) training inputs and outputs
        4. normalize all the (centerized) inputs and outputs
        5. scalize all the (normalized) inputs and outputs
        6. save the preprocess parameters
        """
        input_mean = self.get_mean_value([inputs[i] for i in SPLIT_IDX['train_idx']])
        output_mean = self.get_mean_value([outputs[i] for i in SPLIT_IDX['train_idx']])
        
        if self.is_centerization is True:
            center_inputs = self.CNS(inputs, 'C', mean=input_mean)
            center_outputs = self.CNS(outputs, 'C', mean=output_mean)
        else:
            center_inputs = inputs.copy()
            center_outputs = outputs.copy()

        input_min = self.get_min_value([center_inputs[i] for i in SPLIT_IDX['train_idx']])
        input_max = self.get_max_value([center_inputs[i] for i in SPLIT_IDX['train_idx']])
        output_min = self.get_min_value([center_outputs[i] for i in SPLIT_IDX['train_idx']])
        output_max = self.get_max_value([center_outputs[i] for i in SPLIT_IDX['train_idx']])
        
        if self.is_normalization is True:
            norm_inputs = self.CNS(center_inputs, 'N', 
                                    min_value=input_min, 
                                    max_value=input_max)    
            norm_outputs = self.CNS(center_outputs, 'N', 
                                    min_value=output_min, 
                                    max_value=output_max)
        else:
            norm_inputs = center_inputs.copy()
            norm_outputs = center_outputs.copy()

        scale_inputs = self.CNS(norm_inputs, 'S', scale=self.input_scale)
        scale_outputs = self.CNS(norm_outputs, 'S', scale=self.output_scale)

        norm_params = {
            "input_mean":   input_mean,
            "input_min":    input_min,
            "input_max":    input_max,
            "input_scale":  self.input_scale,
            "output_mean":  output_mean,
            "output_min":   output_min,
            "output_max":   output_max,
            "output_scale": self.output_scale
        }

        total_inputs_tensor = self.get_tensor_data(scale_inputs)
        total_outputs_tensor = self.get_tensor_data(scale_outputs)

        inputs_train, inputs_eval = self._split_data(total_inputs_tensor, 
                                                     SPLIT_IDX['batch_idx'], 
                                                     SPLIT_IDX['eval_idx'])
        
        outputs_train, outputs_eval = self._split_data(total_outputs_tensor, 
                                                       SPLIT_IDX['batch_idx'], 
                                                       SPLIT_IDX['eval_idx'])
        data = {
            'inputs_train': inputs_train,
            'outputs_train': outputs_train,
            'inputs_eval': inputs_eval,
            'outputs_eval': outputs_eval
        }
        return data, norm_params

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
    
    @staticmethod
    def CNS(data: List[Array], preprocess: str, 
            **kwargs: float) -> List[Array]:
        """[C]enterize/[N]ormalize/[S]calize the input data
        """
        def _CNS(_data):
            if preprocess == 'C':
                return _data - kwargs['mean']
            elif preprocess == 'N':
                return 2*(_data-kwargs['min_value'])/(kwargs['max_value']-kwargs['min_value']) - 1
            elif preprocess == 'S':
                return _data*kwargs['scale']
        
        if isinstance(data, list):
            num = len(data)
            processed_data = [None]*num
            for i in range(num):
                processed_data[i] = _CNS(data[i])
        elif isinstance(data, np.ndarray):
            processed_data = _CNS(data)
        else:
            raise ValueError("Unsupported data type. Expected list or numpy array.")
        return processed_data

    def get_tensor_data(self, data: List[Array]) -> List[torch.tensor]:
        """Convert data to tensor
        
        parameters:
        -----------
        data: the list of array

        returns:
        -------
        tensor_list: a list of tensors, which are in the shape of 1 x channel x height x width
        """
        if isinstance(data, list):
            tensor_list = [torch.tensor(arr, device=self.device).view(1, self.channel, self.height, self.width) for arr in data]
        elif isinstance(data, np.ndarray):
            tensor_list = torch.tensor(data, device=self.device).view(1, self.channel, self.height, self.width)
        return tensor_list

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

class DataProcess():
    """Prepare the inputs and outputs (labels) for the neural network
    PARAMS:
        |-- mode: offline or online data
        |-- data_format: seq2seq or win2win
        |-- input_name: name of the inputs
        |-- output_name: name of the outputs
    """
    def __init__(self, mode: str, PARAMS: dict) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.root = fcs.get_parent_path(lvl=0)
        self.mode = mode
        self.data_format = PARAMS['data_format']
        
        if self.mode == 'online':
            path_norm_params = os.path.join(self.root, 'data', 'pretrain_model', 'norm_params')
            with open(path_norm_params, 'rb') as file:
                _data = pickle.load(file)
                self.norm_params = _data["norm_params"]
        else:
            self.norm_params = None

        self.SPLIT_IDX = None

        self.PARAMS = PARAMS
        self.initialization()

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
    
    def get_idx(self, k: int, num_data: int) -> Tuple[list, list]:
        """Get training data indices and evaluation
        data indices
        """
        num_train = math.floor(num_data*k)
        all_idx = list(range(num_data))
        train_idx = self.select_idx(all_idx, num_train)
        eval_idx = list(set(all_idx) - set(train_idx))
        return all_idx, train_idx, eval_idx

    def get_SPLIT_IDX(self, k: int, num_data: int) -> dict:
        """
        """
        all_idx, train_idx, eval_idx = self.get_idx(k, num_data)
        batch_idx = self.select_batch_idx(train_idx, self.PARAMS['batch_size'])
        
        SPLIT_IDX = {
            'all_idx':   all_idx,
            'train_idx': train_idx,
            'eval_idx':  eval_idx,
            'batch_idx': batch_idx
        }
        return SPLIT_IDX
    
    @staticmethod
    def save_data(root: Path, 
                  file: str, **kwargs: Any) -> None:
        """save data
        """
        data_dict = kwargs
        path = os.path.join(root, file)
        with open(path, 'wb') as file:
            pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _offline_init(self) -> None:
        """Initialize the data process in an offline
        manner. The offline data is used for offline
        training.
        1. load the key
        2. load the data
        3. split data into training dataset and evaluation dataset
        4. split training dataset into mini batch
        5. save the data and indices
        """
        input_name = self.PARAMS['input_name']
        output_name = self.PARAMS['output_name']
        self.path_data = os.path.join(self.root, 'data', 'pretraining')
        
        keys = self._load_keys()
        data = self._load_data()
        
        raw_inputs = data[keys.index(input_name)]
        raw_outputs = data[keys.index(output_name)]
        return raw_inputs, raw_outputs
    
    def initialization(self):
        """Initialize the data process
        1. load the data processor
        2. preprocess the data if necessary
        """
        if self.PARAMS['data_format'] == 'seq2seq':
            self._DATA_PROCESS = DataSeq(self.device, self.PARAMS)
        elif self.PARAMS['data_format'] == 'win2win':
            self._DATA_PROCESS = DataWin(self.device, self.PARAMS)
        else:
            raise ValueError(f'The specified data format does not exist!')
        
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

    def import_data(self, raw_inputs: Array) -> None:
        """Import the online data
        """
        self.raw_inputs = raw_inputs.copy()

    def inverse_output(self, scale_outputs: Array) -> Array:
        """Inverse the output data
        """
        preprocess_outputs = self._DATA_PROCESS.inverse_preprocess(scale_outputs, target='output')
        return self._DATA_PROCESS.inverse_online_data(preprocess_outputs, self.norm_params)
 
    def get_data(self, **kwargs):
        """Return the inputs and outputs (labels) for the neural networks
        
        parameters:
        ----------- 
        mode: get offline training data or online training data 
        """
        if self.mode == 'offline':
            raw_inputs, raw_outputs = self._offline_init()
            preprocess_inputs = self._DATA_PROCESS.preprocess_data(raw_inputs, target='input')
            preprocess_outputs = self._DATA_PROCESS.preprocess_data(raw_outputs, target='output')

            if self.SPLIT_IDX is None:
                self.SPLIT_IDX = self.get_SPLIT_IDX(self.PARAMS['k'], len(preprocess_inputs))
        
            data, norm_params = self._DATA_PROCESS.generate_data('offline', inputs=preprocess_inputs, 
                                                                outputs=preprocess_outputs,
                                                                SPLIT_IDX=self.SPLIT_IDX)
            
            self.save_data(self.path_data, 'SPLIT_DATA', raw_inputs=raw_inputs,
                           raw_outputs=raw_outputs, SPLIT_IDX=self.SPLIT_IDX)
            
            self.save_data(self.path_data, 'norm_params', norm_params=norm_params)
            return data
            
        elif self.mode == 'online':
            preprocess_inputs = self._DATA_PROCESS.preprocess_data(kwargs['raw_inputs'], target='input')
            # if self.norm_params is None:
            #     self.norm_params = self.load_norm_params(kwargs['root'])
            data = self._DATA_PROCESS.generate_data('online', inputs=preprocess_inputs,
                                                                 norm_params=self.norm_params)
            return data
        
        else:
            raise ValueError(f'The specified data mode does not exist!')

    def load_norm_params(self, root: Path) -> dict:
        """Load the norm parameters

        parameters:
        -----------
        root: path to the src folder

        returns:
        --------
        norm_params: parameters used for normalizetion
        """
        path = os.path.join(root, 'data', 'pretraining', 'src')
        with open(path, 'rb') as file:
            norm_params = pickle.load(file)
        return norm_params
            

