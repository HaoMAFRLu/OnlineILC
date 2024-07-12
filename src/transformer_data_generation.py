import numpy as np
import sys, os
import pickle
import math
import random
import torch

import utils as fcs

def read_data():
    parent_dir = fcs.get_parent_path(lvl=1)
    path_data = os.path.join(parent_dir, 'data', 'pretraining')
    path = os.path.join(path_data, 'keys')
    
    with open(path, 'rb') as file:
        keys = pickle.load(file)
    
    path = os.path.join(path_data, 'ilc')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    
    src = data[keys.index('yref')]
    tgt = data[keys.index('d')]
    target = data[keys.index('u')]

    return src, tgt, target

def select_idx(idx: list, num: int) -> list:
    """Select certain number of elements from the index list

    parameters:
    -----------
    idx: the given index list
    num: the number of to select elements
    """
    return random.sample(idx, num)

def select_batch_idx(idx: list, batch_size: int) -> list:
    """Split the index according to the given batch size

    parameters:
    -----------
    idx: the given index list
    batch_size: the batch size
    """
    batch_idx = []
    rest_idx = idx
    while len(rest_idx) > batch_size:
        _batch_idx = select_idx(rest_idx, batch_size)
        batch_idx.append(_batch_idx)
        rest_idx = list(set(rest_idx) - set(_batch_idx))
    
    if len(rest_idx) > 0:
        batch_idx.append(rest_idx)
    return batch_idx

def generate_split_idx(k, batch_size, num_data):
    num_train = math.floor(num_data*k)
    all_idx = list(range(num_data))

    train_idx = select_idx(all_idx, num_train)
    num_eval = num_data - num_train
    eval_idx = list(set(all_idx) - set(train_idx))
    batch_idx = select_batch_idx(train_idx, batch_size)

    SPLIT_IDX = {
        'num_train': num_train,
        'num_eval': num_eval,
        'all_idx': all_idx,
        'train_idx': train_idx,
        'eval_idx': eval_idx,
        'batch_idx': batch_idx
    }
    return SPLIT_IDX

def split_data(data, idx: list):
    """
    """
    return torch.cat([data[i] for i in idx], dim=1)

def _split_data(data, batch_idx: list, eval_idx: list):
    """
    """
    train = []
    eval = []
    eval.append(split_data(data, eval_idx))

    l = len(batch_idx)
    for i in range(l):
        train.append(split_data(data, batch_idx[i]))
    
    return train, eval

def get_max_value(data) -> float:
    """Return the maximum value
    """
    return np.max(np.concatenate(data))

def get_mean_value(data) -> float:
    """Return the mean value of the data
    """
    return np.mean(np.concatenate(data))

def get_min_value(data) -> float:
    """Return the minimum value
    """
    return np.min(np.concatenate(data))

def normalize(data, scale: float=1.0):
    """Map the data into [-1, 1]*scale
    """
    min_value = get_min_value(data)
    max_value = get_max_value(data)

    num_data = len(data)
    data_norm = [None] * num_data
    for i in range(num_data):
        data_norm[i] = (2*(data[i]-min_value)/(max_value-min_value) - 1) * scale
    
    mean = get_mean_value(data_norm)
    return data_norm - mean

def get_tensor_data(data, batch_size, device):
    """Convert data to tensor
    
    parameters:
    -----------
    data: the list of array

    returns:
    -------
    tensor_list: a list of tensors, which are in the shape of 1 x channel x height x width
    """
    tensor_list = [torch.tensor(arr, device=device).view(550, 1, 1) for arr in data]
    return tensor_list

def get_data():
    batch_size = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    src, tgt, target = read_data()
    SPLIT_IDX = generate_split_idx(k=0.8, batch_size=batch_size, num_data=len(src))
    _src = normalize(src, scale=1.0)
    _tgt = normalize(tgt, scale=1.0)
    _target = normalize(target, scale=1.0)
    src_tensor = get_tensor_data(_src, batch_size, device)
    tgt_tensor = get_tensor_data(_tgt, batch_size, device)
    target_tensor = get_tensor_data(_target, batch_size, device)
    src_train, src_eval = _split_data(src_tensor, SPLIT_IDX['batch_idx'], SPLIT_IDX['eval_idx'])
    tgt_train, tgt_eval = _split_data(tgt_tensor, SPLIT_IDX['batch_idx'], SPLIT_IDX['eval_idx'])
    target_train, target_eval = _split_data(target_tensor, SPLIT_IDX['batch_idx'], SPLIT_IDX['eval_idx'])
    data = {
        'inputs_train': src_train,
        'mid_train': tgt_train,
        'outputs_train': target_train,
        'inputs_eval': src_eval,
        'mid_eval': tgt_eval,
        'outputs_eval': target_eval
        }
    return data
