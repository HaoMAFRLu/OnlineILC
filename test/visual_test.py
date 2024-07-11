"""Visualization test
"""
import os, sys
import numpy as np
from dataclasses import asdict
import random
import torch
import argparse
import time

random.seed(10086)
torch.manual_seed(10086)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from networks import NETWORK_CNN
from data_process import DataProcess
from params import *
from visualization import Visual

def test():
    # generate the data
    DATA_PARAMS = OFFLINE_DATA_PARAMS(
    mode='seq2seq',
    k=0.8,
    batch_size=20,
    input_name='u',
    output_name='yout',
    channel=1,
    height=550,
    width=1
    )

    DATA_PROCESS = DataProcess(**asdict(DATA_PARAMS))
    data = DATA_PROCESS.get_data('offline', is_normalization=True)
    
    # generate the model and load the pre-trained parameters
    PARAMS = NN_PARAMS(
        is_initialization=False,
        loss_function='Huber',
        lambda_regression=0.0,
        learning_rate=1e-4,
        weight_decay=0.0,
        channel=1,
        height=550,
        width=1,
        filter_size=5,
        output_dim=550
    )

    VIS_PARAMS = VISUAL_PARAMS()

    NN = NETWORK_CNN('gpu', asdict(PARAMS))
    VISUAL = Visual(VIS_PARAMS)
    VISUAL.load_model(model=NN)
    VISUAL.plot_results(NN, data, is_save=True)    


if __name__ == '__main__':
    test()