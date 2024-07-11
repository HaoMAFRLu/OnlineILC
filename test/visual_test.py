"""Visualization test
"""
import os, sys
import numpy as np
import random
import torch
import argparse
import time

random.seed(10086)
torch.manual_seed(10086)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from networks import NETWORK_CNN
from data_process import DataProcess
from params import PARAMS_GENERATOR
from visualization import Visual

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PARAMS_LIST = ["OFFLINE_DATA_PARAMS",
                   "NN_PARAMS",
                   "VISUAL_PARAMS"]
    
    params_generator = PARAMS_GENERATOR()
    params_generator.get_params(PARAMS_LIST)

    DATA_PROCESS = DataProcess(params_generator.PARAMS['OFFLINE_DATA_PARAMS'])
    data = DATA_PROCESS.get_data('offline', is_normalization=True)
    
    model = NETWORK_CNN(device, params_generator.PARAMS['NN_PARAMS'])
    model.build_network()

    VISUAL = Visual(params_generator.PARAMS['VISUAL_PARAMS'])
    VISUAL.load_model(model=model.NN)
    VISUAL.plot_results(model.NN,
                        data['inputs_eval'],
                        data['outputs_eval'])    

if __name__ == '__main__':
    test()