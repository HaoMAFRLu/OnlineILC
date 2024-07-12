"""Visualization test
"""
import os, sys
import numpy as np
import random
import torch
from dataclasses import asdict

random.seed(10086)
torch.manual_seed(10086)

def test(path):
    sys.path.append(path+'src')
    from networks import NETWORK_CNN
    from data_process import DataProcess
    from params import PARAMS_GENERATOR, VISUAL_PARAMS
    from visualization import Visual

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PARAMS_LIST = ["OFFLINE_DATA_PARAMS",
                   "NN_PARAMS"]
    
    params_generator = PARAMS_GENERATOR()
    params_generator.get_params(PARAMS_LIST)

    DATA_PROCESS = DataProcess(params_generator.PARAMS['OFFLINE_DATA_PARAMS'])
    data = DATA_PROCESS.get_data('offline')
    
    model = NETWORK_CNN(device, params_generator.PARAMS['NN_PARAMS'])
    model.build_network()

    VIS_PARAMS = VISUAL_PARAMS(
        is_save=True,
        paths=["offline_training", "20240711_222843"],
        checkpoint="checkpoint_epoch_10000",
        data='train'
    )
    VISUAL = Visual(asdict(VIS_PARAMS))
    VISUAL.load_model(model=model.NN)
    VISUAL.plot_results(model.NN,
                        data['inputs_train'],
                        data['outputs_train'])    

if __name__ == '__main__':
    test("/home/hao/Desktop/MPI/Online_Convex_Optimization/OnlineILC/data/offline_training/20240711_222843")