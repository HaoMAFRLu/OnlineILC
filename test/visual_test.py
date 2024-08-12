"""Visualization test
"""
import os, sys
import numpy as np
import random
import torch
from dataclasses import asdict
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from visualization import Visual

import networks
import data_process
import params


random.seed(10086)
torch.manual_seed(10086)

def test():
    root = "/home/hao/Desktop/MPI/Online_Convex_Optimization/OnlineILC/data"
    folder = "offline_training"
    file = "medium_plus"
    path = os.path.join(root, folder, file, 'src')
    sys.path.insert(0, path)
    importlib.reload(params)
    importlib.reload(networks)
    importlib.reload(data_process)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PARAMS_LIST = ["OFFLINE_DATA_PARAMS",
                   "NN_PARAMS"]
    
    params_generator = params.PARAMS_GENERATOR(os.path.join(path, 'config.json'))
    params_generator.get_params(PARAMS_LIST)

    DATA_PROCESS = data_process.DataProcess('offline', params_generator.PARAMS['OFFLINE_DATA_PARAMS'])
    data = DATA_PROCESS.get_data()
    
    model = networks.NETWORK_CNN(device, params_generator.PARAMS['NN_PARAMS'])
    model.build_network()

    VIS_PARAMS = params.VISUAL_PARAMS(
        is_save=True,
        paths=[folder, file],
        checkpoint="checkpoint_epoch_5000",
        data='eval'
    )
    VISUAL = Visual(asdict(VIS_PARAMS))

    VISUAL.load_model(model=model.NN)
    loss_data = VISUAL.load_loss(VISUAL.path_loss)
    VISUAL.plot_loss(loss_data)
    VISUAL.plot_results(model.NN,
                        data['inputs_'+VIS_PARAMS.data],
                        data['outputs_'+VIS_PARAMS.data])    

if __name__ == '__main__':
    test()