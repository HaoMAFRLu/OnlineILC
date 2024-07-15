"""Visualization test
"""
import os, sys
import numpy as np
import random
import torch
from dataclasses import asdict

random.seed(10086)
torch.manual_seed(10086)

def test():
    root = "/home/hao/Desktop/MPI/Online_Convex_Optimization/OnlineILC/data"
    folder = "offline_training"
    file = "20240715_125124"
    path = os.path.join(root, folder, file, 'src')
    # sys.path.insert(0, path)
    from networks import NETWORK_CNN
    from data_process import DataProcess
    from params import PARAMS_GENERATOR, VISUAL_PARAMS
    from visualization import Visual

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PARAMS_LIST = ["OFFLINE_DATA_PARAMS",
                   "NN_PARAMS"]
    
    params_generator = PARAMS_GENERATOR(os.path.join(path, 'config.json'))
    params_generator.get_params(PARAMS_LIST)

    DATA_PROCESS = DataProcess(params_generator.PARAMS['OFFLINE_DATA_PARAMS'])
    data = DATA_PROCESS.get_data('offline')
    
    model = NETWORK_CNN(device, params_generator.PARAMS['NN_PARAMS'])
    model.build_network()

    VIS_PARAMS = VISUAL_PARAMS(
        is_save=True,
        paths=[folder, file],
        checkpoint="checkpoint_epoch_10000",
        data='train'
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