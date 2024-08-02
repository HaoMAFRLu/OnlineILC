"""Classes for online learning algorithm
"""
import torch
from pathlib import Path
import os, sys
import importlib
from typing import Tuple, List

import utils as fcs

import networks
import data_process
import params
import environmnet
from trajectory import TRAJ

class OnlineLearning():
    """Classes for online learning
    """
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.root = fcs.get_parent_path(lvl=0)
        self.initialization()

    def build_model(self) -> torch.nn:
        """Build a new model, if want to learn from scratch
        """
        pass

    def reload_module(self, path: Path) -> None:
        """Reload modules from the specified path
        
        parameters:
        -----------
        path: path to the src folder
        """
        sys.path.insert(0, path)
        importlib.reload(networks)
        importlib.reload(data_process)
        importlib.reload(params)

    @staticmethod
    def get_params(path: Path) -> Tuple[dict]:
        """Return the hyperparameters for each module

        parameters:
        -----------
        path: path to folder of the config file
        """
        PATH_CONFIG = os.path.join(path, 'config.json')
        PARAMS_LIST = ["SIM_PARAMS", 
                       "OFFLINE_DATA_PARAMS", 
                       "NN_PARAMS"]
        
        params_generator = params.PARAMS_GENERATOR(PATH_CONFIG)
        params_generator.get_params(PARAMS_LIST)
        return (params_generator.PARAMS['SIM_PARAMS'],
                params_generator.PARAMS['OFFLINE_DATA_PARAMS'],
                params_generator.PARAMS['NN_PARAMS'])

    def env_initialization(self, PARAMS: dict) -> environmnet:
        """Initialize the simulation environment
        """
        self.env = environmnet.BEAM('Control_System', PARAMS)
        self.env.initialization()

    def data_process_initialization(self, PARAMS: dict) -> None:
        """Initialize the data processor

        parameters:
        -----------
        PARAMS: hyperparameters
        """
        self.DATA_PROCESS = data_process.DataProcess('online', PARAMS)
    
    def NN_initialization(self, path: Path, PARAMS: dict) -> None:
        """Build the model and load the pretrained weights
        """
        self.model = networks.NETWORK_CNN(self.device, PARAMS)
        self.model.build_network()
        checkpoint = torch.load(path)
        self.model.NN.load_state_dict(checkpoint['model_state_dict'])
    
    def traj_initialization(self) -> None:
        """Create the class of reference trajectories
        """
        self.traj = TRAJ()

    def initialization(self) -> torch.nn:
        """Initialize everything:
        (0. reload the module from another src path, and load the weights)
        1. generate parameters for each module
            |-- SIM_PARAMS: parameters for initializing the simulation
            |-- DATA_PARAMS: parameters for initializing the online data processor
            |-- NN_PARAMS: parameters for initializing the neural network
        2. load and initialize the simulation environment
        3. load and initialize the data process
        4. build and load the pretrained neural network
        """
        SIM_PARAMS, DATA_PARAMS, NN_PARAMS = self.get_params(self.root)
        self.traj_initialization()
        self.env_initialization(SIM_PARAMS)
        self.data_process_initialization(DATA_PARAMS)
        
        path = os.path.join(self.root, 'data', 'pretrain_model', 'model.pth')
        self.NN_initialization(path, NN_PARAMS)
    
    def online_learning(self, nr_iterations: int=100) -> None:
        """Online learning.
        1. sample a reference trajectory randomly
        2. do the inference using the neural network -> u
        3. execute the simulation and observe the loss
        4. update the last layer using kalman filter
        """
        # sample a reference trajectory
        for i in range(nr_iterations):
            # sample a reference trajectory
            yref, t_stamp = self.traj.get_traj()
            y_processed = self.DATA_PROCESS.get_data(raw_inputs=yref[0, 1:])
            self.model.NN.eval()
            d_tensor = self.model.NN(y_processed.float())
            d = d_tensor.squeeze().to('cpu').detach().numpy()
            
            yout = self.env(d)
            # self.extract_last_layer()
            # self.kalman_filter(self.model.NN, yref, yout)
            # self.change_last_layer()


