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
    def __init__(self, path: Path) -> None:
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
    def get_params() -> Tuple[dict]:
        """Return the hyperparameters for each module
        """
        PARAMS_LIST = ["SIM_PARAMS", 
                       "ONLINE_DATA_PARAMS", 
                       "NN_PARAMS",
                       "TRAJ_PARAMS"]
        params_generator = params.PARAMS_GENERATOR()
        params_generator.get_params(PARAMS_LIST)
        return (params_generator.PARAMS['SIM_PARAMS'],
                params_generator.PARAMS['ONLINE_DATA_PARAMS'],
                params_generator.PARAMS['NN_PARAMS'])

    def env_initialization(self, PARAMS: dict) -> environmnet:
        """Initialize the simulation environment
        """
        self.env = environmnet.BEAM('Control_System', PARAMS)
        self.env.initialization()

    def data_process_initialization(self, path: Path, 
                                    PARAMS: dict) -> None:
        """
        """
        self.DATA_PROCESS = data_process.DataProcess('online', PARAMS)
        data = self.DATA_PROCESS.get_data(root=path, raw_inputs=None)
        print('here')
    
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

    def initialization(self, path: Path) -> torch.nn:
        """Initialize everything:
        0. reload the module from another src path, and load the weights
        1. generate parameters for each module
        2. load and initialize the simulation environment
        3. load and initialize the data process
        4. build and load the pretrained neural network
        parameters:
        -----------
        path: path to the src folder
        """
        if path is None:
            checkpoint = None
        else:
            self.reload_module(path)
            checkpoint = fcs.load_model(path)

        SIM_PARAMS, DATA_PARAMS, NN_PARAMS, OL_PARAMS = self.get_params()
        self.traj_initialization()
        self.env_initialization(SIM_PARAMS)
        self.data_process_initialization(path, DATA_PARAMS)
        self.NN_initialization(path, NN_PARAMS)
    
    def online_learning(self, PARAMS: dict) -> None:
        """Online learning.
        """
        # sample a reference trajectory
        for i in range(PARAMS["nr_iterations"]):
            # sample a reference trajectory
            yref = self.traj.get_traj()
            y_processed = self.DATA_PROCESS.get_data(yref)
            self.model.NN.eval()
            u = self.model.NN(y_processed)
            yout = self.env(u)
            self.kalman_filter(self.model.NN, yref, yout)


