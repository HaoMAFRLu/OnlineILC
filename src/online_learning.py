"""Classes for online learning algorithm
"""
import torch
from pathlib import Path
import os, sys
import importlib
from typing import Tuple, List
import pickle
import numpy as np
import numba as nb
from datetime import datetime

import utils as fcs
from mytypes import Array, Array2D, Array3D

import networks
import data_process
import params
import environmnet
from trajectory import TRAJ
from kalman_filter import KalmanFilter


second_linear_output = []

class OnlineLearning():
    """Classes for online learning
    """
    def __init__(self, nr_interval: int=100) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.root = fcs.get_parent_path(lvl=0)
        self.nr_interval = nr_interval

        current_time = datetime.now()
        folder_name = current_time.strftime("%Y%m%d_%H%M%S")
        self.path_model = os.path.join(self.root, 'data', 'online_training')
        fcs.mkdir(self.path_model)

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
    
    def hook(module, input, output):
        """Get the intermediate value in the neural network
        """
        second_linear_output.append(output)

    def NN_initialization(self, path: Path, PARAMS: dict) -> None:
        """Build the model and load the pretrained weights
        """
        self.model = networks.NETWORK_CNN(self.device, PARAMS)
        self.model.build_network()
        checkpoint = torch.load(path)
        self.model.NN.load_state_dict(checkpoint['model_state_dict'])
        hook_handle = self.model.NN.fc[2].register_forward_hook(self.hook)
    
    def traj_initialization(self) -> None:
        """Create the class of reference trajectories
        """
        self.traj = TRAJ()

    def load_dynamic_model(self) -> None:
        """Load the dynamic model of the underlying system,
        including the matrices B and Bd
        """
        path_file = os.path.join(self.root, 'data', 'linear_model', 'linear_model')
        with open(path_file, 'rb') as file:
            _data = pickle.load(file)
        
        self.B = _data[0]
        self.Bd = _data[1]
    
    def kalman_filter_initialization(self, PARAMS: dict) -> None:
        """Initialize the kalman filter
        """
        self.kalman_filter = KalmanFilter(PARAMS)

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
        self.load_dynamic_model()
        SIM_PARAMS, DATA_PARAMS, NN_PARAMS, KF_PARAMS = self.get_params(self.root)
        self.traj_initialization()
        self.env_initialization(SIM_PARAMS)
        self.data_process_initialization(DATA_PARAMS)
        
        path = os.path.join(self.root, 'data', 'pretrain_model', 'model.pth')
        self.NN_initialization(path, NN_PARAMS)

        self.kalman_filter_initialization(KF_PARAMS)
    
    @nb.jit(nopython=True)
    def get_u(self, y: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Get the input u based on the disturbance
        """
        return np.linalg.inv(self.B)@(y-self.Bd@d)

    @staticmethod
    def get_loss(y1: np.ndarray, y2: np.ndarray) -> float:
        """Calculate the loss
        """
        return 0.5*np.linalg.norm(y1-y2)
        
    @staticmethod
    def tensor2np(a: torch.tensor) -> Array:
        """Covnert tensor to numpy
        """
        return a.squeeze().to('cpu').detach().numpy()

    def extract_NN_info(self, NN: torch.nn) -> Tuple[Array, Array]:
        """Extract the infomation of the neural network

        parameters:
        -----------
        NN: the given neural network

        returns:
        --------
        c: the output of the second last layer
        E: the column vector of the parameters of the last layer,
           including the bias
        """
        c_tensor = second_linear_output[-1]
        c = self.tensor2np(c_tensor)
        
        last_layer = NN.fc[-1]
        weights = last_layer.weight.data
        bias = last_layer.bias.data

        E = np.stack((weights, bias), axis=-1)
        return c, E

    def assign_last_layer(self, NN: torch.nn, value: Array) -> None:
        """Assign the value of the last layer of the neural network.
        """
        last_layer = NN.fc[-1]
        with torch.no_grad():
            last_layer.weight.copy_(value)
            last_layer.bias.copy_(value)

    def save_checkpoint(self, idx: int) -> None:
        """Save the model
        """
        pass

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
            yref, _ = self.traj.get_traj()
            y_processed = self.DATA_PROCESS.get_data(raw_inputs=yref[0, 1:])
            self.model.NN.eval()
            d_tensor = self.model.NN(y_processed.float())
            d = self.tensor2np(d_tensor)            
            u = self.get_u(yref, d)
            yout, _ = self.env.one_step(u)
            loss = self.get_loss(yout, yref)
            c, E = self.extract_NN_info(self.model.NN)
            Ebar = self.kalman_filter(c, E, yref, yout)
            self.assign_last_layer(self.model.NN, Ebar)
            
            print('loss')

            if (i+1) % self.nr_interval == 0:
                self.save_checkpoint(i+1)