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
import time

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
    def __init__(self, mode: str=None, 
                 nr_interval: int=100,
                 nr_data_interval: int=1) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.root = fcs.get_parent_path(lvl=0)
        
        self.nr_interval = nr_interval
        self.nr_data_interval = nr_data_interval
        self.mode = mode
        
        parent = fcs.get_parent_path(lvl=1)
        current_time = datetime.now()
        folder_name = current_time.strftime("%Y%m%d_%H%M%S")
        self.path_model = os.path.join(parent, 'data', 'online_training', folder_name)
        self.path_data = os.path.join(self.path_model, 'data')
        fcs.mkdir(self.path_model)
        fcs.mkdir(self.path_data)

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
                       "NN_PARAMS",
                       "KF_PARAMS"]
        
        params_generator = params.PARAMS_GENERATOR(PATH_CONFIG)
        params_generator.get_params(PARAMS_LIST)
        return (params_generator.PARAMS['SIM_PARAMS'],
                params_generator.PARAMS['OFFLINE_DATA_PARAMS'],
                params_generator.PARAMS['NN_PARAMS'],
                params_generator.PARAMS['KF_PARAMS'])

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
        
        def hook(module, input, output):
            """Get the intermediate value in the neural network
            """
            second_linear_output.append(output)

        self.model = networks.NETWORK_CNN(self.device, PARAMS)
        self.model.build_network()
        checkpoint = torch.load(path)
        self.model.NN.load_state_dict(checkpoint['model_state_dict'])
        hook_handle = self.model.NN.fc[2].register_forward_hook(hook)
    
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
    
    def kalman_filter_initialization(self, mode: str,
                                     PARAMS: dict) -> None:
        """Initialize the kalman filter
        """
        self.kalman_filter = KalmanFilter(mode, self.B, self.Bd, PARAMS)

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

        self.kalman_filter_initialization(self.mode, KF_PARAMS)
    
    # @nb.jit(nopython=True)
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
    
    def np2tensor(self, a: Array) -> torch.tensor:
        """Covnert numpy to tensor
        """        
        a_tensor = torch.from_numpy(a).to(self.device)
        return a_tensor
    
    def extract_last_layer(self, NN: torch.nn) -> Tuple[Array2D, Array]:
        """Return the parameters of the last layer, including
        the weights and bis
        """
        last_layer = NN.fc[-1]
        w_tensor = last_layer.weight.data
        b_tensor = last_layer.bias.data

        w = self.tensor2np(w_tensor)
        b = self.tensor2np(b_tensor)
        return w, b
    
    def extract_output(self, NN: torch.nn) -> Array:
        """Extract the ouput of the last second layer
        """
        phi_tensor = second_linear_output[-1]
        return self.tensor2np(phi_tensor)

    def extract_NN_info(self, NN: torch.nn) -> Tuple[Array, Array]:
        """Extract the infomation of the neural network

        parameters:
        -----------
        NN: the given neural network

        returns:
        --------
        phi: the output of the second last layer
        vec: the column vector of the parameters of the last layer,
           including the bias
        """
        phi = self.extract_output(NN).reshape(-1, 1)
        w_, b_ = self.extract_last_layer(NN)

        w = fcs.get_flatten(w_).reshape(-1, 1)
        b = b_.reshape(-1, 1)
        vec = np.vstack((w, b))
        
        return phi, vec

    def assign_last_layer(self, NN: torch.nn, value: Array) -> None:
        """Assign the value of the last layer of the neural network.
        """
        last_layer = NN.fc[-1]
        
        w = fcs.get_unflatten(value[0:last_layer.weight.numel()].flatten(), 550)
        b = value[last_layer.weight.numel():].flatten()

        w_tensor = self.np2tensor(w)    
        b_tensor = self.np2tensor(b)

        with torch.no_grad():
            last_layer.weight.copy_(w_tensor)
            last_layer.bias.copy_(b_tensor)

    def save_checkpoint(self, idx: int) -> None:
        """Save the model
        """
        checkpoint = {
            'epoch': idx,
            'model_state_dict': self.model.NN.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict()
        }
        path_checkpoint = os.path.join(self.path_model, f'checkpoint_epoch_{idx}.pth')
        torch.save(checkpoint, path_checkpoint)

    def save_data(self, idx: int, **kwargs) -> None:
        """Save the data
        """
        path_data = os.path.join(self.path_data, str(idx))
        with open(path_data, 'wb') as file:
            pickle.dump(kwargs, file)
        
    def get_svd(self, A: Array2D) -> Tuple[Array2D, Array2D, Array2D]:
        """Do the SVD
        """
        return np.linalg.svd(A)
    
    @staticmethod
    def svd_inference(U: Array2D, S: Array, VT: Array2D) -> Array2D:
        """Return the original matrix
        """
        l = U.shape[0]
        r = VT.shape[0]
        I = np.zeros((l-r, r))
        K = np.vstack((np.diag(S.flatten()), I))
        return U@K@VT
    
    def online_learning(self, nr_iterations: int=100) -> None:
        """
        """
        if self.mode is None:
            self._online_learning(nr_iterations)
        elif self.mode == 'svd':
            self._online_learning_svd(nr_iterations)

    def _online_learning(self, nr_iterations: int=100) -> None:
        """Online learning.
        1. sample a reference trajectory randomly
        2. do the inference using the neural network -> u
        3. execute the simulation and observe the loss
        4. update the last layer using kalman filter
        """
        self.model.NN.eval()

        for i in range(nr_iterations):
            tt = time.time()

            yref, _ = self.traj.get_traj()
            y_processed = self.DATA_PROCESS.get_data(raw_inputs=yref[0, 1:])

            d_tensor = self.model.NN(y_processed.float())
            d = self.tensor2np(d_tensor)
            d = self.DATA_PROCESS.inverse_output(d)

            u = self.get_u(yref[0, 1:].reshape(-1 ,1), d.reshape(-1, 1))
            t1 = time.time()
            yout, _ = self.env.one_step(u.flatten())
            tsim = time.time() - t1
            
            loss = self.get_loss(yout.flatten(), yref[0, 1:].flatten())
            phi, vec = self.extract_NN_info(self.model.NN)

            if self.kalman_filter.d is None:
                self.kalman_filter.import_d(vec)

            t1 = time.time()
            self.kalman_filter.get_A(phi)
            vec_, tk, td, tp = self.kalman_filter.estimate(yout, u)      
            self.assign_last_layer(self.model.NN, vec_)
            t2 = time.time()

            ttotal = time.time() - tt
            fcs.print_info(
                Epoch=[str(i+1)+'/'+str(nr_iterations)],
                Loss=[loss],
                Ttotal = [ttotal],
                Tsim = [tsim],
                # Tk=[tk],
                # Td=[td],
                # Tp=[tp]
            )

            if (i+1) % self.nr_interval == 0:
                self.save_checkpoint(i+1)

    def _online_learning_svd(self, nr_iterations: int=100) -> None:
        """
        """
        self.model.NN.eval()
        w, b = self.extract_last_layer(self.model.NN)
        G = np.hstack((w, b.reshape(-1, 1)))
        U, S, VT = self.get_svd(G)
        self.kalman_filter.get_Bd_bar(self.Bd, U)
        self.kalman_filter.VT = VT

        yref, _ = self.traj.get_traj()
        y_processed = self.DATA_PROCESS.get_data(raw_inputs=yref[0, 1:])

        for i in range(nr_iterations):
            tt = time.time()

            
            d_tensor = self.model.NN(y_processed.float())
            d = self.tensor2np(d_tensor)
            d = self.DATA_PROCESS.inverse_output(d)

            u = self.get_u(yref[0, 1:].reshape(-1 ,1), d.reshape(-1, 1))
            t1 = time.time()
            yout, _ = self.env.one_step(u.flatten())
            tsim = time.time() - t1

            loss = self.get_loss(yout.flatten(), yref[0, 1:].flatten())

            phi = self.extract_output(self.model.NN)  # get the output of the last second layer
            
            if self.kalman_filter.d is None:
                self.kalman_filter.import_d(S.reshape(-1, 1))

            t1 = time.time()
            self.kalman_filter.get_A(phi)
            S, tk, td, tp = self.kalman_filter.estimate(yout, u)      
            
            G = self.svd_inference(U, S, VT)
            vec = fcs.get_flatten(G)
            self.assign_last_layer(self.model.NN, vec)
            t2 = time.time()

            ttotal = time.time() - tt
            fcs.print_info(
                Epoch=[str(i+1)+'/'+str(nr_iterations)],
                Loss=[loss],
                Ttotal = [ttotal],
                Tsim = [tsim],
                Tk=[tk],
                Td=[td],
                Tp=[tp]
            )

            if (i+1) % self.nr_data_interval == 0:
                self.save_data(i, 
                               u=u,
                               yref=yref,
                               d=d,
                               yout=yout,
                               loss=loss)
            if (i+1) % self.nr_interval == 0:
                self.save_checkpoint(i+1)