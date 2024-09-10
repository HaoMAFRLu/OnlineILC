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
    def __init__(self, mode: str='full_states',
                 location:str='local',
                 rolling: int=1,
                 nr_interval: int=500,
                 nr_shift_dis: int=2,
                 nr_data_interval: int=1,
                 nr_marker_interval: int=20,
                 folder_name: str=None,
                 sigma_w=None,
                 sigma_y=None,
                 sigma_d=None,
                 sigma_ini=None) -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.root = fcs.get_parent_path(lvl=0)
        self.location = location
        self.rolling = rolling
        self.nr_interval = nr_interval
        self.nr_shift_dis = nr_shift_dis
        self.nr_data_interval = nr_data_interval
        self.nr_marker_interval = nr_marker_interval
        self.mode = mode

        parent = fcs.get_parent_path(lvl=1)

        if folder_name is None:
            current_time = datetime.now()
            folder_name = current_time.strftime('%Y%m%d_%H%M%S')
        
        self.path_model = os.path.join(parent, 'data', 'test', folder_name)
        self.path_data = os.path.join(self.path_model, 'data')

        fcs.mkdir(self.path_model)
        fcs.mkdir(self.path_data)

        parent_dir = fcs.get_parent_path(lvl=1)
        fcs.copy_folder(os.path.join(parent_dir, 'src'), self.path_model)
        fcs.copy_folder(os.path.join(parent_dir, 'test'), self.path_model)
        
        self.sigma_w = sigma_w
        self.sigma_y = sigma_y
        self.sigma_d = sigma_d
        self.sigma_ini = sigma_ini
        
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
    
    def traj_initialization(self, distribution: str='original') -> None:
        """Create the class of reference trajectories
        """
        self.traj = TRAJ(distribution)

    def load_dynamic_model(self) -> None:
        """Load the dynamic model of the underlying system,
        including the matrices B and Bd
        """
        path_file = os.path.join(self.root, 'data', 'linear_model', 'linear_model')
        with open(path_file, 'rb') as file:
            _data = pickle.load(file)
        
        self.B = _data[0]
        self.Bd = _data[1]
        self.inv_B = np.linalg.inv(self.B)
        self.pinv_B = np.linalg.pinv(self.B)
    
    def kalman_filter_initialization(self, PARAMS: dict) -> None:
        """Initialize the kalman filter
        """
        self.kalman_filter = KalmanFilter(mode=self.mode, B=self.B, Bd=self.Bd, 
                                          PARAMS=PARAMS, rolling=self.rolling, 
                                          sigma_w=self.sigma_w, 
                                          sigma_y=self.sigma_y, 
                                          sigma_d=self.sigma_d, 
                                          sigma_ini=self.sigma_ini,
                                          location=self.location)

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

        self.kalman_filter_initialization(PARAMS=KF_PARAMS)
    
    # @nb.jit(nopython=True)
    def get_u(self, y: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Get the input u based on the disturbance
        """
        return self.inv_B@(y-self.Bd@d)

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
    
    def extract_last_layer(self, NN: torch.nn) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract the last layer of the neural network
        """
        last_layer = NN.fc[-1]
        return last_layer.weight.data, last_layer.bias.data

    def extract_last_layer_vec(self, NN: torch.nn) -> torch.Tensor:
        """Extract the last layer and vectorize them
        """
        w, b = self.extract_last_layer(NN)
        return torch.cat((w.t().flatten(), b.flatten()), dim=0).view(-1, 1)
    
    # def extract_last_layer(self, NN: torch.nn) -> Tuple[Array2D, Array]:
    #     """Return the parameters of the last layer, including
    #     the weights and bis
    #     """
    #     w_tensor, b_tensor = self.extract_last_layer_tensor(NN)
    #     w = self.tensor2np(w_tensor)
    #     b = self.tensor2np(b_tensor)
    #     return w, b
    
    def extract_output(self) -> torch.Tensor:
        """
        """
        return second_linear_output[-1]

    # def extract_output(self) -> Array:
    #     """Extract the ouput of the last second layer
    #     """
    #     return self.tensor2np(self.extract_output_tensor())

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
        vec = self.extract_last_layer_vec(NN)
        phi = self.extract_output()        
        return phi, vec

    def _recover_last_layer(self, value: torch.Tensor, num: int) -> None:
        """
        """
        w = value[0:num].view(-1, 550).t()
        b = value[num:].flatten()
        return w, b        

    def assign_last_layer(self, NN: torch.nn, value: torch.Tensor) -> None:
        """Assign the value of the last layer of the neural network.
        """
        last_layer = NN.fc[-1]
        num = last_layer.weight.numel()
        w, b = self._recover_last_layer(value, num)
        
        with torch.no_grad():
            last_layer.weight.copy_(w)
            last_layer.bias.copy_(b)

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
        
    def get_svd(self, A: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """Do the SVD in pytorch
        """
        return torch.linalg.svd(A)
    
    def svd_inference(self, U: Array2D, S: Array, VT: Array2D) -> Array2D:
        """Return the original matrix
        """
        l = U.shape[0]
        r = VT.shape[0]
        if l>r:
            I = torch.zeros((l-r, r)).to(self.device).float()
            K = torch.vstack((torch.diag(S.flatten()), I))
        elif l<r:
            I = torch.zeros((l, r-l)).to(self.device).float()
            K = torch.hstack((np.diag(S), I))
        return torch.matmul(U, torch.matmul(K, VT))
    
    def _get_d(self, yref: Array) -> Array:
        """
        """
        y_processed = self.DATA_PROCESS.get_data(raw_inputs=yref[0, 1:])
        d_tensor = self.model.NN(y_processed.float())
        d = self.tensor2np(d_tensor)
        return self.DATA_PROCESS.inverse_output(d)

    def _rum_sim(self, yref: Array2D) -> Tuple:
        """
        """
        d = self._get_d(yref)
        u = self.get_u(yref[0, 1:].reshape(-1 ,1), d.reshape(-1, 1))
        yout, _ = self.env.one_step(u.flatten())
        loss = self.get_loss(yout.flatten(), yref[0, 1:].flatten())
        return yout, d, u, loss

    def online_learning(self, nr_iterations: int=100,
                        is_shift_dis: bool=False,
                        is_scratch: bool=False,
                        **kwargs) -> None:
        """
        """
        if self.mode == 'full_states':
            self._online_learning(nr_iterations, is_scratch, is_shift_dis)
        elif self.mode == 'svd':
            # self._online_learning_svd(nr_iterations, is_scratch, is_shift_dis)
            self._online_learning_shift_representation(nr_iterations)
        elif self.mode == 'svd_gradient':
            self._online_learning_gradient(nr_iterations, is_scratch,
                                           is_shift_dis,
                                           alpha=kwargs["alpha"],
                                           epsilon=kwargs["epsilon"],
                                           eta=kwargs["eta"])
        elif self.mode == 'representation':
            self._online_learning_representation(nr_iterations, is_shift_dis)

    def shift_distribution(self):
        """change the distribution
        """
        self.traj_initialization('shift')
        yref_marker, _ = self.traj.get_traj()
        return yref_marker

    def _online_learning_shift_representation(self, nr_iterations: int=100) -> None:
        """
        """
        self.traj = TRAJ('shift')
        new_element = torch.tensor([1]).to(self.device).float()
        path = '/home/hao/Desktop/MPI/Online_Convex_Optimization/OnlineILC/data/test/20240906_144508'
        path_file = os.path.join(path, 'data_hidden_states')
        with open(path_file, 'rb') as f:
            dphi_list = pickle.load(f)
        dphi = dphi_list[-1]

        def _rum_sim(yref: Array2D, W, dphi) -> Tuple:
            """
            """
            y_processed = self.DATA_PROCESS.get_data(raw_inputs=yref[0, 1:])
            _ = self.model.NN(y_processed.float())
            phi = self.extract_output() 
            phi_bar = torch.cat((phi.flatten(), new_element)).view(-1, 1)
            d = torch.matmul(W, phi_bar+dphi)/1000.0
            d = d.detach().cpu().numpy()
            u = self.get_u(yref[0, 1:].reshape(-1 ,1), d.reshape(-1, 1))
            yout, _ = self.env.one_step(u.flatten())
            loss = self.get_loss(yout.flatten(), yref[0, 1:].flatten())
            return yout, d, u, loss, phi_bar
        
        def run_marker_step(yref: Array2D, W, dphi, path: Path) -> None:
            """Evaluate the marker trajectory
            """
            self.nr_marker += 1
            yout, d, u, loss, phi_bar = _rum_sim(yref, W, dphi)
            self.loss_marker.append(np.round(loss, 4))
            fcs.print_info(
            Marker=[str(self.nr_marker)],
            Loss=[self.loss_marker[-6:]])

            path_marker_file = os.path.join(path, str(self.nr_marker))
            with open(path_marker_file, 'wb') as file:
                pickle.dump(yref, file)
                pickle.dump(yout, file)
                pickle.dump(d, file)
                pickle.dump(u, file)
                pickle.dump(loss, file)
            
        def get_L(phi, dphi):
            phi_bar = fcs.add_one(phi)
            phi_bar = phi_bar + dphi.flatten()

            v = self.VT_asnp@phi_bar.reshape(-1, 1)
            if self.dir == 'v':
                par_pi_par_omega = self.Bd_bar@np.vstack((np.diag(v.flatten()), self.padding))/1000.0
            elif self.dir == 'h':
                par_pi_par_omega = self.Bd_bar@np.diag(v.flatten()[:550])/1000.0
            return -par_pi_par_omega

        self.model.NN.eval()
        
        self.U, S, self.VT = self.kf_initialization(self.model.NN)

        self.U_asnp = self.U.cpu().numpy()
        self.VT_asnp = self.VT.cpu().numpy()

        dim = len(S)
        if dim < 550:
            self.padding = np.zeros((550-dim, dim))
            self.dir = 'v'
        elif dim >= 550:
            self.padding = None
            self.dir = 'h'

        self.Bd_bar = self.Bd@self.U_asnp

        self.kalman_filter.import_matrix(d=S.view(-1, 1))
        yref_marker, path_marker = self.marker_initialization()

        for i in range(nr_iterations):
            tt = time.time()

            self.NN_update(S)
            W = self.svd_inference(self.U, S, self.VT)

            if i%self.nr_marker_interval == 0:
                run_marker_step(yref_marker, W, dphi, path_marker)
            
            yref, _ = self.traj.get_traj()
            t1 = time.time()
            yout, d, u, loss, phi_bar = _rum_sim(yref, W, dphi)
            tsim = time.time() - t1
            phi = self.extract_output()  # get the output of the last second layer

            t1 = time.time()
            self.kalman_filter.get_A(phi, dphi=dphi)
            S, tk, td, tp = self.kalman_filter.estimate(yout, self.B@u.reshape(-1, 1))      
            t2 = time.time()

            L = get_L(phi.detach().cpu().numpy(), dphi.detach().cpu().numpy())
            gradient = L.T@(yout.reshape(-1, 1) - yref[0, 1:551].reshape(-1, 1))
            
            ttotal = time.time() - tt

            self.total_loss += loss
            fcs.print_info(
                Epoch=[str(i+1)+'/'+str(nr_iterations)],
                Loss=[loss],
                AvgLoss=[self.total_loss/(i+1)],
                Ttotal = [ttotal],
                Tsim = [tsim],
                Tk=[tk],
                Td=[td],
                Tp=[tp]
            )

            if (i+1) % self.nr_data_interval == 0:
                self.save_data(i, 
                               hidden_states=S,
                               u=u,
                               yref=yref,
                               d=d,
                               yout=yout,
                               loss=loss,
                               gradient=gradient.flatten())
                
            if (i+1) % self.nr_interval == 0:
                self.save_checkpoint(i+1)



    def _online_learning_representation(self, nr_iterations: int=100,
                             is_shift_dis: bool=False) -> None:
        """
        """
        self.traj = TRAJ('shift')
        new_element = torch.tensor([1]).to(self.device).float()

        def _rum_sim(yref: Array2D, W, dphi) -> Tuple:
            """
            """
            y_processed = self.DATA_PROCESS.get_data(raw_inputs=yref[0, 1:])
            _ = self.model.NN(y_processed.float())
            phi = self.extract_output() 
            phi_bar = torch.cat((phi.flatten(), new_element)).view(-1, 1)
            d = torch.matmul(W, phi_bar+dphi)/1000.0
            d = d.detach().cpu().numpy()
            u = self.get_u(yref[0, 1:].reshape(-1 ,1), d.reshape(-1, 1))
            yout, _ = self.env.one_step(u.flatten())
            loss = self.get_loss(yout.flatten(), yref[0, 1:].flatten())
            return yout, d, u, loss, phi_bar
        
        def run_marker_step(yref: Array2D, W, dphi, path: Path) -> None:
            """Evaluate the marker trajectory
            """
            self.nr_marker += 1
            yout, d, u, loss, phi_bar = _rum_sim(yref, W, dphi)
            self.loss_marker.append(np.round(loss, 4))
            fcs.print_info(
            Marker=[str(self.nr_marker)],
            Loss=[self.loss_marker[-6:]])

            path_marker_file = os.path.join(path, str(self.nr_marker))
            with open(path_marker_file, 'wb') as file:
                pickle.dump(yref, file)
                pickle.dump(yout, file)
                pickle.dump(d, file)
                pickle.dump(u, file)
                pickle.dump(loss, file)
    
        self.model.NN.eval()
        w, b = self.extract_last_layer(self.model.NN)
        W = torch.cat((w, b.view(-1, 1)), dim=1)  # the linear matrix of the last layer

        dphi = torch.zeros(65, 1).to(self.device).float()
        self.kalman_filter.import_matrix(d=dphi.view(-1, 1))
        yref_marker, path_marker = self.marker_initialization()

        for i in range(nr_iterations):
            tt = time.time()
            
            if (is_shift_dis is True) and (i > self.nr_shift_dis):
                is_shift_dis = False
                yref_marker = self.shift_distribution()

            if i%self.nr_marker_interval == 0:
                run_marker_step(yref_marker, W, dphi, path_marker)
            
            yref, _ = self.traj.get_traj()
            t1 = time.time()
            yout, d, u, loss, phi_bar = _rum_sim(yref, W, dphi)
            tsim = time.time() - t1

            t1 = time.time()
            self.kalman_filter.get_A(W)
            z = self.B@u.reshape(-1, 1) + self.Bd@torch.matmul(W, phi_bar).detach().cpu().numpy()/1000.0
            dphi, tk, td, tp = self.kalman_filter.estimate(yout, z)      
            t2 = time.time()

            L = -self.Bd@W.detach().cpu().numpy()/1000.0
            gradient = L.T@(yout.reshape(-1, 1) - yref[0, 1:551].reshape(-1, 1))
            
            ttotal = time.time() - tt

            self.total_loss += loss
            fcs.print_info(
                Epoch=[str(i+1)+'/'+str(nr_iterations)],
                Loss=[loss],
                AvgLoss=[self.total_loss/(i+1)],
                Ttotal = [ttotal],
                Tsim = [tsim],
                Tk=[tk],
                Td=[td],
                Tp=[tp]
            )

            if (i+1) % self.nr_data_interval == 0:
                self.save_data(i, 
                               hidden_states=dphi,
                               u=u,
                               yref=yref,
                               d=d,
                               yout=yout,
                               loss=loss,
                               gradient=gradient.flatten())
                
            if (i+1) % self.nr_interval == 0:
                self.save_checkpoint(i+1)


    def _online_learning(self, nr_iterations: int=100, 
                         is_scratch: bool=False, 
                         is_shift_dis: bool=False) -> None:
        """Online learning.
        1. sample a reference trajectory randomly
        2. do the inference using the neural network -> u
        3. execute the simulation and observe the loss
        4. update the last layer using kalman filter
        """
        self.model.NN.eval()
        yref_marker, path_marker = self.marker_initialization()
        vec = self.extract_last_layer_vec(self.model.NN)

        if is_scratch is True:
            vec = vec*0.0

        self.kalman_filter.import_matrix(d=vec.view(-1, 1))

        for i in range(nr_iterations):
            tt = time.time()

            if (is_shift_dis is True) and (i > self.nr_shift_dis):
                is_shift_dis = False
                yref_marker = self.shift_distribution()

            self.assign_last_layer(self.model.NN, vec)

            if i%self.nr_marker_interval == 0:
                self.run_marker_step(yref_marker, path_marker)

            t1 = time.time()
            yref, _ = self.traj.get_traj()
            yout, d, u, loss = self._rum_sim(yref)
            tsim = time.time() - t1

            phi, vec = self.extract_NN_info(self.model.NN)

            t1 = time.time()
            self.kalman_filter.get_A(phi)
            vec, tk, td, tp = self.kalman_filter.estimate(yout, self.B@u.reshape(-1, 1))
            t2 = time.time()

            gradient = -self.kalman_filter.A.detach().cpu().numpy().T@(yout.reshape(-1, 1) - yref[0, 1:551].reshape(-1, 1))

            ttotal = time.time() - tt
            self.total_loss += loss
            fcs.print_info(
                Epoch=[str(i+1)+'/'+str(nr_iterations)],
                Loss=[loss],
                AvgLoss=[self.total_loss/(i+1)],
                Umax=[np.max(np.abs(u))],
                Ttotal = [ttotal],
                Tsim = [tsim]
            )

            if (i+1) % self.nr_data_interval == 0:
                self.save_data(i,
                               u=u,
                               yref=yref,
                               d=d,
                               yout=yout,
                               loss=loss,
                               gradient=gradient.flatten())
                
            if (i+1) % self.nr_interval == 0:
                self.save_checkpoint(i+1)

    def kf_initialization(self, NN: torch.nn) -> None:
        """Initialize the kalman filter
        """
        w, b = self.extract_last_layer(NN)
        W = torch.cat((w, b.view(-1, 1)), dim=1)
        U, S, VT = self.get_svd(W)
        self.kalman_filter.import_matrix(U=U, VT=VT)
        self.kalman_filter.get_Bd_bar()
        return U, S, VT

    def marker_initialization(self) -> Tuple[Array2D, Path]:
        """Generate the marker trajectory and the 
        path to marker folder
        """
        self.nr_marker = 0
        self.loss_marker = []
        self.total_loss = 0
        yref_marker, _ = self.traj.get_traj()
        path_marker = os.path.join(self.path_model, 'loss_marker')
        fcs.mkdir(path_marker)
        return yref_marker, path_marker

    def run_marker_step(self, yref: Array2D, path: Path) -> None:
        """Evaluate the marker trajectory
        """
        self.nr_marker += 1
        yout, d, u, loss = self._rum_sim(yref)
        self.loss_marker.append(np.round(loss, 4))
        fcs.print_info(
        Marker=[str(self.nr_marker)],
        Loss=[self.loss_marker[-6:]])

        path_marker_file = os.path.join(path, str(self.nr_marker))
        with open(path_marker_file, 'wb') as file:
            pickle.dump(yref, file)
            pickle.dump(yout, file)
            pickle.dump(d, file)
            pickle.dump(u, file)
            pickle.dump(loss, file)

    def get_par_pi_par_omega(self, phi):
        phi_bar = fcs.add_one(phi)
        v = self.VT_asnp@phi_bar.reshape(-1, 1)
        if self.dir == 'v':
            par_pi_par_omega = self.Bd_bar@np.vstack((np.diag(v.flatten()), self.padding))/1000.0
        elif self.dir == 'h':
            par_pi_par_omega = self.Bd_bar@np.diag(v.flatten()[:550])/1000.0
        return par_pi_par_omega
    
    def get_L(self, par_pi_par_omega):
        return -par_pi_par_omega

    def get_Lambda(self, L, par_pi_par_omega, 
                   alpha: float=0.1, epsilon: float=0.001):
        I = np.eye(L.shape[1])
        return L.T@L + alpha*par_pi_par_omega.T@par_pi_par_omega + epsilon*I

    def update_A(self, A, Lambda):
        return A+Lambda
    
    def _online_learning_gradient(self, nr_iterations: int=100,
                                  is_scratch: bool=False,
                                  is_shift_dis: bool=False,
                                  alpha: float=0.1,
                                  epsilon: float=0.1,
                                  eta: float=0.3):
        """Online learning using quasi newton method
        """
        self.model.NN.eval()
        w, b = self.extract_last_layer(self.model.NN)
        W = torch.cat((w, b.view(-1, 1)), dim=1)
        self.U, S, self.VT = self.get_svd(W)
        self.U_asnp = self.U.cpu().numpy()
        self.VT_asnp = self.VT.cpu().numpy()

        if is_scratch is True:
            S = S*0.0

        dim = len(S)
        if dim < 550:
            self.padding = np.zeros((550-dim, dim))
            self.dir = 'v'
        elif dim >= 550:
            self.padding = None
            self.dir = 'h'
        A = np.zeros((dim, dim))

        self.Bd_bar = self.Bd@self.U_asnp

        yref_marker, path_marker = self.marker_initialization()

        for i in range(nr_iterations):
            tt = time.time()
            
            if (is_shift_dis is True) and (i > self.nr_shift_dis):
                is_shift_dis = False
                yref_marker = self.shift_distribution()

            self.NN_update(S)

            if i%self.nr_marker_interval == 0:
                self.run_marker_step(yref_marker, path_marker)
            
            yref, _ = self.traj.get_traj()
            t1 = time.time()
            yout, d, u, loss = self._rum_sim(yref)
            tsim = time.time() - t1
            phi = self.extract_output()
            
            par_pi_par_omega = self.get_par_pi_par_omega(phi.detach().cpu().numpy())
            L = self.get_L(par_pi_par_omega)
            Lambda = self.get_Lambda(L, par_pi_par_omega, alpha, epsilon)
            A = self.update_A(A, Lambda)
            gradient = L.T@(yout.reshape(-1, 1) - yref[0, 1:551].reshape(-1, 1))
            delta_S = eta*np.linalg.inv(A/(i+1))@gradient
            # delta_S = eta*L.T@(yout.reshape(-1, 1) - yref[0, 1:551].reshape(-1, 1))
            S = S - torch.from_numpy(delta_S).to(S.dtype).to('cuda:0').view(S.shape)

            ttotal = time.time() - tt
            self.total_loss += loss
            fcs.print_info(
                Epoch=[str(i+1)+'/'+str(nr_iterations)],
                Loss=[loss],
                AvgLoss=[self.total_loss/(i+1)],
                Ttotal = [ttotal],
                Tsim = [tsim],
                # Tk=[tk],
                # Td=[td],
                # Tp=[tp]
            )

            if (i+1) % self.nr_data_interval == 0:
                self.save_data(i, 
                               hidden_states=S,
                               u=u,
                               yref=yref,
                               d=d,
                               yout=yout,
                               loss=loss,
                               gradient=gradient.flatten())
                
            if (i+1) % self.nr_interval == 0:
                self.save_checkpoint(i+1)

    def _online_learning_svd(self, nr_iterations: int=100,
                             is_shift_dis: bool=False,
                             is_scratch: bool=False) -> None:
        """
        """
        self.model.NN.eval()
        self.U, S, self.VT = self.kf_initialization(self.model.NN)
        
        if is_scratch is True:
            S = S*0.0

        self.U_asnp = self.U.cpu().numpy()
        self.VT_asnp = self.VT.cpu().numpy()

        dim = len(S)
        if dim < 550:
            self.padding = np.zeros((550-dim, dim))
            self.dir = 'v'
        elif dim >= 550:
            self.padding = None
            self.dir = 'h'

        self.Bd_bar = self.Bd@self.U_asnp

        self.kalman_filter.import_matrix(d=S.view(-1, 1))
        yref_marker, path_marker = self.marker_initialization()

        for i in range(nr_iterations):
            tt = time.time()
            
            if (is_shift_dis is True) and (i > self.nr_shift_dis):
                is_shift_dis = False
                yref_marker = self.shift_distribution()

            self.NN_update(S)

            if i%self.nr_marker_interval == 0:
                self.run_marker_step(yref_marker, path_marker)
            
            yref, _ = self.traj.get_traj()
            t1 = time.time()
            yout, d, u, loss = self._rum_sim(yref)
            tsim = time.time() - t1
            phi = self.extract_output()  # get the output of the last second layer

            t1 = time.time()
            self.kalman_filter.get_A(phi)
            S, tk, td, tp = self.kalman_filter.estimate(yout, self.B@u.reshape(-1, 1))      
            t2 = time.time()

            par_pi_par_omega = self.get_par_pi_par_omega(phi.detach().cpu().numpy())
            L = self.get_L(par_pi_par_omega)
            gradient = L.T@(yout.reshape(-1, 1) - yref[0, 1:551].reshape(-1, 1))
            
            ttotal = time.time() - tt

            self.total_loss += loss
            fcs.print_info(
                Epoch=[str(i+1)+'/'+str(nr_iterations)],
                Loss=[loss],
                AvgLoss=[self.total_loss/(i+1)],
                Ttotal = [ttotal],
                Tsim = [tsim],
                Tk=[tk],
                Td=[td],
                Tp=[tp]
            )

            if (i+1) % self.nr_data_interval == 0:
                self.save_data(i, 
                               hidden_states=S,
                               u=u,
                               yref=yref,
                               d=d,
                               yout=yout,
                               loss=loss,
                               gradient=gradient.flatten())
                
            if (i+1) % self.nr_interval == 0:
                self.save_checkpoint(i+1)

    def NN_update(self, S: torch.Tensor) -> None:
        """Update the last layer of the neural network
        """
        W = self.svd_inference(self.U, S, self.VT)
        vec = W.t().reshape(-1, 1)
        self.assign_last_layer(self.model.NN, vec)


        