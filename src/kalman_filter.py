"""Classes for the kalman filter
"""
import numpy as np
import numba as nb
import time
import torch

from mytypes import Array, Array2D
import utils as fcs
class KalmanFilter():
    """
    """
    def __init__(self, mode: str,
                 B: Array2D, Bd: Array2D, 
                 PARAMS: dict) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mode = mode

        self.sigma_w = PARAMS["sigma_w"]
        self.sigma_d = PARAMS["sigma_d"]
        self.sigma_y = PARAMS["sigma_y"]
        self.sigma_ini = PARAMS["sigma_ini"]
        self.dim = PARAMS["dim"]
        self.q = self.dim * 550

        self.d = None
        self.B = B
        self.Bd = Bd

        self.initialization()
    
    def initialization(self) -> None:
        """Initialize the kalman filter
        """
        if self.mode is None:
            self.I = np.eye(self.q)
        elif self.mode == 'svd':
            self.I = np.eye(min(550, self.dim))
            if self.dim < 550:
                self.Z = np.zeros((550-self.dim, self.dim))
                self.dir = 'v'
            elif self.dim > 550:
                self.dir = 'h'
        elif self.mode == 'ada-svd':
            self.A = None
            self.I = np.eye(min(550, self.dim))
            if self.dim < 550:
                self.Z = np.zeros((550-self.dim, self.dim))
                self.dir = 'v'
            elif self.dim > 550:
                self.dir = 'h'

        self.R_ = np.eye(550) * self.sigma_y
        self.Q = self.I * self.sigma_d
        self.P = self.I * self.sigma_ini

        if self.mode is None:
            self.tensor_intialization()
    
    def tensor_intialization(self) -> None:
        """Initialize all the matrices as tensors
        """
        self.R_ = torch.from_numpy(self.R_).to(self.device).float()
        self.I = torch.from_numpy(self.I).to(self.device).float()
        self.Q = torch.from_numpy(self.Q).to(self.device).float()
        self.P = torch.from_numpy(self.P).to(self.device).float()
        self.B = torch.from_numpy(self.B).to(self.device).float()
        self.Bd = torch.from_numpy(self.Bd).to(self.device).float()

    def update_covariance(self, dim: int, **kwargs) -> None:
        """Update the covariance matrices
        """
        if dim > 550:
            self.R_ = fcs.diagonal_concatenate(self.R_*1.7, 
                                               np.eye(550)*self.sigma_y, 
                                               kwargs["max_rows"])

    def import_d(self, d: Array2D) -> None:
        """Import the initial value of the disturbance
        
        parameters:
        -----------
        d: the given disturbance, column wise
        """
        if isinstance(d, np.ndarray):
            self.d = d.copy()
        elif isinstance(d, torch.Tensor):
            self.d = d.clone()

    def add_one(self, phi: Array) -> Array:
        """Add element one at the end
        """
        return np.hstack([phi.flatten(), 1])

    @staticmethod
    def get_v(VT: Array2D, phi: Array) -> Array:
        """Return v
        """
        return VT@phi.reshape(-1, 1)
        
    def get_A(self, phi, **kwargs) -> None:
        """Get the dynamic matrix A
        
        parameters:
        -----------
        phi: the output of the last second layer
        """
        if isinstance(phi, np.ndarray):
            phi_bar = self.add_one(phi)
        elif isinstance(phi, torch.Tensor):
            new_element = torch.tensor([1]).to(self.device)
            phi_bar = torch.cat((phi.flatten(), new_element))
        
        if self.mode is None:            
            # self.A = np.kron(phi_bar, self.Bd)
            self.A = torch.kron(phi_bar.view(1, -1), self.Bd.contiguous())/1000.0

        elif self.mode == 'svd':
            v = self.get_v(self.VT, phi_bar)
            if self.dir == 'v':
                self.A = self.Bd_bar@np.vstack((np.diag(v.flatten()), self.Z))/1000.0
            elif self.dir == 'h':
                self.A = self.Bd_bar@np.diag(v.flatten()[:550])/1000.0
        elif self.mode == 'ada-svd':
            v = self.get_v(self.VT, phi_bar)
            if self.dir == 'v':
                cur_A = self.Bd_bar@np.vstack((np.diag(v.flatten()), self.Z))/1000.0
            elif self.dir == 'h':
                cur_A = self.Bd_bar@np.diag(v.flatten()[:550])/1000.0
            self.A = fcs.adjust_matrix(self.A, cur_A, kwargs["max_rows"])
            self.update_covariance(self.A.shape[0], max_rows=kwargs["max_rows"])

    def get_Bd_bar(self, Bd: Array2D, U: Array2D) -> None:
        """Return Bd_bar
        """
        self.Bd_bar = Bd@U
    
    def _estimate_numpy(self, yout: Array2D, Bu: Array2D) -> Array2D:
        """Numpy version
        """
        def get_difference(yout, Bu):
            return yout-Bu
        
        def get_P_prediction(P, Q):
            return P+Q
        
        def get_K(P, A, R):
            return (P@A.T)@np.linalg.inv(A@P@A.T + R)
        
        def update_d(d, K, z, A):
            return d + K@(z - A@d)
    
        def get_R(A, R_, sigma_w):
            return (A@A.T)*sigma_w + R_
    
        def dot_product(A, B):
            return np.dot(A, B)

        def update_P(I, K, A, P, R):
            KA = dot_product(K, A)
            I_KA = I - KA
            KRT = dot_product(dot_product(K, R), K.T)
            result = dot_product(dot_product(I_KA, P), I_KA.T) + KRT
            return result

        R = get_R(self.A, self.R_, self.sigma_w)
        z = get_difference(yout, Bu)
        d_pred = self.d.copy()
        
        P_pred = get_P_prediction(self.P, self.Q)

        t = time.time()
        K = get_K(P_pred, self.A, R)
        t_k = time.time() - t
        
        t = time.time()
        self.d = update_d(d_pred, K, z, self.A)
        t_d = time.time() - t

        t = time.time()
        self.P = update_P(self.I, K, self.A, P_pred, R)
        t_p = time.time() - t

        return self.d, t_k, t_d, t_p

    def _estimate_tensor(self, yout: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
        """Torch version
        """
        def get_difference(yout, Bu):
            return yout-Bu
        
        def get_P_prediction(P, Q):
            return P+Q
        
        def get_K(P, A, R):
            PAT = torch.matmul(P, A.t())
            APAT = torch.matmul(A, PAT)
            inv_APATR = torch.inverse(APAT+R)
            return torch.matmul(PAT, inv_APATR)
        
        def update_d(d, K, z, A):
            Ad = torch.matmul(A, d)
            zAd = z - Ad
            return d + torch.matmul(K, zAd)
    
        def get_R(A, R_, sigma_w):
            AAT = torch.matmul(A, A.t())*sigma_w
            return AAT + R_

        def update_P(I, K, A, P, R):
            KA = torch.matmul(K, A)
            I_KA = I - KA
            KR = torch.matmul(K, R)
            KRT = torch.matmul(KR, K.t())
            I_KAP = torch.matmul(I_KA, P)
            return torch.matmul(I_KAP, I_KA.t()) + KRT

        R = get_R(self.A, self.R_, self.sigma_w)
        z = get_difference(yout, Bu)
        d_pred = self.d.clone()
        
        P_pred = get_P_prediction(self.P, self.Q)

        t = time.time()
        K = get_K(P_pred, self.A, R)
        t_k = time.time() - t
        
        t = time.time()
        self.d = update_d(d_pred, K, z, self.A)
        t_d = time.time() - t

        t = time.time()
        self.P = update_P(self.I, K, self.A, P_pred, R)
        t_p = time.time() - t

        return self.d, t_k, t_d, t_p

    def estimate(self, yout: Array, Bu: Array) -> Array:
        """Estimate the states
        
        parameters:
        -----------
        yout: output of the underlying system
        Bu: B*u, u is the input of the system
        """
        if self.mode is None:
            yout_tensor = torch.from_numpy(yout).to(self.device).float()
            Bu_tensor = torch.from_numpy(Bu).to(self.device).float()
            return self._estimate_tensor(yout_tensor, Bu_tensor)
        else:
            return self._estimate_numpy(yout, Bu)


        