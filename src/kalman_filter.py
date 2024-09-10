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
                 rolling:int, 
                 PARAMS: dict,
                 sigma_w: float=None,
                 sigma_y: float=None,
                 sigma_d: float=None,
                 sigma_ini: float=None,
                 location: str='local') -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.location = location
        self.rolling =rolling
        self.max_rows = self.rolling*550


        if sigma_y is None:
            self.sigma_w = PARAMS["sigma_w"]
            self.sigma_d = PARAMS["sigma_d"]
            self.sigma_y = PARAMS["sigma_y"]
            self.sigma_ini = PARAMS["sigma_ini"]
        else:
            self.sigma_w = sigma_w
            self.sigma_d = sigma_d
            self.sigma_y = sigma_y
            self.sigma_ini = sigma_ini
        
        self.decay_R = PARAMS["decay_R"]
        self.dim = PARAMS["dim"]
        self.q = self.dim * 550
        self.h = min(550, self.dim)

        self.d = None
        self.B = B
        self.Bd = Bd

        self.initialization()
    
    def initialization(self) -> None:
        """Initialize the kalman filter
        """
        if self.mode == 'full_states':
            self._initialization(self.q)
        elif self.mode == 'svd':
            self._initialization(self.h)
        elif self.mode == 'representation':
            self._initialization(self.h)

        # if self.mode == 'full_states':
        #     self.I = np.eye(self.q)
        # elif self.mode == 'svd':
        #     self.A = None
        #     self.I = np.eye(min(550, self.dim))
        #     if self.dim < 550:
        #         self.padding = np.zeros((550-self.dim, self.dim))
        #         self.dir = 'v'
        #     elif self.dim > 550:
        #         self.dir = 'h'

        # self.R_ = np.eye(550) * self.sigma_y
        # self.Q = self.I * self.sigma_d
        # self.P = self.I * self.sigma_ini

        # self.tensor_intialization()
    
    def move_to_device(self, data: Array) -> torch.Tensor:
        """Move data to the device
        """ 
        return torch.from_numpy(data).to(self.device).float()

    def _initialization(self, l):
        """initialization
        """
        # the matrices A and R_ are initialized during training
        self.A = None
        self.R_ = None
        self.yout_tensor = None
        self.Bu_tensor = None

        self.I = np.eye(l)
        self.Iq = np.eye(550)
        self.P = np.eye(l)*self.sigma_ini
        self.P_pred = np.zeros((l, l))
        self.Q = np.eye(l)*self.sigma_d

        if self.dim < 550:
            self.padding = np.zeros((550-self.dim, self.dim))
            self.dir = 'v'
        elif self.dim >= 550:
            self.padding = None
            self.dir = 'h'

        self.A_tmp = np.zeros((550, l))
        self.d_pred = np.zeros((l, 1))
        self.ini_tensor()
    
    def ini_tensor(self):
        """
        """
        if self.padding is not None:
            self.padding = self.move_to_device(self.padding)
        self.A_tmp = self.move_to_device(self.A_tmp)
        self.Iq = self.move_to_device(self.Iq)
        self.I = self.move_to_device(self.I)
        self.d_pred = self.move_to_device(self.d_pred)
        self.P = self.move_to_device(self.P)
        self.P_pred = self.move_to_device(self.P_pred)
        self.Q = self.move_to_device(self.Q)
        self.B = self.move_to_device(self.B)
        self.Bd = self.move_to_device(self.Bd)

    def import_d(self, d: Array2D) -> None:
        """Import the initial value of the disturbance
        
        parameters:
        -----------
        d: the given disturbance, column wise
        """
        if isinstance(d, np.ndarray):
            self.d = d.copy()
        elif isinstance(d, torch.Tensor):
            self.d.copy_(d.view(-1, 1))

    def add_one(self, phi: Array) -> Array:
        """Add element one at the end
        """
        return np.hstack([phi.flatten(), 1])

    # @staticmethod
    # def get_v(VT: Array2D, phi: Array) -> Array:
    #     """Return v
    #     """
    #     return VT@phi.reshape(-1, 1)

    @staticmethod
    def get_v(VT: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Return v
        """
        return torch.matmul(VT, phi.view(-1, 1))
    
    def _update_A(self):
        """Update the matrix A
        """
        if self.A is None:
            self.A = self.A_tmp.clone()
        else:
            original_rows = self.A.shape[0]
            new_rows = self.A_tmp.shape[0]
            total_rows = original_rows + new_rows
            if total_rows > self.rolling*new_rows:
                self.A.copy_(torch.vstack((self.A[new_rows:, :], self.A_tmp)))
            else:
                self.A = torch.vstack((self.A, self.A_tmp))
    
    def _update_R_(self):
        """Update the matrix R_
        """
        if self.R_ is None:
            self.R_ = self.Iq*self.sigma_y
        else:
            original_rows = self.R_.shape[0]
            new_rows = 550
            total_rows = original_rows + new_rows
            if total_rows > self.rolling*new_rows:
                self.R_.copy_(torch.block_diag(self.R_[new_rows:, new_rows:]*self.decay_R, self.Iq*self.sigma_y))
            else:
                self.R_= torch.block_diag(self.R_*self.decay_R, self.Iq*self.sigma_y)

    # def _update_P(self):
    #     """Update the matrix P and Q
    #     """
    #     if self.P is None:
    #         self.P = self.I*self.sigma_ini
    #         self.Q = self.I*self.sigma_d
    #     else:
    #         original_rows = self.P.shape[0]
    #         new_rows = min(self.dim, 550)
    #         total_rows = original_rows + new_rows
    #         if total_rows > self.rolling*new_rows:
    #             self.P.copy_(torch.block_diag(self.P[new_rows:, new_rows:], self.P[-new_rows:, -new_rows:]))
    #         else:
    #             self.P = torch.block_diag(self.P, self.P[-new_rows:, -new_rows:])
    #             self.Q = torch.block_diag(self.Q, self.I*self.sigma_d)

    def update_matrix(self):
        """Update A, R_ and P in the svd mode
        """
        self._update_A()
        self._update_R_()
        # self._update_P()

    def get_A(self, phi: torch.Tensor, **kwargs) -> None:
        """Get the dynamic matrix A
        
        parameters:
        -----------
        phi: the output of the last second layer
        """
        new_element = torch.tensor([1]).to(self.device).float()
        phi_bar = torch.cat((phi.flatten(), new_element))
        if 'dphi' in kwargs:
            phi_bar = phi_bar + kwargs['dphi'].flatten()

        if self.mode == 'full_states':            
            self.A_tmp.copy_(torch.kron(phi_bar.view(1, -1), self.Bd.contiguous())/1000.0)

        elif self.mode == 'svd':
            v = self.get_v(self.VT, phi_bar)
            if self.dir == 'v':
                self.A_tmp.copy_(torch.matmul(self.Bd_bar,torch.vstack((torch.diag(v.flatten()), self.padding)))/1000.0)
            elif self.dir == 'h':
                self.A_tmp.copy_(torch.matmul(self.Bd_bar, torch.diag(v.flatten()[:550]))/1000.0)
        
        elif self.mode == 'representation':
            self.A_tmp.copy_(torch.matmul(self.Bd, phi)/1000.0)

        self.update_matrix()

    

    def import_matrix(self, **kwargs):
        """Import the matrix ()
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    # def get_Bd_bar(self) -> None:
    #     """Return Bd_bar
    #     """
    #     
    #     self.Bd_bar = self.Bd@self.U
    
    def get_Bd_bar(self) -> None:
        """Return Bd_bar
        """
        self.Bd_bar = torch.matmul(self.Bd, self.U)

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
            with torch.no_grad():
                return torch.matmul(torch.matmul(P, A.t()), torch.inverse(torch.matmul(A, torch.matmul(P, A.t())) + R))
        
        def update_d(d, K, z, A):
            with torch.no_grad():
                return d + torch.matmul(K, z - torch.matmul(A, d))
    
        def get_R(A, R_, sigma_w):
            with torch.no_grad():
                return torch.matmul(A, A.t()) * sigma_w + R_

        def update_P(I, K, A, P, R):
            with torch.no_grad():
                return torch.matmul(torch.matmul(I - torch.matmul(K, A), P), (I - torch.matmul(K, A)).t()) + torch.matmul(torch.matmul(K, R), K.t())
        
        # two versions
        t = time.time()
        if self.is_rolling is True:
            self.R.copy_(get_R(self.A, self.R_, self.sigma_w))
            self.z.copy_(get_difference(yout, Bu))
            self.P_pred.copy_(get_P_prediction(self.P, self.Q))
            self.K.copy_(get_K(self.P_pred, self.A, self.R))
        else:
            self.R = get_R(self.A, self.R_, self.sigma_w)
            self.z = get_difference(yout, Bu)
            self.P_pred.copy_(get_P_prediction(self.P, self.Q))
            self.K = get_K(self.P_pred, self.A, self.R)
        tk = time.time() - t

        self.d_pred.copy_(self.d)
        t = time.time()
        self.d.copy_(update_d(self.d_pred, self.K, self.z, self.A))
        td = time.time() - t

        t = time.time()
        self.P.copy_(update_P(self.I, self.K, self.A, self.P_pred, self.R))
        tp = time.time() - t

        return self.d, tk, td, tp

    def update_inputs(self, yout: Array2D, Bu: Array2D) -> None:
        """
        """
        if self.yout_tensor is None:
            self.is_rolling = False
            self.yout_tensor = self.move_to_device(yout)
            self.Bu_tensor = self.move_to_device(Bu)
        else: 
            original_rows = self.yout_tensor.shape[0]
            new_rows = yout.shape[0]
            total_rows = original_rows + new_rows
        
            if total_rows > self.rolling*new_rows:
                self.is_rolling = True
                self.yout_tensor.copy_ (torch.vstack((self.yout_tensor[new_rows:, :], self.move_to_device(yout))))           
                self.Bu_tensor.copy_(torch.vstack((self.Bu_tensor[new_rows:, :], self.move_to_device(Bu))))           
            else:
                self.is_rolling = False
                self.yout_tensor = torch.vstack((self.yout_tensor, self.move_to_device(yout)))          
                self.Bu_tensor = torch.vstack((self.Bu_tensor, self.move_to_device(Bu)))

    def estimate(self, yout: Array, Bu: Array) -> Array:
        """Estimate the states
        
        parameters:
        -----------
        yout: output of the underlying system
        Bu: B*u, u is the input of the system
        """
        self.update_inputs(yout, Bu)
        return self._estimate_tensor(self.yout_tensor, self.Bu_tensor)
        

        