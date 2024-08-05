"""Classes for the kalman filter
"""
import numpy as np
import numba as nb
import time
import torch

from mytypes import Array, Array2D

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
            self.I = np.eye(self.dim)
            self.Z = np.zeros((550-self.dim, self.dim))
        
        self.R_ = np.eye(550) * self.sigma_y
        self.Q = self.I * self.sigma_d
        self.P = self.I * self.sigma_ini

    def import_d(self, d: Array2D) -> None:
        """Import the initial value of the disturbance
        
        parameters:
        -----------
        d: the given disturbance, column wise
        """
        self.d = d.copy()

    def add_one(self, phi: Array) -> Array:
        """Add element one at the end
        """
        return np.hstack([phi.flatten(), 1])

    @staticmethod
    def get_v(VT: Array2D, phi: Array) -> Array:
        """Return v
        """
        return VT@phi.reshape(-1, 1)
        
    def get_A(self, phi: Array) -> None:
        """Get the dynamic matrix A
        
        parameters:
        -----------
        phi: the output of the last second layer
        """
        phi_bar = self.add_one(phi)
        if self.mode is None:            
            self.A = np.kron(phi_bar, self.Bd)
        elif self.mode == 'svd':
            v = self.get_v(self.VT, phi_bar)
            self.A = self.Bd_bar@np.vstack((np.diag(v.flatten()), self.Z))   
    
    def get_Bd_bar(self, Bd: Array2D, U: Array2D) -> None:
        """Return Bd_bar
        """
        self.Bd_bar = Bd@U
    
    @staticmethod
    # @nb.jit(nopython=True)
    def get_difference(yout, B, u):
        return yout-B@u

    @staticmethod
    # @nb.jit(nopython=True)
    def prediction(A, d):
        return A@d
    
    @staticmethod
    # @nb.jit(nopython=True)
    def get_P_prediction(P, Q):
        return P+Q
    
    @staticmethod
    # @nb.jit(nopython=False)
    def get_K(P, A, R):
        return (P@A.T)@np.linalg.inv(A@P@A.T + R)
    
    @staticmethod
    # @nb.jit(nopython=True)
    def update_d(d, K, z, A):
        return d + K@(z - A@d)
    
    # @staticmethod
    # @nb.jit(nopython=True)
    # def update_P(I, K, A, P, R):
    #     KA = K@A
    #     I_KA = I - KA
    #     KRT = K@R@K.T
    #     return I_KA@P@I_KA.T + KRT

    # @nb.jit(nopython=True)
    # def update_P(I: np.ndarray, K: np.ndarray,
    #              A: np.ndarray, P: np.ndarray, 
    #              R: np.ndarray):
    #     KA = np.dot(K, A)
    #     I_KA = I - KA
    #     KRT = np.dot(np.dot(K, R), K.T)
    #     result = np.dot(np.dot(I_KA, P), I_KA.T) + KRT
    #     return result
    
    # def update_P(self, I, K, A, P, R):
    #     I = torch.tensor(I).to(self.device)
    #     K = torch.tensor(K).to(self.device)
    #     A = torch.tensor(A).to(self.device)
    #     P = torch.tensor(P).to(self.device)
    #     R = torch.tensor(R).to(self.device)
    #     KA = torch.matmul(K, A)
    #     I_minus_KA = I - KA
    #     KRT = torch.matmul(torch.matmul(K, R), K.t())
    #     result = torch.matmul(torch.matmul(I_minus_KA, P), I_minus_KA.t()) + KRT
    #     return result.to('cpu')

    # @nb.jit(nopython=True)
    def get_R(self):
        return (self.A@self.A.T)*self.sigma_w + self.R_

    def estimate(self, yout: Array, u: Array) -> Array:
        """Estimate the states
        
        parameters:
        -----------
        c: the output of the second last layer
        Bd: the disturbance dynamics
        yout: output of the underlying system
        u: input of the system
        """
        # @nb.jit(nopython=True)
        # def update_P(I: np.ndarray, K: np.ndarray,
        #              A: np.ndarray, P: np.ndarray, 
        #              R: np.ndarray):
        #     KA = np.dot(K, A)
        #     I_KA = I - KA
        #     KRT = np.dot(np.dot(K, R), K.T)
        #     result = np.dot(np.dot(I_KA, P), I_KA.T) + KRT
        #     return result

        # @nb.jit(nopython=True)
        def dot_product(A, B):
            return np.dot(A, B)

        # @nb.jit(nopython=True)
        def update_P(I, K, A, P, R):
            KA = dot_product(K, A)
            I_KA = I - KA
            KRT = dot_product(dot_product(K, R), K.T)
            result = dot_product(dot_product(I_KA, P), I_KA.T) + KRT
            return result

        R = self.get_R()
        z = self.get_difference(yout, self.B, u)
        # d_pred = self.prediction(self.I, self.d)
        d_pred = self.d.copy()
        
        P_pred = self.get_P_prediction(self.P, self.Q)

        t = time.time()
        K = self.get_K(P_pred, self.A, R)
        t_k = time.time() - t
        
        t = time.time()
        self.d = self.update_d(d_pred, K, z, self.A)
        t_d = time.time() - t

        t = time.time()
        self.P = update_P(self.I, K, self.A, P_pred, R)
        t_p = time.time() - t

        return self.d, t_k, t_d, t_p