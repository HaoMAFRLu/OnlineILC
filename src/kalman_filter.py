"""Classes for the kalman filter
"""
import numpy as np
import numba as nb

from mytypes import Array, Array2D

class KalmanFilter():
    """
    """
    def __init__(self, dim: int, PARAMS: dict) -> None:
        self.sigma_w = PARAMS["sigma_w"]
        self.sigma_y = PARAMS["sigma_y"]

    @staticmethod
    @nb.jit(nopython=True)
    def get_difference(yout, B, u):
        return yout-B@u

    @staticmethod
    @nb.jit(nopython=True)
    def prediction(A, d):
        return A@d
    
    @staticmethod
    @nb.jit(nopython=True)
    def get_P_prediction(P, Q):
        return P+Q
    
    @staticmethod
    @nb.jit(nopython=False)
    def get_K(P, A, R):
        return (P@A.T)@np.linalg.inv(A@P@A.T + R)
    
    @staticmethod
    @nb.jit(nopython=True)
    def update_d(d, K, z, A):
        return d + K@(z - A@d)
    
    @staticmethod
    @nb.jit(nopython=True)
    def update_P(I, K, A, P, R):
        H = I - K@A
        return H@P@H.T + K@R@K.T
    
    def estimate(self, yout: Array, u: Array) -> Array:
        """Estimate the states
        
        parameters:
        -----------
        yout: output of the underlying system
        u: input of the system
        """
        A = self.get_A()
        R = self.get_R()
        z = self.get_difference(yout, self.B, u)
        d_pred = self.prediction(self.I, self.d)
        P_pred = self.get_P_prediction(self.P, self.Q)
        K = self.get_K(P_pred, A, R)
        self.d = self.update_d(d_pred, K, z, A)
        self.P = self.update_P(self.I, K, A, P_pred, R)
        return self.d