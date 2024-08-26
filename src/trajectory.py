"""Classes for reference trajectories
"""
import numpy as np
import random

random.seed(9527)

import minimum_jerk.minjerk as mj
from mytypes import Array, Array2D

class TRAJ():
    """
    """
    def __init__(self) -> None:
        self.T       = 5.5
        self.dt      = 0.01
        self.range_y = 0.2
        self.range_v = 2.0

    def get_random_value(self, start: float, 
                         end: float, step: float) -> float:
        """Randomly choose one value in [start, end] with interval step
        """
        values = np.arange(start, end, step)
        random_value = random.choice(values)
        return random_value

    def get_t(self) -> Array:
        """Get the array of time points
        """
        return np.array([0.0,  self.get_random_value(1.2, 1.8, self.dt),
                         self.get_random_value(2.9, 3.5, self.dt), 5.0, self.T])
        # return np.array([0.0,  self.get_random_value(1.1, 1.5, self.dt),
        #                  self.get_random_value(2.7, 3.2, self.dt), 5.0, self.T])

    def get_y(self) -> Array:
        """Get the array of the positions
        """
        return np.array([0.0, self.get_random_value(-self.range_y, self.range_y, self.dt),
                         self.get_random_value(-self.range_y, self.range_y, self.dt), 0.0, 0.0])
        # return np.array([0.0, self.get_random_value(-0.4, 0.0, self.dt),
        #                  self.get_random_value(0.0, 0.4, self.dt), 0.0, 0.0])

    def get_v(self) -> Array:
        """Get the array of the velocities
        """
        return np.array([0.0, self.get_random_value(-self.range_v, self.range_v, self.dt),
                         self.get_random_value(-self.range_v, self.range_v, self.dt), 0.0, 0.0])
        # return np.array([0.0, self.get_random_value(-1.5, 1.5, self.dt),
        #                  self.get_random_value(-1.5, 1.5, self.dt), 0.0, 0.0])
    
    def get_a(self) -> Array:
        """Get the array of the accelerations
        """
        return np.zeros(5)
    
    def get_traj(self) -> Array:
        """
        """
        t = self.get_t()
        y = self.get_y()
        v = self.get_v()
        a = self.get_a()
        pp, vv, aa, jj, tt = mj.minimum_jerk_trajectory(y, v, a, t, self.dt)  
        return pp, tt