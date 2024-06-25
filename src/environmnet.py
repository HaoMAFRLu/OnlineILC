"""Classes for simulation
"""
import numpy as np
import os
import matlab.engine
from dataclasses import dataclass
from pathlib import Path

class Beam:
    """The beam simulation, implemented in simulink
    
    parameters:
    -----------
    model_name: the name of the model
    SIM_PARAMS: simulation paramteres
        |--StopTime: the simulation time, in second
        |--ModelOutput: the output variable name
    """
    def __init__(self, model_name: str, SIM_PARAMS: dataclass) -> None:
        self.model_name = model_name
        self.SIM_PARAMS = SIM_PARAMS
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(self.root, 'model')
        self.model_path = self.get_model_path(self.model_name)

    def start_engine(self) -> None:
        """Start the simulink engine
        """
        self.ENGINE = matlab.engine.start_matlab()
    
    def set_parameters(self, SIM_PARAMS: dataclass) -> None:
        """Set the parameters of the simulation

        parameters:
        -----------
        SIM_PARAMS: the simulation parameters
        """
        for key, value in SIM_PARAMS.__dict__.items():
            self.ENGINE.set_param(self.model_name, key, value, nargout=0)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the path to the model
        """
        _model_name = model_name + '.slx'
        model_path = os.path.join(self.root, 'model', _model_name)
        return model_path

    def add_path(self, path: Path) -> None:
        """Add path to matlab
        *This is an important step, otherwise python will
        only try to search for model components in the Matlab 
        root directory.
        """
        self.ENGINE.addpath(path, nargout=0)

    def load_system(self, model_path: Path) -> None:
        """Load the model               
        """
        self.ENGINE.load_system(model_path)

    def kill_system(self) -> None:
        """Kill the simulation
        """
        self.ENGINE.quit()

    def initialization(self):
        """Initialize the simulation environment
        """
        self.start_engine()
        self.add_path(self.path)
        self.load_system(self.model_path)
        self.set_parameters(self.SIM_PARAMS)
    
    def run_sim(self) -> None:
        """Run the simulation, after specified the inputs
        """
        self.simout = self.ENGINE.sim(self.model_path)
        # try:
        #     self.ENGINE.set_param(self.model_name, 'SimulationCommand', 'start', nargout=0)
        #     # self.ENGINE.pause(nargout=0)  # Pause for simulation to run
        #     self.ENGINE.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        # except matlab.engine.MatlabExecutionError as e:
        #     print("Error during simulation:", e)
        #     raise

    def get_output(self):
        """Get the output of the simulation
        """
        y = self.ENGINE.get(self.simout, 'y')
        data = self.matlab_2_nparray(self.ENGINE.get(y, 'Data'))

        theta = self.ENGINE.get(self.simout, 'theta')
        data = self.matlab_2_nparray(self.ENGINE.get(theta['signal1'], 'Data')
                                     )
        print('here')
        return data
    
    @staticmethod
    def nparray_2_matlab(value: np.ndarray) -> matlab.double:
        """Convert data in np.ndarray to matlab.double
        """
        return matlab.double(value.tolist())

    @staticmethod
    def matlab_2_nparray(value: matlab.double) -> np.ndarray:
        """Convert data in matlab.double to np.ndarray
        """
        return np.array(value)

    def set_input(self, name: str, value: matlab.double) -> None:
        """Import the input to the system
        """
        if isinstance(value, np.ndarray):
            value = self.nparray_2_matlab(value)
        self.ENGINE.workspace[name] = value
