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
            self.ENGINE.set_param(self.model_name, key, value)
    
    @staticmethod
    def get_model_path(model_name: str) -> Path:
        """Get the path to the model
        """
        _model_name = model_name + '.slx'
        root = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(root, 'model', _model_name)
        return model_path

    def load_system(self) -> None:
        """Load the model        
        """
        model_path = self.get_model_path()
        self.ENGINE.load_system(model_path)

    def kill_system(self) -> None:
        """Kill the simulation
        """
        self.ENGINE.quit()

    def initialization(self):
        """Initialize the simulation environment
        """
        self.start_engine()
        self.load_system()
        self.set_parameters(self.SIM_PARAMS)
    
    def run_sim(self) -> None:
        """Run the simulation, after specified the inputs
        """
        self.ENGINE.sim(self.model_name, nargout=0)
    
    def get_output(self):
        """Get the output of the simulation
        """
        pass

    def import_input(self, u):
        """Import the input to the system
        """
        pass
