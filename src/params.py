"""Define the simulation parameters for different environments
"""
from dataclasses import dataclass

@dataclass
class Beam_SIM_PARAMS:
    """The dataclass for the beam simulation
    """
    StopTime: str
    StartTime: str
    AbsTol: str
    Solver: str
    SimulationMode: str

@dataclass
class OFFLINE_DATA_PARAMS:
    """The hyperparameters for genearating 
    offline training data
    """
    mode: str
    k: float
    batch_size: int
    input_name: str
    output_name: str
    channel: int
    height: int
    width: int

@dataclass
class NN_PARAMS:
    """The hyperparameters for neural networks
    """
    is_initialization: bool
    loss_function: str
    learning_rate: float
    weight_decay: float
    channel: int
    height: int
    width: int
    filter_size: int
    output_dim: int

