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
class PRETRAIN_PARAMS:
    """The hyperparameters for pretraining
    """
    mode: str
    k: float
    batch_size: int
    input_name: str
    output_name: str
    channel: int
    height: int
    width: int