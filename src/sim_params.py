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
    