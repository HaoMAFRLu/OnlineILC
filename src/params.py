"""Define the simulation parameters for different environments
"""
from dataclasses import dataclass, fields, asdict
from pathlib import Path
import utils as fcs
import os
import json

@dataclass
class SIM_PARAMS:
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
    data_format: str
    is_normalization: bool
    is_centerization: bool
    input_scale: float
    output_scale: float
    k: float
    batch_size: int
    input_name: str
    output_name: str
    channel: int
    height: int
    width: int

# @dataclass
# class ONLINE_DATA_PARAMS:
#     """The hyperparameters for generating online data processor
#     """
#     data_format: str
#     is_normalization: bool
#     is_centerization: bool
#     input_scale: float
#     output_scale: float
#     input_name: str
#     output_name: str
#     channel: int
#     height: int
#     width: int

@dataclass
class NN_PARAMS:
    """The hyperparameters for neural networks
    """
    data_format: str
    is_initialization: bool
    loss_function: str
    lambda_regression: float
    learning_rate: float
    weight_decay: float
    channel: int
    height: int
    width: int
    filter_size: int
    output_dim: int

@dataclass
class VISUAL_PARAMS:
    """The parameters for visualization
    """
    is_save: bool
    paths: list
    checkpoint: str
    data: str

@dataclass
class KF_PARAMS:
    """The hyperparameters for the kalman filter
    """
    sigma_w: float
    sigma_ini: float
    sigma_y: float
    sigma_d: float
    decay_R: float
    dim: int

dataclass_map = {
    "SIM_PARAMS":          SIM_PARAMS,
    "OFFLINE_DATA_PARAMS": OFFLINE_DATA_PARAMS,
    "NN_PARAMS":           NN_PARAMS,
    "VISUAL_PARAMS":       VISUAL_PARAMS,
    "KF_PARAMS":           KF_PARAMS
}

class PARAMS_GENERATOR():
    """Generator parameters according to the config file

    parameters:
    -----------
    PATH_CONFIG: path to the config
    """
    def __init__(self, PATH_CONFIG: Path=None) -> None:
        self.root = fcs.get_parent_path()
        self.initialization()
        self.get_config(PATH_CONFIG)
    
    def get_dataclass_instance(self, config_key, CONFIG):
        if config_key in dataclass_map:
            dataclass_type = dataclass_map[config_key]
            # Filter out extra keys that are not part of the dataclass
            dataclass_fields = {field.name for field in fields(dataclass_type)}
            filtered_data = {k: v for k, v in CONFIG.items() if k in dataclass_fields}
            return asdict(dataclass_type(**filtered_data))
        else:
            raise ValueError(f"Unknown config key: {config_key}")

    def initialization(self):
        """Initialize the parameters
        """
        self.PARAMS = {}

    def get_config(self, PATH_CONFIG: Path=None) -> None:
        """Get the config file
        """
        if PATH_CONFIG is None:
            PATH_CONFIG = os.path.join(self.root, 'config.json')
        
        with open(PATH_CONFIG, 'rb') as file:
            self.CONFIG = json.load(file)
        
    def get_params(self, PARAMS_LIST: list) -> None:
        """Generate the dataclass for the given names
        """
        if isinstance(PARAMS_LIST, str):
            self.PARAMS[PARAMS_LIST] = self.get_dataclass_instance(PARAMS_LIST, self.CONFIG)
        elif isinstance(PARAMS_LIST, list):
            for key in PARAMS_LIST:
                self.PARAMS[key] = self.get_dataclass_instance(key, self.CONFIG)
