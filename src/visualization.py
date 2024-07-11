"""Classes used for showing the training results
"""

import numpy as np
from pathlib import Path
import torch
import torch.nn
from typing import Any, List, Tuple
import os

class Visual():
    """
    """
    def __init__(self, PARAMS: dict) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.abspath(os.path.join(current_dir, os.pardir))
        self.path_params = self.get_path_params(PARAMS['paths'], PARAMS['checkpoint'])

    def get_path_params(paths: List[str], file: str):
        pass

    def load_model(self, params_path: Path=None,
                   **kwargs: Any) -> None:
        """Specify the model structure and 
        load the pre-trained model parameters
        
        parameters:
        -----------
        params_path: path to the pre-trained parameters
        model: the model structure
        optimizer: the optimizer
        """
        if params_path is None:
            params_path = self.params_path

        checkpoint = torch.load(params_path)
        if 'model' in kwargs:
            kwargs['model'].load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer' in kwargs:
            kwargs['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])








