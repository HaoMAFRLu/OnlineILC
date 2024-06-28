"""Some useful functions
"""
import numpy as np
import os
from pathlib import Path

def mkdir(path: Path) -> None:
    """Check if the folder exists and create it if it does not exist.
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)