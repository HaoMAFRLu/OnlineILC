"""Some useful functions
"""
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any, List, Tuple
from tabulate import tabulate

from mytypes import Array, Array2D

def mkdir(path: Path) -> None:
    """Check if the folder exists and create it if it does not exist.
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def _set_axes_radius_2d(ax, origin, radius) -> None:
    x, y = origin
    ax.set_xlim([x - radius, x + radius])
    ax.set_ylim([y - radius, y + radius])
    
def set_axes_equal_2d(ax: Axes) -> None:
    """Set equal x, y axes
    """
    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius_2d(ax, origin, radius)

def set_axes_format(ax: Axes, x_label: str, y_label: str) -> None:
    """Format the axes
    """
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    # ax.legend(loc='upper left')
    ax.grid()

def preprocess_kwargs(**kwargs):
    """Project the key
    """
    replacement_rules = {
        "__slash__": "/",
        "__percent__": "%"
    }

    processed_kwargs = {}
    key_map = {}
    for key, value in kwargs.items():
        new_key = key
        
        for old, new in replacement_rules.items():
            new_key = new_key.replace(old, new)

        processed_kwargs[key] = value
        key_map[key] = new_key
    
    return processed_kwargs, key_map

def print_info(**kwargs):
    """Print information on the screen
    """
    processed_kwargs, key_map = preprocess_kwargs(**kwargs)
    columns = [key_map[key] for key in processed_kwargs.keys()]
    data = list(zip(*processed_kwargs.values()))
    table = tabulate(data, headers=columns, tablefmt="grid")
    print(table)