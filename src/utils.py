"""Some useful functions
"""
import numpy as np
import os
from pathlib import Path
from matplotlib.axes import Axes
from typing import Any, List, Tuple
from tabulate import tabulate
import shutil
import torch

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

def get_parent_path(lvl: int=0):
    """Get the lvl-th parent path as root path.
    Return current file path when lvl is zero.
    Must be called under the same folder.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    if lvl > 0:
        for _ in range(lvl):
            path = os.path.abspath(os.path.join(path, os.pardir))
    return path

def copy_folder(src, dst):
    try:
        if os.path.isdir(src):
            folder_name = os.path.basename(os.path.normpath(src))
            dst_folder = os.path.join(dst, folder_name)
            shutil.copytree(src, dst_folder)
            print(f"Folder '{src}' successfully copied to '{dst_folder}'")
        elif os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"File '{src}' successfully copied to '{dst}'")
        else:
            print(f"Source '{src}' is neither a file nor a directory.")
    except FileExistsError:
        print(f"Error: Destination '{dst}' already exists.")
    except FileNotFoundError:
        print(f"Error: Source '{src}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
def load_model(path: Path) -> None:
    """Load the model parameters
    
    parameters:
    -----------
    params_path: path to the pre-trained parameters
    """
    return torch.load(path)