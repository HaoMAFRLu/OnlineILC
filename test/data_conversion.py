"""Convert matlab data to python
"""

import scipy.io
import os
import pickle
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

def data_conversion():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_dir = os.path.join(parent_dir, 'data', 'ilc')
    file_names = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    num_files = len(file_names) - 1
    yref_list = [None] * num_files
    yout_list = [None] * num_files
    u_list = [None] * num_files
    d_list = [None] * num_files

    for i in range(num_files):
        name = str(i+1) + '.mat'
        path_data = os.path.join(data_dir, name)
        mat_data = scipy.io.loadmat(path_data)
        sim_result = mat_data["sim_result"]
        yref = sim_result['yref'][0, 0].flatten().astype(np.float32)
        yout = sim_result['yout'][0, 0].flatten().astype(np.float32)
        d = sim_result['d'][0, 0].flatten().astype(np.float32)
        u = sim_result['u'][0, 0].flatten().astype(np.float32)
        yref_list[i] = yref
        yout_list[i] = yout
        u_list[i] = u
        d_list[i] = d

    data = (yref_list, yout_list, u_list, d_list)
    keys = ['yref', 'yout', 'u', 'd']

    path_pretraining = os.path.join(parent_dir, 'data', 'pretraining')
    fcs.mkdir(path_pretraining)

    path_file = os.path.join(path_pretraining, 'ilc')
    with open(path_file, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    path_file = os.path.join(path_pretraining, 'keys')
    with open(path_file, 'wb') as file:
        pickle.dump(keys, file, protocol=pickle.HIGHEST_PROTOCOL)

    B = sim_result['B'][0, 0].astype(np.float32)
    Bd = sim_result['Bd'][0, 0].astype(np.float32)
    linear_model = (B, Bd)
    path_linear_model = os.path.join(parent_dir, 'data', 'linear_model')
    fcs.mkdir(path_linear_model)
    path_file = os.path.join(path_linear_model, 'linear_model')
    with open(path_file, 'wb') as file:
        pickle.dump(linear_model, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    data_conversion()