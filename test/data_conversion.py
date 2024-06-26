"""Convert matlab data to python
"""

import scipy.io
import os
import pickle5
import numpy as np

def data_conversion():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_dir = os.path.join(parent_dir, 'data', 'ilc')
    file_names = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    num_files = len(file_names) - 1
    data = [None] * num_files
    for i in range(num_files):
        name = str(i+1) + '.mat'
        path_data = os.path.join(data_dir, name)
        mat_data = scipy.io.loadmat(path_data)
        sim_result = mat_data["sim_result"]
        yref = sim_result['yref'][0, 0].flatten().astype(np.float32)
        d = sim_result['d'][0, 0].flatten().astype(np.float32)
        u = sim_result['u'][0, 0].flatten().astype(np.float32)
        data[i] = (yref, d, u)
    print('here')

if __name__ == "__main__":
    data_conversion()