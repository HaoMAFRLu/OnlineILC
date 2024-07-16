"""Test for class DataProcess
"""
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_process import DataProcess
from params import PARAMS_GENERATOR

def data_test():
    params_generator = PARAMS_GENERATOR()
    params_generator.get_params('OFFLINE_DATA_PARAMS')
    DATA_PROCESS = DataProcess('offline', params_generator.PARAMS['OFFLINE_DATA_PARAMS'])
    data = DATA_PROCESS.get_data()
    print('here')

if __name__ == "__main__":
    data_test()