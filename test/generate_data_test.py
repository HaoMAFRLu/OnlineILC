"""Test for class DataProcess
"""
import os, sys
import numpy as np
from dataclasses import dataclass, asdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_process import DataProcess
from params import PRETRAIN_PARAMS

def data_test():
    PARAMS = PRETRAIN_PARAMS(
        mode='seq2seq',
        k=0.8,
        batch_size=20,
        input_name='yref',
        output_name='d',
        channel=1,
        height=550,
        width=1
    )
    DATA_PROCESS = DataProcess(**asdict(PARAMS))
    data = DATA_PROCESS.get_data('offline')
    print('here')

if __name__ == "__main__":
    data_test()