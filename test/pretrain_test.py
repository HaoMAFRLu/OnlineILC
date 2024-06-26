"""Test for class DataProcess
"""
import os, sys
import numpy as np
from dataclasses import dataclass, asdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_process import DataProcess
from pretraining import PreTrain
from params import OFFLINE_DATA_PARAMS, NN_PARAMS 

def test():
    DATA_PARAMS = OFFLINE_DATA_PARAMS(
        mode='seq2seq',
        k=0.8,
        batch_size=20,
        input_name='yref',
        output_name='d',
        channel=1,
        height=550,
        width=1
    )

    PARAMS = NN_PARAMS(
        is_initialization=False,
        loss_function='Huber',
        learning_rate=1e-3,
        weight_decay=0.0,
        channel=1,
        height=550,
        width=1,
        filter_size=7,
        output_dim=550
    )

    DATA_PROCESS = DataProcess(**asdict(DATA_PARAMS))
    data = DATA_PROCESS.get_data('offline')
    
    PRE_TRAIN = PreTrain(mode='seq2seq', **asdict(PARAMS))
    PRE_TRAIN.import_data(data)
    PRE_TRAIN.train()

if __name__ == "__main__":
    test()