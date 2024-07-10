"""Test for class DataProcess
"""
import os, sys
import numpy as np
from dataclasses import asdict
import random
import torch
import argparse
import time

random.seed(10086)
torch.manual_seed(10086)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_process import DataProcess
from pretraining import PreTrain
from params import OFFLINE_DATA_PARAMS, NN_PARAMS 

def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available.")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("GPU is not available.")

def test():
    parser = argparse.ArgumentParser(description='offline training')
    parser.add_argument('num_epoch', type=int, help='number of training epoch')
    args = parser.parse_args()

    DATA_PARAMS = OFFLINE_DATA_PARAMS(
        mode='seq2seq',
        k=0.8,
        batch_size=20,
        input_name='u',
        output_name='y',
        channel=1,
        height=550,
        width=1
    )

    PARAMS = NN_PARAMS(
        is_initialization=False,
        loss_function='Huber',
        lambda_regression=0.0,
        learning_rate=1e-4,
        weight_decay=0.0,
        channel=1,
        height=550,
        width=1,
        filter_size=5,
        output_dim=550
    )

    DATA_PROCESS = DataProcess(**asdict(DATA_PARAMS))
    data = DATA_PROCESS.get_data('offline', is_normalization=True)
    
    PRE_TRAIN = PreTrain(mode='seq2seq', **asdict(PARAMS))
    PRE_TRAIN.import_data(data)

    t_start = time.time()
    PRE_TRAIN.learn(num_epochs=args.num_epoch)
    t_end = time.time()
    total_time = t_end - t_start

    print(f"Total time: {total_time} seconds")

if __name__ == "__main__":
    test()
    check_gpu()