"""Test for class DataProcess
"""
import os, sys
import random
import torch
import argparse
import time

random.seed(10086)
torch.manual_seed(10086)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from pretraining import PreTrain
from params import PARAMS_GENERATOR
import transformer_data_generation as tdg

def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available.")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("GPU is not available.")

def test():
    # parser = argparse.ArgumentParser(description='offline training')
    # parser.add_argument('num_epoch', type=int, help='number of training epoch')
    # args = parser.parse_args()

    PARAMS_LIST = ["NN_PARAMS"]
    params_generator = PARAMS_GENERATOR()
    params_generator.get_params(PARAMS_LIST)

    data = tdg.get_data()
    
    PRE_TRAIN = PreTrain(params_generator.PARAMS['NN_PARAMS'])
    PRE_TRAIN.import_data(data)

    t_start = time.time()
    # PRE_TRAIN.learn(num_epochs=args.num_epoch)
    PRE_TRAIN.learn(num_epochs=10)
    t_end = time.time()
    total_time = t_end - t_start

    print(f"Total time: {total_time} seconds")

if __name__ == "__main__":
    test()
    check_gpu()