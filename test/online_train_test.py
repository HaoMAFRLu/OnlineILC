"""Test for online training
model id: 20240802_141501 -> hidden dim = 17
model id: 20240716_193445 -> hidden dim = 65
"""
import os, sys
import torch
import random

random.seed(9527)
torch.manual_seed(9527)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from online_learning import OnlineLearning

def test():
    online_learning = OnlineLearning('svd')
    online_learning.online_learning(500)

if __name__ == '__main__':
    test()