"""Test for online training
model id: 20240802_141501 -> hidden dim = 17
model id: 20240716_193445 -> hidden dim = 65
model id: 20240805_132954 -> hidden dim = 551
"""
import os, sys
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from online_learning import OnlineLearning

def test():
    random.seed(9527)
    torch.manual_seed(9527)
    online_learning = OnlineLearning('ada-svd')
    online_learning.online_learning(1000)

if __name__ == '__main__':
    test()