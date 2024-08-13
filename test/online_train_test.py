"""Test for online training
model id: 20240813_105549 -> hidden dim = 5 -> tiny
model id: 20240802_141501 -> hidden dim = 17 -> small
model id: 20240809_145528 -> hidden dim = 55 -> small_plus
model id: 20240812_133404 -> hidden dim = 64 -> medium_minus
model id: 20240716_193445 -> hidden dim = 65 -> medium -> work well
model id  20240812_093914 -> hidden dim = 66 -> medium_tiny
model id: 20240809_122329 -> hidden dim = 81 -> medium_pro
model id: 20240808_095020 -> hidden dim = 129 -> medium_plus
model id: 20240805_132954 -> hidden dim = 551 -> large
"""
import os, sys
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from online_learning import OnlineLearning

def test():
    random.seed(9527)
    torch.manual_seed(9527)
    online_learning = OnlineLearning()
    online_learning.online_learning(5000, is_scratch=False)

if __name__ == '__main__':
    test()