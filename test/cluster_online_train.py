"""Test for online training
model id: 20240813_105549 -> hidden dim = 5 -> tiny
model id: 20240802_141501 -> hidden dim = 17 -> small
model id: 20240813_234753 -> hidden dim = 33 -> small_pro
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
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from online_learning import OnlineLearning
import utils as fcs

# def update_config(file_path, param_path, new_value):
#     with open(file_path, 'r') as file:
#         config = json.load(file)

#     keys = param_path.split('.')
#     sub_config = config
#     for key in keys[:-1]:
#         sub_config = sub_config[key]
#     sub_config[keys[-1]] = new_value

#     with open(file_path, 'w') as file:
#         json.dump(config, file, indent=4)

def test():
    root = fcs.get_parent_path(lvl=0)
    path = os.path.join(root, 'config.json')

    random.seed(9527)
    torch.manual_seed(9527)

    parser = argparse.ArgumentParser(description="Online Training")
    parser.add_argument('--sigma-w', type=float, required=True, help="Sigma w")
    parser.add_argument('--sigma-y', type=float, required=True, help="Sigma y")
    parser.add_argument('--sigma-d', type=float, required=True, help="Sigma d")
    parser.add_argument('--sigma-ini', type=float, required=True, help="Sigma initial")
    args = parser.parse_args()

    # update_config(path, 'sigma_y', args.sigma_y)
    # update_config(path, 'sigma_d', args.sigma_d)
    # update_config(path, 'sigma_ini', args.sigma_ini)

    folder_name = str(args.sigma_w)+'_'+str(args.sigma_y)+'_'+str(args.sigma_d)+'_'+str(args.sigma_ini)

    online_learning = OnlineLearning(mode='svd',
                                     rolling=1,
                                     location='cluster',
                                     folder_name=folder_name,
                                     sigma_w=args.sigma_w,
                                     sigma_y=args.sigma_y,
                                     sigma_d=args.sigma_d,
                                     sigma_ini=args.sigma_ini)
    
    online_learning.online_learning(6000, 
                                    is_shift_dis=True,
                                    is_scratch=True)

if __name__ == '__main__':
    test()