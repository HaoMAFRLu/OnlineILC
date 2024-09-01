import numpy as np
import os, sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

root = fcs.get_parent_path(lvl=1)
file = os.path.join(root, 'params_gradient.txt')
value_alpha = [f"{10**-i:.0e}" for i in range(0, 3)]
value_epsilon = [f"{10**-i:.0e}" for i in range(0, 3)]
value_eta = [f"{x:.1f}" for x in [0.1, 0.3, 0.5, 0.7, 0.9]]

combinations = itertools.product(value_alpha, value_epsilon, value_eta)

# 打开文件以写入参数
with open(file, 'w') as f:
    for combo in combinations:
        f.write(" ".join(combo) + "\n")




