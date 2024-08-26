import numpy as np
import os, sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

root = fcs.get_parent_path(lvl=1)
file = os.path.join(root, 'params.txt')
value_w = ['0']
# value_y = [f"{10**-i:.0e}" for i in range(3, 9)]
value_y = ['1e-03']
value_d = [f"{10**-i:.0e}" for i in range(3, 10)]
value_ini = [f"{10**-i:.0e}" for i in range(3, 10)]

combinations = itertools.product(value_w, value_y, value_d, value_ini)

# 打开文件以写入参数
with open(file, 'w') as f:
    for combo in combinations:
        f.write(" ".join(combo) + "\n")




