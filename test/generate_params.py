import numpy as np
import os, sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

root = fcs.get_parent_path(lvl=1)
file = os.path.join(root, 'params.txt')
values = [f"{10**-i:.0e}" for i in range(3, 8)]
combinations = itertools.product(values, repeat=3)

# 打开文件以写入参数
with open(file, 'w') as f:
    for combo in combinations:
        f.write(" ".join(combo) + "\n")




