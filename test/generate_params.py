import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

root = fcs.get_parent_path(lvl=1)
values = np.logspace(-3, -8, num=6)  # from 1e-1 to 1e-8
file = os.path.join(root, 'params.txt')

with open(file, 'w') as f:
    for v1 in values:
        for v2 in values:
            for v3 in values:
                f.write(f"{v1} {v2} {v3}\n")
