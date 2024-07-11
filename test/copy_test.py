import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

p1 = "/home/hao/Desktop/MPI/Online_Convex_Optimization/OnlineILC/src"
p2 = "/home/hao/Desktop/MPI/Online_Convex_Optimization/OnlineILC/data/offline_training/20240711_214531"
fcs.copy_folder(p1, p2)