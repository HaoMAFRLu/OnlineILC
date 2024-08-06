import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from params import PARAMS_GENERATOR
from environmnet import BEAM


def get_random_input(T: int, dt: float, 
                     mean: float=0.0, std: float=1.0):
    l = int(T/dt)
    u = np.random.normal(mean, 50, size=l)
    t = np.array(range(l))*dt
    u_in = np.stack((t, u), axis=1)
    return u_in

def run_simulink_model():
    nr_test = 10
    PARAMS_LIST = ["SIM_PARAMS"]     
    params_generator = PARAMS_GENERATOR()
    params_generator.get_params(PARAMS_LIST)
    # params_generator.PARAMS['SIM_PARAMS']["SimulationMode"] = "accelerator"

    model_name = 'Control_System'
    beam = BEAM(model_name, params_generator.PARAMS['SIM_PARAMS'])
    beam.initialization()
    u_in = get_random_input(5.5, 0.01)
    beam.set_input('dt', 0.01)
    beam.set_input('u_in', u_in)
    t_total = 0
    for i in range(nr_test):
        t1 = time.time()
        beam.run_sim()
        simOut = beam.get_output()
        t2 = time.time()
        tt = t2 - t1
        print(tt)
        t_total += tt
    print(t_total/nr_test)
    

if __name__ == "__main__":
    run_simulink_model()
