import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from params import Beam_SIM_PARAMS
from environmnet import BEAM


def get_random_input(T: int, dt: float, 
                     mean: float=0.0, std: float=1.0):
    l = int(T/dt)
    u = np.random.normal(mean, std, size=l)
    t = np.array(range(l))*dt
    u_in = np.stack((t, u), axis=1)
    return u_in

def run_simulink_model():
    SIM_PARAMS = Beam_SIM_PARAMS(
        StopTime='1',
        StartTime='0.0',
        AbsTol='1e-7',
        Solver='ode23t',
        SimulationMode='normal'
    )
    model_name = 'Control_System'
    beam = BEAM(model_name, SIM_PARAMS)
    beam.initialization()
    u_in = get_random_input(1, 0.01)
    beam.set_input('dt', 0.01)
    beam.set_input('u_in', u_in)
    beam.run_sim()
    simOut = beam.get_output()

if __name__ == "__main__":
    run_simulink_model()
