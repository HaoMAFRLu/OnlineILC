from src.environmnet import Beam
from src.sim_params import Beam_SIM_PARAMS

def run_simulink_model():
    SIM_PARAMS = Beam_SIM_PARAMS(
        StopTime='5',
        StartTime='0.0',
        AbsTol='1e-7',
        Solver='ode23t',
        SimulationMode='normal'
    )
    model_name = 'Control_System'
    beam = Beam(model_name, SIM_PARAMS)

if __name__ == "__main__":
    run_simulink_model()
