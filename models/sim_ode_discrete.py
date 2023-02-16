"""
Simulating
    y'(t) = f(y(t), u(t)) avec y(t) = (y_1(t), y_2(t), y_3(t), y_4(t))
Boundary conditions
    Initial condition choosen randomly in [0, k]^4
Observations
    y_3, y_4, t
"""

from waves.simulation import Simulation
import numpy as np


class SimODEDiscrete(Simulation):
    def __init__(self, k=1.0, **kwargs):
        sim_params = {
            'y0': lambda x: np.random.uniform(low=0.0, high=k, size=(len(x),)),
            'dt': 1e-4,
            'dx': 1,
            'xmin': 0,
            'xmax': 3,
        }
        sim_params.update(kwargs)
        super().__init__(**sim_params)
    
    @property
    def n_controls(self):
        return 1

    def update_y(self, u=0):
        # TODO, compute y_t+1 as a function of y_t (current_y) and control u
        current_y = np.copy(self.y)
        self.y[0] = 0
        self.y[1] = 0
        self.y[2] = 0
        self.y[3] = 0

    def get_obs(self):
        return np.array([self.t, self.y[2], self.y[3]])  # TODO need normalization
