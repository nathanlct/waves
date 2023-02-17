"""
Simulating
    y'(t) = f(y(t), u(t)) avec y(t) = (y_1(t), y_2(t), y_3(t), y_4(t))
Boundary conditions
    Initial condition choosen randomly in [0, k]^4
Observations
    y_3, y_4, t

Example simulation command
    python simulate.py SimODEDiscrete --kwargs "dict(k=1)" --dt 1e-3

Example training command
    python train.py SimODEDiscrete --kwargs "dict(k=1, t_norm=2.0)" --tmax 2.0 --dt 1e-3 \
        --action_min 0 --action_max "10 * self.sim.k" --cpus 1 --steps 1e9
"""

from waves.simulation import Simulation
import numpy as np


class SimODEDiscrete(Simulation):
    def __init__(self, k=1.0, t_norm=5.0, y_norm=None, **kwargs):
        """
        See parent class.
        
        k: state space is [0, k]^4, action space is [0, 10*k]
        t_norm: normalization factor for time in the observations
        y_norm: normalization factor for states in the observations (default value is k)
        """
        sim_params = {
            'y0': lambda x: np.random.uniform(low=0.0, high=k, size=(len(x),)),
            'dt': 1e-4,
            'dx': 1,
            'xmin': 0,
            'xmax': 3,
        }
        sim_params.update(kwargs)
        super().__init__(**sim_params)
        
        if self.xmin != 0 or self.xmax != 3 or self.dx != 1:
            raise ValueError('xmin, xmax and dx cannot be modified in this simulation.')
        
        self.k = k
        self.t_norm = t_norm
        self.y_norm = y_norm or self.k
    
    @property
    def n_controls(self):
        return 1

    def update_y(self, u=0):
        # TODO, compute y_t+1 as a function of y_t (current_y[i] for i=0,1,2,3) and control u (in [0, 10*k], cf action_min and action_max)
        current_y = np.copy(self.y)
        self.y[0] = 0
        self.y[1] = 0
        self.y[2] = 0
        self.y[3] = 0

    def get_obs(self):
        return np.array([
            self.t / self.t_norm,
            self.y[2] / self.y_norm,
            self.y[3] / self.y_norm,
        ])
