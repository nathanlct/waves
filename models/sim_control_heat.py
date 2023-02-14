"""
Simulating
    y_t - y_xx = y^3   
Boundary conditions
    y(t,0) = u1
    y(t,1) = u2
    for some control u = (u1, u2)
    that can observe y(0,x) and t
"""

from waves.simulation import Simulation
import numpy as np


class SimControlHeat(Simulation):
    def __init__(self, **kwargs):
        sim_params = {
            'y0': lambda x: 3 * np.sin(np.pi * x),
            'dt': 0.45e-4,
            'dx': 1e-2,
            'xmin': 0,
            'xmax': 1,
        }
        sim_params.update(kwargs)
        super().__init__(**sim_params)
    
    @property
    def n_controls(self):
        return 2

    def update_y(self, u=(0, 0)):
        # boundary conditions
        u1, u2 = u
        self.y[0] = u1
        self.y[-1] = u2

        # compute y_xx = (y(t,n+1)-2y(t,n)+y(t,n-1))/dx^2
        yxx = np.diff(self.y, n=2) / (self.dx * self.dx)

        # compute y^3
        y3 = np.power(self.y, 3)[1:-1]

        # y_t = (y(t+dt) - y(t)) / dt => y(t+dt) = y(t) + dt * (y^3 + y_xx)
        self.y[1:-1] += self.dt * (y3 + yxx)

    def get_obs(self):
        return np.concatenate(([self.t], self.data['y'][0]))
