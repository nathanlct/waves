"""
Simulating
    y_t + f(y)_x = 0
    y(t,0) = y(t,1) + u(t, y(t,1/2))
for some control u
"""

from waves.simulation import Simulation
import numpy as np


class SimStabilizeMidObs(Simulation):
    def __init__(self, f=lambda x: x + 1e-3 * (x * x), **kwargs):
        sim_params = {
            'y0_generator': self._sin_cos_combination,
            'dt': 3e-4,
            'dx': 1e-3,
            'xmin': 0,
            'xmax': 1,
        }
        sim_params.update(kwargs)
        super().__init__(**sim_params)
        self.f = f
    
    @property
    def n_controls(self):
        return 1

    def update_y(self, u=0):
        # boundary condition with control u dt
        self.y[0] = self.y[-1] + u  # - 0.2 * self.y[len(self.y) // 2] + u

        # compute df(y)/dx
        yx = np.diff(self.f(self.y)) / self.dx

        # dy/dt + df(y)/dx = 0  =>  y(t+dt) = y(t) - dt * df(y)/dx
        self.y[1:] -= self.dt * yx

    def get_obs(self):
        return np.array([self.t, self.y[len(self.y) // 2]])

    def _sin_cos_combination(self, n=100, amplitude=5.0):
        """
        Generate an initial condition y0 = sum_n a_n sin(nx) + bn cos(nx)
        with all a_n, b_n in [0,1) and sum_n (a_n + b_n) = amplitude
        """
        # generate coefficients a_n, b_n
        coefs = np.random.random(2 * n)
        # normalize coefficients
        coefs *= amplitude / np.sum(coefs)
        # generate function
        return lambda x: np.sum(
            coefs[:n] * np.sin(np.arange(n) * x) + coefs[n:] * np.cos(np.arange(n) * x)
        )

