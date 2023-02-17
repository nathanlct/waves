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
            "y0": lambda x: np.random.uniform(low=0.0, high=k, size=(len(x),)),
            "dt": 1e-4,
            "dx": 1,
            "xmin": 0,
            "xmax": 3,
        }
        sim_params.update(kwargs)
        super().__init__(**sim_params)

        if self.xmin != 0 or self.xmax != 3 or self.dx != 1:
            raise ValueError("xmin, xmax and dx cannot be modified in this simulation.")

        self.k = k
        self.t_norm = t_norm
        self.y_norm = y_norm or self.k

        # Store initial condition
        self.y0_value = self.y

    @property
    def n_controls(self):
        return 1

    def dynamics(self, x=[0, 0, 0, 0], u=0):
        """
        Dynamic of the system
        """
        nu = 0.49  # caractere de differentiation
        nuE = 0.25  # taux d'eclosion
        deltaE = 0.03  # taux d'Oeufs gattes
        deltaS = 0.12  # taux de mort de males steriles
        deltaM = 0.1  # taux de mort de males fertiles
        deltaF = 0.04  # taux de mort de femelle
        mus = 0.06
        gammas = 1  # preference d'accouplement de femelle avec les males fertiles
        betaE = 8  # taux de ponte
        a = nuE + deltaE
        b = (1 - nu) * nuE
        c = betaE * nu * nuE
        K = (1 / (1 - ((deltaF * a) / c))) * self.y0_value
        print("Here")
        print(self.y0_value)

        assert len(x) == 4
        return np.array(
            [
                betaE * x[2] * (1 - (x[0] / K)) - (nuE + deltaE) * x[0],
                (1 - nu) * nuE * x[0] - deltaM * x[1],
                nu * nuE * x[0] * (x[1] / (x[1] + (gammas * x[3]))) - deltaF * x[2],
                u - deltaS * x[3],
            ]
        )

    def update_y(self, u=0):
        current_y = np.copy(self.y)
        self.y = list(np.array(current_y) + self.dt * self.dynamics(current_y, u))

    def get_obs(self):
        return np.array(
            [self.t / self.t_norm, self.y[2] / self.y_norm + self.y[3] / self.y_norm,]
        )
