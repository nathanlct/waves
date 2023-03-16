"""
Simulating
    y'(t) = f(y(t), u(t)) avec y(t) = (y_1(t), y_2(t), y_3(t), y_4(t))
Boundary conditions
    Initial condition choosen randomly in [0, K]^4
Observations
    y_3, y_4, t

Example simulation command
    python simulate.py SimODEDiscrete --dt 1e-3

Example training command
    python train.py SimODEDiscrete --kwargs "dict(t_norm=100)" --tmax 100.0 --dt 1e-3 \
        --action_min 0 --action_max "10 * self.sim.K" --cpus 1 --steps 1e9

    python train.py SimODEDiscrete --steps 1e9 --cpus 3 --tmax 4000 --dt 1e-2 --n_steps_per_action 100 --n_past_states 0 --action_min "0" --action_max "10*self.sim.K" --sim_kwargs "dict(obs_time=False, obs_M=True, obs_F=False,  obs_y=False, obs_y0=False, rwd_y123=1, rwd_y4=0.005)" --verbose

"""

from waves.simulation import Simulation

import numpy as np


def normalize(x, xmin, xmax):
    """Transform x from [xmin, xmax] range to [-1, 1] range.
    """
    return (x - xmin) / (xmax - xmin) * 2.0 - 1.0


class SimODEDiscrete(Simulation):
    def __init__(
        self,
        K=50578.0,
        obs_time=False,
        obs_y=False,
        obs_F=False,
        obs_y0=False,
        obs_MMS=False,
        obs_M=False,
        rwd_y123=0,
        rwd_y4=0,
        rwd_y4_last100=0,
        rwd_dyy3=0,
        **kwargs,
    ):
        """
        See parent class.

        K: state space is [0, K]^4, action space is [0, 10*K]
        
        obs_x: whether to add observation "x" to the state space (default False = don't add that state)
        rwd_x: coefficient in front of the "x" term in the reward function (defaut 0 = don't add that term)
        """
        sim_params = {
            # "y0": lambda x: np.random.uniform(low=0.0, high=k, size=(len(x),)),
            # "y0": lambda x: np.random.uniform(low=0.0, high=K, size=(len(x),)),
            "y0": lambda x: np.array([50000, 63748, 153125, 0]),
            "dt": 1e-4,
            "dx": 1,
            "xmin": 0,
            "xmax": 3,
        }
        sim_params.update(kwargs)
        super().__init__(**sim_params)

        if self.xmin != 0 or self.xmax != 3 or self.dx != 1:
            raise ValueError("xmin, xmax and dx cannot be modified in this simulation.")

        self.K = K

        self.obs_time = obs_time
        self.obs_y = obs_y
        self.obs_F = obs_F
        self.obs_M = obs_M
        self.obs_y0 = obs_y0
        self.obs_MMS = obs_MMS
        self.rwd_y123 = rwd_y123
        self.rwd_y4 = rwd_y4
        self.rwd_y4_last100 = rwd_y4_last100
        self.rwd_dyy3 = rwd_dyy3

        print(
            f"Initializing simulation with K={K}, and state space of size {len(self.get_obs())}."
        )

    @property
    def n_controls(self):
        return 1

    def dynamics(self, x=[0, 0, 0, 0], u=[0]):
        """
        Dynamic of the system
        """
        u = np.abs(u[0])

        nu = 0.49  # caractere de differentiation
        nuE = 0.25  # taux d'eclosion
        deltaE = 0.03  # taux d'Oeufs gattes
        deltaS = 0.12  # taux de mort de males steriles
        deltaM = 0.1  # taux de mort de males fertiles
        deltaF = 0.04  # taux de mort de femelle
        gammas = 1  # preference d'accouplement de femelle avec les males fertiles
        betaE = 8  # taux de ponte
        # a = nuE + deltaE
        # b = (1 - nu) * nuE
        # c = betaE * nu * nuE
        # y0 = self.y_lst[0]
        # K = (1 / (1 - ((deltaF * a) / c))) * y0[0]

        # override
        # u = 0.985 * deltaS * (self.y[1] + self.y[3])
        # if x[1] + x[3] <= 4 * self.K:
        #     u = 0.99 * deltaS * (x[1] + x[3])

        assert len(x) == 4
        return np.array(
            [
                betaE * x[2] * (1 - (x[0] / self.K)) - (nuE + deltaE) * x[0],
                (1 - nu) * nuE * x[0] - deltaM * x[1],
                nu * nuE * x[0] * (x[1] / (x[1] + (gammas * x[3]))) - deltaF * x[2],
                u - deltaS * x[3],
            ]
        )

    def update_y(self, u=0):
        current_y = np.copy(self.y)
        self.y = np.array(current_y) + self.dt * self.dynamics(current_y, u)

    def get_obs(self):
        state = []

        if self.obs_time:
            state.append(normalize(self.t, 0, self.tmax))

        if self.obs_y:
            state.append(normalize(self.y[0], 0, 2 * self.K))
            state.append(normalize(self.y[1], 0, 2 * self.K))
            state.append(normalize(self.y[2], 0, 2 * self.K))
            # TODO y[3] can reach much larger values
            # we should probably enforce a max to make sure the observations don't blow
            state.append(normalize(self.y[3], 0, 50 * self.K))

        if self.obs_F:
            state.append(normalize(self.y[2], 0, 2 * self.K))

        if self.obs_M:
            state.append(normalize(self.y[1], 0, 2 * self.K))

        if self.obs_y0:
            state.append(normalize(self.y[0], 0, 2 * self.K))
            state.append(normalize(self.y[1], 0, 2 * self.K))
            state.append(normalize(self.y[2], 0, 2 * self.K))
            state.append(normalize(self.y[3], 0, 50 * self.K))

        if self.obs_MMS:
            state.append(normalize(self.y[1] + self.y[3], 0, 50 * self.K))

        return np.array(state)

    def reward(self):
        reward = 0
        reward_info = {}

        # penalize norm of first three states
        if self.rwd_y123 > 0:
            rwd_y123 = (
                -self.rwd_y123
                * np.linalg.norm([self.y[0], self.y[1], self.y[2]])
                / self.K
            )
            reward_info["rwd_y123"] = rwd_y123
            reward += rwd_y123

        # penalize (norm of) fourth state
        if self.rwd_y4 > 0:
            rwd_y4 = -self.rwd_y4 * self.y[3] / self.K
            reward_info["rwd_y4"] = rwd_y4
            reward += rwd_y4

        # penalize fourth state in the last 100 seconds
        if self.rwd_y4_last100 > 0:
            if self.t > self.tmax - 100:
                rwd_y4_last100 = -self.rwd_y4_last100 * self.y[3] / self.K
            else:
                rwd_y4_last100 = 0
            reward_info["rwd_y4_last100"] = rwd_y4_last100
            reward += rwd_y4_last100

        if self.rwd_dyy3 > 0:
            # rwd_dyy3 = -self.rwd_dyy3 * np.abs(
            #     (self.y_lst[-3][3] - 2 * self.y_lst[-2][3] + self.y_lst[-1][3])
            #     / (self.dt) ** 2
            # )
            # pénaliser la variance sur une durée définie plutôt ?
            rwd_dyy3 = (
                -self.rwd_dyy3
                * np.sqrt(np.var(self.y_lst[-int(np.ceil(10 / self.dt)) :]))
                / self.K
            )
            reward_info["rwd_dyy3"] = rwd_dyy3
            reward += rwd_dyy3

        return reward, reward_info
