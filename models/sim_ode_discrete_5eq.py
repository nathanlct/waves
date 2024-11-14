"""
Simulating
    y'(t) = f(y(t), u(t)) avec y(t) = (y_1(t), y_2(t), y_3(t), y_4(t), y_5(t))
Boundary conditions
    Initial condition choosen randomly in [0, K]^4
Observations
    y_3, y_4, t

Example simulation command
    python simulate.py SimODEDiscrete --dt 1e-3

Example training command
    python train.py SimODEDiscrete --kwargs "dict(t_norm=100)" --tmax 100.0 --dt 1e-3 \
        --action_min 0 --action_max "10 * self.sim.K" --cpus 1 --steps 1e9
"""

import numpy as np

from waves.simulation import Simulation


def normalize(x, xmin, xmax):
    """Transform x from [xmin, xmax] range to [-1, 1] range."""
    return (x - xmin) / (xmax - xmin) * 2.0 - 1.0


class SimODEDiscrete5Eq(Simulation):
    def __init__(
        self,
        K=50000.0,
        obs_time=False,
        obs_y=False,
        obs_y0=False,
        obs_MMS=False,
        obs_MS=False,
        obs_F=False,
        obs_all_females=False,
        rwd_y1=0,
        rwd_y2=0,
        rwd_y3=0,
        rwd_y4=0,
        rwd_y4_last100=0,
        rwd_y5=0,
        rwd_u=0,
        **kwargs,
    ):
        """
        See parent class.

        K: state space is [0, K]^4, action space is [0, 10*K]

        obs_x: whether to add observation "x" to the state space (default False = don't add that state)
        rwd_x: coefficient in front of the "x" term in the reward function (defaut 0 = don't add that term)
        """
        sim_params = {
            "y0": lambda x: np.random.uniform(low=0.0, high=5 * K, size=(len(x),)),
            # "y0": lambda x: np.array([50000, 63748, 153125, 0]),
            "dt": 1e-4,
            "dx": 1,
            "xmin": 0,
            "xmax": 4,
        }
        sim_params.update(kwargs)
        super().__init__(**sim_params)

        if self.xmin != 0 or self.xmax != 4 or self.dx != 1:
            raise ValueError("xmin, xmax and dx cannot be modified in this simulation.")

        self.K = K

        self.obs_time = obs_time
        self.obs_y = obs_y
        self.obs_y0 = obs_y0
        self.obs_MMS = obs_MMS
        self.obs_MS = obs_MS
        self.obs_F = obs_F
        self.obs_all_females = obs_all_females
        self.rwd_y1 = rwd_y1
        self.rwd_y2 = rwd_y2
        self.rwd_y3 = rwd_y3
        self.rwd_y4 = rwd_y4
        self.rwd_y4_last100 = rwd_y4_last100
        self.rwd_y5 = rwd_y5
        self.rwd_u = rwd_u

        # model parameters
        self.nu = 0.49  # caractere de differentiation
        self.nuE = 0.25  # taux d'eclosion
        self.deltaE = 0.03  # taux d'Oeufs gattes
        self.deltaS = 0.12  # taux de mort de males steriles
        self.deltaM = 0.1  # taux de mort de males fertiles
        self.deltaF = 0.04  # taux de mort de femelle
        self.mus = 0.06
        self.gammas = 1  # preference d'accouplement de femelle avec les males fertiles
        self.betaE = 8  # taux de ponte

        # print(f"Initializing simulation with K={K}, and state space of size {len(self.get_obs())}.")

    @property
    def n_controls(self):
        return 1

    def dynamics(self, x=[0, 0, 0, 0, 0], u=[0]):
        """
        Dynamic of the system
        """
        u = np.abs(u[0])

        # a = nuE + deltaE
        # b = (1 - nu) * nuE
        # c = betaE * nu * nuE
        # y0 = self.y_lst[0]
        # K = (1 / (1 - ((deltaF * a) / c))) * y0[0]

        assert len(x) == 5
        return np.array(
            [
                self.betaE * x[2] * (1 - (x[0] / self.K)) - (self.nuE + self.deltaE) * x[0],
                (1 - self.nu) * self.nuE * x[0] - self.deltaM * x[1],
                self.nu * self.nuE * x[0] * (x[1] / (x[1] + (self.gammas * x[3]))) - self.deltaF * x[2],
                u - self.deltaS * x[3],
                self.nu * self.nuE * (x[3] /  (x[1] + (self.gammas * x[3]))) - self.deltaF * x[4],
            ]
        )

    def update_y(self, u=0):
        current_y = np.copy(self.y)
        self.y = np.array(current_y) + self.dt * self.dynamics(current_y, u)

    def get_obs(self, **kwargs):
        state = []

        y = self.y

        if self.obs_time:
            state.append(normalize(self.t, 0, self.tmax))

        if self.obs_y:
            state.append(normalize(y[0], 0, 2 * self.K))
            state.append(normalize(y[1], 0, 2 * self.K))
            state.append(normalize(y[2], 0, 2 * self.K))
            # TODO y[3] can reach much larger values
            # we should probably enforce a max to make sure the observations don't blow up
            state.append(normalize(y[3], 0, 50 * self.K))
            state.append(normalize(y[4], 0, 50 * self.K))

        if self.obs_y0:
            state.append(normalize(y[0], 0, 2 * self.K))
            state.append(normalize(y[1], 0, 2 * self.K))
            state.append(normalize(y[2], 0, 2 * self.K))
            state.append(normalize(y[3], 0, 50 * self.K))
            state.append(normalize(y[4], 0, 50 * self.K))

        if self.obs_MMS:
            mms = kwargs.get("MMS", y[1] + y[3])
            state.append(normalize(mms, 0, 100 * self.K))
            state.append(normalize(min(mms, 50 * self.K), 0, 50 * self.K))
            state.append(normalize(min(mms, 10 * self.K), 0, 10 * self.K))
            state.append(normalize(min(mms, self.K), 0, self.K))
            state.append(normalize(min(mms, 5000), 0, 5000))
            state.append(normalize(min(mms, 500), 0, 500))
            state.append(normalize(min(mms, 50), 0, 50))
            state.append(normalize(min(mms, 5), 0, 5))

        if self.obs_MS:
            state.append(normalize(y[3], 0, 50 * self.K))

        if self.obs_F:
            state.append(normalize(y[2], 0, 2 * self.K))
            state.append(normalize(min(y[2], 5000), 0, 5000))
            state.append(normalize(min(y[2], 500), 0, 500))
            state.append(normalize(min(y[2], 50), 0, 50))
            state.append(normalize(min(y[2], 5), 0, 5))

        if self.obs_all_females:
            # F*(M+\gamma_s M_s)/M
            all_females = kwargs.get('F', y[2] + y[4])
            state.append(normalize(all_females, 0, 100 * self.K))
            state.append(normalize(min(all_females, 50 * self.K), 0, 50 * self.K))
            state.append(normalize(min(all_females, 10 * self.K), 0, 10 * self.K))
            state.append(normalize(min(all_females, self.K), 0, self.K))
            state.append(normalize(min(all_females, 5000), 0, 5000))
            state.append(normalize(min(all_females, 500), 0, 500))
            state.append(normalize(min(all_females, 50), 0, 50))
            state.append(normalize(min(all_females, 5), 0, 5))

        return np.array(state)

    def reward(self, action):
        reward = 0
        reward_info = {}

        # penalize norm of first three states and last state
        if self.rwd_y1 > 0:
            rwd_y1 = -self.rwd_y1 * self.y[0] / self.K
            reward_info["rwd_y1"] = rwd_y1
            reward += rwd_y1
        if self.rwd_y2 > 0:
            rwd_y2 = -self.rwd_y2 * self.y[1] / self.K
            reward_info["rwd_y2"] = rwd_y2
            reward += rwd_y2
        if self.rwd_y3 > 0:
            rwd_y3 = -self.rwd_y3 * self.y[2] / self.K
            reward_info["rwd_y3"] = rwd_y3
            reward += rwd_y3
        if self.rwd_y4 > 0:
            rwd_y4 = -self.rwd_y4 * self.y[3] / self.K
            reward_info["rwd_y4"] = rwd_y4
            reward += rwd_y4
        if self.rwd_y5 > 0:
            rwd_y5 = -self.rwd_y5 * self.y[4] / self.K
            reward_info["rwd_y5"] = rwd_y5
            reward += rwd_y5
        
        if self.rwd_u > 0:
            rwd_u = -self.rwd_u * float(action) / self.K
            reward_info["rwd_u"] = rwd_u
            reward += rwd_u

        # penalize fourth state in the last 100 seconds
        if self.rwd_y4_last100 > 0:
            if self.t > self.tmax - 100:
                rwd_y4_last100 = -self.rwd_y4_last100 * self.y[3] / self.K
            else:
                rwd_y4_last100 = 0
            reward_info["rwd_y4_last100"] = rwd_y4_last100
            reward += rwd_y4_last100

        return reward, reward_info
