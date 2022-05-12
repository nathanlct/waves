import gym
from gym import spaces
from simulation import SimStabilizeMidObs, SimControlHeat
import numpy as np
from abc import ABC


DEFAULT_ENV_CONFIG = dict(
    dt=0.95 * 1e-2,  # time delta (in seconds)
    dx=1e-2,  # space delta
    xmin=0,  # space lower bound
    xmax=1,  # space upper bound
    y0=lambda _: 0,  # initial condition function (as a function of x)
    y0_generator=None,  # function called at each reset, that takes in no parameters and
                        # that should return an y0 initial condition function that depends on x 
                        # if None, the specified y0 is used all throughout training and never changes
    action_min=-1.0,  # minimum control value
    action_max=1.0,  # maximum control value
    n_past_states=0,  # number of previous states to add in the current state (memory)
    t_max=2.0,  # max duration of each episode (in seconds)
)


class WaveEnv(gym.Env, ABC):
    def __init__(self, sim, config):
        super().__init__()

        self.sim = sim

        # load config
        self.config = config
        self.t_max = config['t_max']

        # initial conditions
        self.y0 = config['y0']
        self.y0_generator = config['y0_generator']

        # memory
        self.sim.reset()
        self.n_observations_base = self.sim.get_obs().shape[0]
        self.n_past_states = config['n_past_states']
        self.reset_memory()

        # observation space
        self.n_observations = self.n_observations_base * (1 + self.n_past_states)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_observations,),
            dtype=np.float32,
        )

        # action space
        self.n_actions = self.sim.n_controls
        self.action_min = config['action_min']
        self.action_max = config['action_max']
        self.action_space = spaces.Box(
            low=self.action_min,
            high=self.action_max,
            shape=(self.n_actions,),
            dtype=np.float32
        )

        # reset env
        self.reset()

    def step(self, action):
        # step simulation with control
        self.sim.step(u=action)

        # update state
        state = self.get_state(append_to_memory=True)

        # compute reward
        reward = self.compute_reward(action)

        # compute done
        done = self.compute_done()

        return state, reward, done, {}

    def get_base_state(self):
        return self.sim.get_obs()

    def get_state(self, append_to_memory=False):
        base_state = self.get_base_state()
        state = np.concatenate((base_state, self.memory))

        if append_to_memory and self.n_past_states > 0:
            self.memory = np.roll(self.memory, self.n_observations_base)
            self.memory[:self.n_observations_base] = base_state

        return state

    def compute_reward(self, action):
        return max(-10, -self.sim.norm_y())

    def compute_done(self):
        return self.sim.t >= self.t_max

    def reset_memory(self):
        self.memory = np.zeros(self.n_observations_base * self.n_past_states)

    def reset(self):
        # get new initial condition
        if self.y0_generator is not None:
            y0 = self.y0_generator()
        else:
            y0 = self.y0
            
        # reset simulation
        self.sim.reset(y0=y0)

        # reset memory
        self.reset_memory()

        # get initial state
        s0 = self.get_state(append_to_memory=True)

        return s0


class SimControlHeatEnv(WaveEnv):
    def __init__(self, config=dict()):
        ADDITIONAL_DEFAULT_CONFIG = dict(
            y0=lambda x: 5 * np.sin(np.pi * x),
        )
        config = DEFAULT_ENV_CONFIG | ADDITIONAL_DEFAULT_CONFIG | config

        sim = SimControlHeat(
            dt=config['dt'], dx=config['dx'],
            xmin=config['xmin'], xmax=config['xmax'],
            y0=config['y0'],
        )

        super().__init__(sim, config)


class SimStabilizeMidObsEnv(WaveEnv):
    def __init__(self, config=dict()):
        ADDITIONAL_DEFAULT_CONFIG = dict(
            f=lambda x: x + 1e-3 * (x * x),
            y0_generator=self._sin_cos_combination,
        )
        config = DEFAULT_ENV_CONFIG | ADDITIONAL_DEFAULT_CONFIG | config

        sim = SimStabilizeMidObs(
            f=config['f'],
            dt=config['dt'], dx=config['dx'],
            xmin=config['xmin'], xmax=config['xmax'],
            y0=config['y0'],
        )

        super().__init__(sim, config)

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


if __name__ == "__main__":
    env = SimControlHeatEnv()

    state = env.reset()
    while True:
        state, reward, done, _ = env.step((0, 0))
        if done:
            break
