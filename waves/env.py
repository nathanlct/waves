from abc import ABC
import numpy as np
import gym


class WavesEnv(gym.Env, ABC):
    def __init__(self, sim_class, sim_kwargs, config):
        super().__init__()

        self.sim = sim_class(**sim_kwargs)

        # load config
        self.config = config
        self.tmax = config['tmax']

        # memory
        self.sim.reset()
        self.n_observations_base = self.sim.get_obs().shape[0]
        self.n_past_states = config['n_past_states']
        self.reset_memory()

        # observation space
        self.n_observations = self.n_observations_base * (1 + self.n_past_states)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_observations,),
            dtype=np.float32,
        )

        # action space
        self.n_actions = self.sim.n_controls
        self.action_min = eval(config['action_min'])
        self.action_max = eval(config['action_max'])
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_actions,),
            dtype=np.float32
        )

        # reset env
        self.reset()

    def step(self, action):
        # convert action from [-1, 1] to [action_min, action_max]
        action = (action + 1.0) * self.action_max / 2.0 - self.action_min
        
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
        return max(-1, -self.sim.norm_y() / 100)

    def compute_done(self):
        return self.sim.t >= self.tmax

    def reset_memory(self):
        self.memory = np.zeros(self.n_observations_base * self.n_past_states)

    def reset(self):
        # reset sim
        self.sim.reset()

        # reset memory
        self.reset_memory()

        # get initial state
        s0 = self.get_state(append_to_memory=True)

        return s0
