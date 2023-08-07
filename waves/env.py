from abc import ABC
import numpy as np
import gym


class WavesEnv(gym.Env, ABC):
    def __init__(self, sim_class, sim_kwargs, config):
        super().__init__()

        # load config
        self.config = config
        self.tmax = config["tmax"]
        self.n_steps_per_action = config["n_steps_per_action"]

        # create sim
        self.sim = sim_class(**sim_kwargs, tmax=self.tmax)

        # memory
        self.sim.reset()
        self.n_observations_base = self.sim.get_obs().shape[0]
        self.mem_n_past_states = config["mem_n_past_states"]
        self.mem_default_value = config["mem_default_value"]
        self.mem_save_every = config["mem_save_every"]
        self.reset_memory()

        # observation space
        self.n_observations = self.n_observations_base * \
            (1 + self.mem_n_past_states)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_observations,), dtype=np.float32,
        )

        # action space
        self.n_actions = self.sim.n_controls
        self.action_min = eval(config["action_min"])
        self.action_max = eval(config["action_max"])
        self.discrete = config["discrete"]
        self.discrete_n_actions = config["discrete_n_actions"]
        if self.discrete:
            self.action_space = gym.spaces.Discrete(self.discrete_n_actions)
        else:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
            )

        # reset env
        self.reset()

    def normalize_action(self, action):
        if self.discrete:
            # convert action in {0, ..., discrete_n_actions - 1} to [action_min, action_max]
            action = action / self.discrete_n_actions * \
                (self.action_max - self.action_min) + self.action_min
            action = np.array([action])
        else:
            # convert action from [-1, 1] to [action_min, action_max]
            action = (action + 1.0) * (
                self.action_max - self.action_min
            ) / 2.0 + self.action_min
        return action

    def step(self, action, normalize_action=True):
        if normalize_action:
            action = self.normalize_action(action)

        # step simulation with control
        for _ in range(self.n_steps_per_action):
            self.sim.step(u=action)

        # add one environment step to memory counter
        self.steps_since_last_appended_in_memory += 1

        # update state
        state = self.get_state(append_to_memory=True)

        # compute reward
        reward, reward_info = self.compute_reward(action)

        info = {
            'scaled_action': action,
            'reward_info': reward_info,
        }

        # compute done
        done = False
        if self.sim.t >= self.tmax:
            done = True
            # for SB3 end-of-horizon bootstrapping
            info["TimeLimit.truncated"] = True

        # end early if norm blows up
        if self.sim.norm_y() > 1000 * self.sim.K:
            done = True
            reward -= 100

        return state, reward, done, info

    def get_base_state(self):
        return self.sim.get_obs()

    def get_state(self, append_to_memory=False):
        base_state = self.get_base_state()
        state = np.concatenate((base_state, self.memory))

        if append_to_memory and self.mem_n_past_states > 0 \
                and self.steps_since_last_appended_in_memory % self.mem_save_every == 0:
            self.memory = np.roll(self.memory, self.n_observations_base)
            self.memory[: self.n_observations_base] = base_state
            self.steps_since_last_appended_in_memory = 0

        return state

    def compute_reward(self, action):
        return self.sim.reward()

    def reset_memory(self):
        self.memory = np.full(self.n_observations_base * self.mem_n_past_states,
                              self.mem_default_value, dtype=np.float32)
        self.steps_since_last_appended_in_memory = 0

    def reset(self):
        # reset sim
        self.sim.reset()

        # reset memory
        self.reset_memory()

        # get initial state
        s0 = self.get_state(append_to_memory=True)
        self.steps_since_last_appended_in_memory = 0

        return s0
