import numpy as np
import gym

from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class WavesEnv(gym.Env):
    def __init__(self, sim_class, sim_kwargs, env_config, n_envs=1):
        # vectorized environment
        self.n_envs = n_envs

        # load config
        self.env_config = env_config
        self.tmax = env_config["tmax"]
        self.n_steps_per_action = env_config["n_steps_per_action"]

        # create sim
        self.sim = sim_class(**sim_kwargs, tmax=self.tmax, n_sims=self.n_envs)

        # memory
        self.sim.reset()
        self.n_observations_base = self.sim.get_obs().shape[1]
        self.n_past_states = env_config["n_past_states"]
        self.reset_memory()

        # observation space
        self.n_observations = self.n_observations_base * (1 + self.n_past_states)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_observations,), dtype=np.float32,
        )

        # action space
        self.n_actions = self.sim.n_controls
        self.action_min = eval(env_config["action_min"])
        self.action_max = eval(env_config["action_max"])
        # self.action_space = gym.spaces.Box(
        #     low=-5.0, high=5.0, shape=(self.n_actions,), dtype=np.float32
        # )
        self.action_space = gym.spaces.Discrete(40)

        # reset env
        self.reset()

    def step(self, actions):
        # convert actions from [-1, 1] to [action_min, action_max]
        # amin = -5.0
        # amax = 5.0
        # actions = (actions - amin) * (
        #     self.action_max - self.action_min
        # ) / (amax - amin) + self.action_min
        actions = np.array(actions) * 10000

        # step simulation with control
        for _ in range(self.n_steps_per_action):
            self.sim.step(u=actions)

        # update state
        states = self.sim.get_obs()
        # states = self.get_state(append_to_memory=True)

        # compute reward
        # reward, reward_info = self.sim.reward()
        # reward = np.array([reward] * self.n_envs)  # TODO
        rewards, rewards_info = self.sim.reward()

        infos = [{
            'scaled_action': actions[i],
            'reward_info': {
                k: v[i] for k, v in rewards_info.items()    
            },
        } for i in range(self.n_envs)]  # TODO

        # compute done
        # done = False
        # TODO all sims are done at the same time for now, and Im bounding the norm of y instead of stopping early it it blows up
        dones = np.full((self.n_envs,), self.sim.t >= self.tmax - 1e-5)  # TODO
        # if self.sim.t >= self.tmax:
        #     done[:] = True

        # end early if norm blows up
        # if self.sim.norm_y() > 1000 * self.sim.K:
        #     done[:] = True
        #     reward[:] -= 100
        #     # info["TimeLimit.truncated"] = True  # for SB3 end-of-horizon bootstrapping

        return states, rewards, dones, infos

    # def get_base_state(self):
    #     # TODO not implemented for vectorized sim
    #     return self.sim.get_obs()

    # def get_state(self, append_to_memory=False):
    #     # TODO not implemented for vectorized sim
    #     base_state = self.get_base_state()
        
    #     if self.n_past_states > 0:
    #         # TODO
    #         raise NotImplementedError("Memory not implemented with vectorized environment")
    #         state = np.concatenate((base_state, self.memory))

    #         if append_to_memory and self.n_past_states > 0:
    #             self.memory = np.roll(self.memory, self.n_observations_base)
    #             self.memory[: self.n_observations_base] = base_state

    #     return np.array([base_state for _ in range(self.n_envs)])  # TODO

    def reset_memory(self):
        self.memory = np.zeros(self.n_observations_base * self.n_past_states)

    def reset(self):
        # reset sim
        self.sim.reset()

        # reset memory
        self.reset_memory()

        # get initial state
        # s0 = self.get_state(append_to_memory=True)
        s0 = self.sim.get_obs()

        return s0
    
    def close(self):
        pass
    
    def env_is_wrapped(self):
        pass
    
    def env_method(self):
        pass
    
    def get_attr(self):
        pass
    
    def seed(self):
        pass
    
    def set_attr(self):
        pass
    
    def step_async(self):
        pass
    
    def step_wait(self):
        pass
