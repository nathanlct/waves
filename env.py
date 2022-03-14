import gym
from gym import spaces
from simulation import Simulation
import numpy as np


class WaveEnv(gym.Env):
    def __init__(self):
        super(WaveEnv, self).__init__()

        self.state_buffer_size = 200
        self.state_append_every = 10
        self.state_append_counter = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1 + self.state_buffer_size,), dtype=np.float32)

        self.sim = Simulation(
            f=lambda x: x*x + x,
            dt=1e-3,
            dx=1e-2,
            xmin=0.5,
            xmax=1.5,
        )

        self.reset()

    def step(self, action):
        # log infos
        infos = {}
        if self.sim.t == 0:
            infos['init_norm_y'] = self.sim.norm_y()

        # sim step
        self.sim.step(u=action)

        # update state
        self.state_append_counter += 1
        if self.state_append_counter == self.state_append_every:
            self.state_append_counter = 0
            # roll previous states buffer
            self.state[2:] = self.state[1:-1]
        self.state[:2] = self.sim.get_obs()
        
        # compute reward
        reward = max(-10, -self.sim.norm_y())

        # compute done
        done = self.sim.t >= 2.0

        # log infos
        if done:
            infos['done_norm_y'] = self.sim.norm_y()

        return self.state, reward, done, infos

    def reset(self):
        self.sim.reset()

        self.state = np.zeros(1 + self.state_buffer_size)
        self.state[:2] = self.sim.get_obs()
        self.state_append_counter = 1 if self.state_append_every > 1 else 0

        return self.state


if __name__ == '__main__':
    env = WaveEnv()

    state = env.reset()
    while True:
        state, reward, done, _= env.step(-1)
        print(reward)
        if done:
            break
