import gym
from gym import spaces
from simulation import Simulation, Simulation_2
import numpy as np


class WaveEnv(gym.Env):
    def __init__(self):
        super(WaveEnv, self).__init__()

        self.state_buffer_size = 1
        self.state_append_every = 1
        self.state_append_counter = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1 + self.state_buffer_size,),
            dtype=np.float32,
        )

        self.sim = Simulation_2(
            f=lambda x: x * (1 / 0.98) + 0 * (x * x),
            dt=0.98 * 1e-2,
            dx=1e-2,
            xmin=0.5,
            xmax=1.5,
        )

        self.reset()

    def step(self, action):
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
        done = self.sim.t >= 15.0

        return self.state, reward, done, {}

    def reset(self, y0=None):
        if y0 is None:
            # generate an initial condition y0 = sum_n a_n sin(nx) + bn cos(nx)
            # with sum_n (a_n + b_n) < 0.2
            n = 30
            a_lst = np.random.random(n)
            a_lst_2 = np.random.random(n)
            b_lst = np.random.random(n)
            b_lst_2 = np.random.random(n)
            norm_coef = 5.0 / (np.sum(a_lst) + np.sum(b_lst))
            norm_coef_2 = 5.0 / (np.sum(a_lst_2) + np.sum(b_lst_2))
            a_lst *= norm_coef
            b_lst *= norm_coef
            a_lst_2 *= norm_coef_2
            b_lst_2 *= norm_coef_2
            n_lst = np.arange(n)
            n_lst_2 = np.arange(n)
            self.y0 = [
                lambda x: np.sum(a_lst * np.sin(n_lst * x) + b_lst * np.cos(n_lst * x)),
                lambda x: np.sum(
                    a_lst_2 * np.sin(n_lst_2 * x) + b_lst_2 * np.cos(n_lst_2 * x)
                ),
            ]
        else:
            self.y0 = y0

        self.sim.reset(y10=self.y0[0], y20=self.y0[1])

        self.state = np.zeros(1 + self.state_buffer_size)
        self.state[:2] = self.sim.get_obs()
        self.state_append_counter = 1 if self.state_append_every > 1 else 0

        return self.state


if __name__ == "__main__":
    env = WaveEnv()

    state = env.reset()
    while True:
        state, reward, done, _ = env.step(-1)
        print(reward)
        if done:
            break
