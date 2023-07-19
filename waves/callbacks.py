import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import matplotlib
matplotlib.use('agg')


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, eval_env, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

        self.eval_env = eval_env

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        pass

    def _on_rollout_end(self):
        actions = []
        rewards = []
        states = []
        env_times = [0]  # store times at a possibly lower frequency than sim times
        reward_infos = defaultdict(lambda: [0])

        # do a rollout on test env using trained policy
        done = False
        state = self.eval_env.reset()
        states.append(state)
        while not done:
            action = self.model.predict(state, deterministic=True)[0]
            state, reward, done, infos = self.eval_env.step(action)
            states.append(state)
            actions.append(infos['scaled_action'])
            rewards.append(reward)
            env_times.append(self.eval_env.sim.t)
            for k, v in infos['reward_info'].items():
                reward_infos[k].append(v)
        actions.append(actions[-1])
        rewards.append(rewards[-1])

        actions = np.array(actions)  # (n_times, n_actions)
        rewards = np.array(rewards)  # (n_times,)
        t_lst = np.array(self.eval_env.sim.t_lst)  # (n_times,)
        x_lst = np.array(self.eval_env.sim.x)  # (n_times,)
        y_lst = np.array(self.eval_env.sim.y_lst)  # (n_times, n_samples)
        states = np.array(states)  # (n_times, n_states)
        n_times, n_samples = y_lst.shape
        n_actions = actions.shape[1]
        n_states = states.shape[1]
        
        # LOG SCALARS
        self.logger.record('eval/norm_y0', np.linalg.norm(y_lst[0]))
        self.logger.record('eval/norm_yf', np.linalg.norm(y_lst[-1]))
        self.logger.record('eval/cum_reward', np.sum(rewards))

        # LOG PLOTS
        def plot(name, x, y_dict, xlabel, ylabel, title, grid=True, legend=True, figsize=(5, 3)):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            for k, v in y_dict.items():
                ax.plot(x, v, label=k)
            if legend:
                ax.legend()
            if grid:
                ax.grid()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            plt.tight_layout()
            self.logger.record(f'eval/{name}', Figure(fig, close=True),
                               exclude=('stdout', 'log', 'json', 'csv'))
            plt.close(fig)

        # y0 and yf by x
        plot('y0_yf_by_x', x_lst, {'y0': y_lst[0], 'yf': y_lst[-1]},
             xlabel='x', ylabel='y(x)', title='Initial and final states')

        # yi, actions, rewards, norms by t
        plot('y_by_t', t_lst, {f'y{i+1}': y_lst[:,i] for i in range(min(10, n_samples))},
             xlabel='t', ylabel='y(t)', title='States by time')
        plot('y123_by_t', t_lst, {f'y{i+1}': y_lst[:,i] for i in range(3)},
             xlabel='t', ylabel='y(t)', title='Small states by time')
        plot('obs_by_t', env_times, {f'x{i+1}': states[:,i] for i in range(n_states)},
             xlabel='t', ylabel='NN input', title='Normalized & centered observations by time')
        plot('u_by_t', env_times, {f'u{i+1}': actions[:,i] for i in range(min(10, n_actions))},
             xlabel='t', ylabel='u(t)', title='Actions by time')
        plot('r_by_t', env_times, {'r': rewards},
             xlabel='t', ylabel='r(t)', title='Reward by time', legend=False)
        plot('norm_by_t', t_lst, {f'np': np.linalg.norm(y_lst, axis=1)},
             xlabel='t', ylabel='norm(y(t))', title='Norm of states by time', legend=False)
        plot('rewards_by_t', env_times, reward_infos,
             xlabel='t', ylabel='r_i(t)', title='Breakdown of reward components by time')
