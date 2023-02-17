"""train.py"""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from waves.callbacks import TensorboardCallback
from waves.env import WavesEnv
from waves.utils import create_log_dir, parse_sim_args


# parse CLI params
parser = argparse.ArgumentParser()
parser.add_argument('sim', type=str, help='Simulation to use, eg. "SimStabilizeMidObs".')

# sim params (should have default values defined in models/ so should be able to be left blank)
parser.add_argument('--kwargs', type=str, default='dict()', help='Additional simulation kwargs, eg. "dict(f=lambda x: x*x)".')
parser.add_argument('--y0', type=str, default=None, help='Initial condition as a (ideally vectorized) function of space, '
                    'eg. "lambda x: np.cos(x*np.pi)". By default, uses the simulation\'s default, if there is one.')
parser.add_argument('--dx', type=float, default=None, help='Space sampling interval. By default, use the simulation\'s default.')
parser.add_argument('--xmin', type=float, default=None, help='Space left boundary. By default, use the simulation\'s default.')
parser.add_argument('--xmax', type=float, default=None, help='Space right boundary. By default, use the simulation\'s default.')
parser.add_argument('--dt', type=float, default=None, help='Time sampling interval. By default, use the simulation\'s default.')

# env params
parser.add_argument('--tmax', type=float, default=2.0, help='Duration (in time) of one episode.')
parser.add_argument('--action_min', type=str, default='-1.0', help='Minimum control value. Could use a simulation-specific '
                        'constant, eg. "self.sim.k * 10".')
parser.add_argument('--action_max', type=str, default='1.0', help='Maximum control value. Could use a simulation-specific '
                        'constant, eg. "self.sim.k * 10".')
parser.add_argument('--n_past_states', type=int, default=0, help='Number of previous states to add in the current state (memory).')

# training params
parser.add_argument('--cpus', type=int, default=1, help='Number of CPUs to use for training.')
parser.add_argument('--steps', type=float, default=1e9, help='Number of timesteps to train for.')

args = parser.parse_args()


# create config
sim_class, sim_kwargs = parse_sim_args(args)

env_kwargs = {
    'sim_class': sim_class,
    'sim_kwargs': sim_kwargs,
    'config': {
        'tmax': args.tmax,
        'action_min': args.action_min,
        'action_max': args.action_max,
        'n_past_states': args.n_past_states,
    },
}

# create experiment dir
exp_dir = create_log_dir(subfolder='train')
print(f'> {exp_dir}')

# create env
vec_env = make_vec_env(WavesEnv, n_envs=args.cpus, env_kwargs=env_kwargs)
eval_env = WavesEnv(**env_kwargs)

# create model
model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=exp_dir / 'tb')

# train model
model.learn(total_timesteps=int(args.steps), callback=TensorboardCallback(eval_env))

# save model
model_path = exp_dir / 'model'
model.save(model_path)
print(f'> {model_path}')
