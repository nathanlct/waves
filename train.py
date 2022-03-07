from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env import WaveEnv
from callbacks import TensorboardCallback


# create experiment dir
now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')
timestamp = datetime.now().timestamp()
exp_dir = Path(f'logs/{int(timestamp)}_{now}/')
exp_dir.mkdir(parents=True, exist_ok=False)
print(f'Created exp dir at {exp_dir}')

# create env
n_cpus = 4
vec_env = make_vec_env(WaveEnv, n_envs=n_cpus)

# create model
model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=exp_dir / 'tb')

# train model
model.learn(total_timesteps=500000, callback=TensorboardCallback())

# save model
model.save(exp_dir / 'model')