from datetime import datetime
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env import SimControlHeatEnv, SimStabilizeMidObsEnv, DEFAULT_ENV_CONFIG
from callbacks import TensorboardCallback


# create experiment dir
now = datetime.now().strftime("%d%b%y_%Hh%Mm%Ss")
timestamp = datetime.now().timestamp()
exp_dir = Path(f"logs/{int(timestamp)}_{now}/")
exp_dir.mkdir(parents=True, exist_ok=False)
print(f"Created exp dir at {exp_dir}")

# training config
config = DEFAULT_ENV_CONFIG
config.update(dict(
    dt=0.45 * 1e-4,
    dx=1e-2,
    xmin=0,
    xmax=1,
    y0=lambda x: 3.6 * np.sin(np.pi * x),
))

# create env
n_cpus = 4
vec_env = make_vec_env(SimControlHeatEnv, n_envs=n_cpus, env_kwargs=dict(config=config))
eval_env = SimControlHeatEnv(config=config)

# create model
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=exp_dir / "tb")

# train model
model.learn(total_timesteps=1000000, callback=TensorboardCallback(eval_env))

# save model
model.save(exp_dir / "model")
print(f"Done, experiment data saved at {exp_dir}")
