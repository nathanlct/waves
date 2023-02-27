import numpy as np
from stable_baselines3 import PPO
import argparse
from pathlib import Path
import json

from waves.env import WavesEnv
from waves.utils import parse_env_args

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("cp_path", type=str, help="Path to .zip checkpoint to load and use "
                    'as a control. There must be a configs.json in the parent directory.')
args = parser.parse_args()

# load trained model and config file
model_path = Path(args.cp_path)
model = PPO.load(str(model_path))
with open(str(model_path.parent.parent / 'configs.json'), 'r') as f:
    configs = json.load(f)
env_kwargs = parse_env_args(configs['args'])

# create env
env = WavesEnv(**env_kwargs)

# eval loop
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

# render
env.sim.render(path=None, display=True, fps=30.0, dpi=100, speed=1.0, no_video=True)
