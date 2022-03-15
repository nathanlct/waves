import numpy as np
from stable_baselines3 import PPO

from env import WaveEnv

# load trained model
model = PPO.load("/Users/hayat/waves/logs/1647360415_15Mar22_17h06m55s/model")

# create env
env = WaveEnv()

# evaluation loop
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        break
env.sim.render(path="./hardcoded.gif")
