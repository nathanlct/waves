import numpy as np
from stable_baselines3 import PPO

from env import WaveEnv

# load trained model
model = PPO.load("/Users/hayat/waves/logs/1647614451_18Mar22_15h40m51s/model")

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
