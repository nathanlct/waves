import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from stable_baselines3 import PPO
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import json

from waves.env import WavesEnv
from waves.utils import parse_env_args

# Define the function f(x,y)


PATH = 'logs/train/1680196256_30Mar23_10h10m56s/checkpoints/model_500000_steps.zip'
# PATH = 'logs/train/1680192742_30Mar23_09h12m22s/checkpoints/model_1500000_steps.zip'


# load trained model and config file
model_path = Path(PATH)
model = PPO.load(str(model_path))
with open(str(model_path.parent.parent / "configs.json"), "r") as f:
    configs = json.load(f)
env_kwargs = parse_env_args(configs["args"])


def normalize(x, xmin, xmax):
    """Transform x from [xmin, xmax] range to [-1, 1] range.
    """
    return (x - xmin) / (xmax - xmin) * 2.0 - 1.0


def f(F, MMS):
    K = 50578.0

    state = np.array([
        normalize(MMS, 0, 50 * K),
        normalize(F, 0, 2 * K),
        normalize(min(F, 5000), 0, 5000),
        normalize(min(F, 500), 0, 500),
        normalize(min(F, 50), 0, 50),
        normalize(min(F, 5), 0, 5),
    ])

    action, _ = model.predict(state, deterministic=True)

    action_min = 0
    action_max = 10 * K
    action = action / 99 * \
        (action_max - action_min) + action_min
    # action = np.array([action])

    return action


# Create a mesh grid
x = np.linspace(0, 10000, 300)
y = np.linspace(0, 10000000, 300)
X, Y = np.meshgrid(x, y)

# Evaluate the function over the mesh grid
Z = np.zeros((len(x), len(y)))  # initialize the output array

for i in range(len(x)):
    print(i)
    for j in range(len(y)):
        Z[i, j] = f(x[i], y[j])


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

# Set the labels for the axes
ax.set_xlabel('F')
ax.set_ylabel('M+MS')
ax.set_zlabel('control')

# Show the plot
plt.show()
