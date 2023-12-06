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


# PATH = 'logs/train/1681967450_19Apr23_22h10m50s/checkpoints/model_5500000_steps.zip'
PATH = 'logs/train/1681964053_19Apr23_21h14m13s/checkpoints/model_3000000_steps.zip'


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

def f(MMS, F):
    K = 50578.0

    state = np.array([
        normalize(MMS, 0, 100 * K),
        normalize(min(MMS, 50 * K), 0, 50 * K),
        normalize(min(MMS, 10 * K), 0, 10 * K),
        normalize(min(MMS, K), 0, K),
        normalize(min(MMS, 5000), 0, 5000),
        normalize(min(MMS, 500), 0, 500),
        normalize(min(MMS, 50), 0, 50),
        normalize(min(MMS, 5), 0, 5),
        normalize(F, 0, 100 * K),
        normalize(min(F, 50 * K), 0, 50 * K),
        normalize(min(F, 10 * K), 0, 10 * K),
        normalize(min(F, K), 0, K),
        normalize(min(F, 5000), 0, 5000),
        normalize(min(F, 500), 0, 500),
        normalize(min(F, 50), 0, 50),
        normalize(min(F, 5), 0, 5),
    ])

    action, _ = model.predict(state, deterministic=True)

    action_min = 0
    action_max = 10 * K
    action = action / 200 * \
        (action_max - action_min) + action_min

    return action

# Create a mesh grid
K = 50578.0
x_mms = np.linspace(0, 120 * K, 100)
y_f = np.linspace(0, 120 * K, 100)
X, Y = np.meshgrid(x_mms, y_f)

# Evaluate the function over the mesh grid
Z = np.zeros((len(x_mms), len(y_f)))  # initialize the output array
for i in range(len(x_mms)):
    for j in range(len(y_f)):
        Z[j, i] = f(x_mms[i], y_f[j])


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

ax.invert_xaxis()

# Set the labels for the axes
ax.set_xlabel('M+MS')
ax.set_ylabel('F*(M+gamma_s*M_s)/M')
ax.set_zlabel('u')

# Show the plot
plt.show()
