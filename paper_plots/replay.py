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
import pprint


PATH = 'logs/train/1681964053_19Apr23_21h14m13s/checkpoints/model_3000000_steps.zip'


# load trained model and config file
model_path = Path(PATH)
model = PPO.load(str(model_path))
with open(str(model_path.parent.parent / "configs.json"), "r") as f:
    configs = json.load(f)
env_kwargs = parse_env_args(configs["args"])
pprint.pprint(env_kwargs)

# create fig
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot 3D surface
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

ax.plot_surface(X, Y, Z, alpha=0.7)

for _ in range(1):
    # create env
    env = WavesEnv(**env_kwargs)

    # run sim
    done = False
    state = env.reset()
    actions = []
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        actions.append(info['scaled_action'])
    actions.append(actions[-1])

    actions = np.array(actions)[:, 0]  # (n_times, n_actions)
    t_lst = np.array(env.sim.t_lst)  # (n_times,)
    y_lst = np.array(env.sim.y_lst)  # (n_times, n_samples)

    gammas = 1
    mms = y_lst[:, 1] + y_lst[:, 3]
    all_females = y_lst[:, 2] * (1 + gammas * y_lst[:, 3] / y_lst[:, 1])

    n_steps_per_action = env_kwargs['config']['n_steps_per_action']
    mms = mms[::n_steps_per_action]
    all_females = all_females[::n_steps_per_action]

    # actions = np.repeat(actions, n_steps_per_action)
    # actions = np.append(actions, actions[-1])

    # 3D plot
    ax.plot(mms, all_females, actions)
    ax.scatter(mms[0], all_females[0], actions[0], s=10, marker='o', label='start')

ax.invert_xaxis()
ax.set_xlabel('M+MS')
ax.set_ylabel('F*(M+gamma_s*M_s)/M')
ax.set_zlabel('u')
plt.show()
