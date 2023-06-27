import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from pathlib import Path
import json

from waves.env import WavesEnv
from waves.utils import parse_env_args



# PATH = 'logs/train/1681967450_19Apr23_22h10m50s/checkpoints/model_5500000_steps.zip'
# PATH = 'logs/train/1683042542_02May23_08h49m02s/checkpoints/model_500000_steps.zip'
PATH = 'logs/train/1683042542_02May23_08h49m02s/checkpoints/model_4000000_steps.zip'
LOG_SCALE = True
USE_EXPLICIT_ACTION = True
ACTION_NOISE_STD = 0  # 10

# load trained model and config file
model_path = Path(PATH)
model = PPO.load(str(model_path))
with open(str(model_path.parent.parent / "configs.json"), "r") as f:
    configs = json.load(f)
env_kwargs = parse_env_args(configs["args"])


def get_action(MMS, F):
    action = 0
    if F == 0 or np.log(MMS) > np.log(F) + 4:
        action = 10 #  min(3, 3*F)
    # elif np.log(MMS) > np.log(F) + 3:
    #     action = 500000 * (4 - np.log(MMS / F))
    else:
        action = 200000
    return action


def f(MMS, F, env):
    state = env.sim.get_obs(MMS=MMS, F=F)
    action, _ = model.predict(state, deterministic=True)
    action = env.normalize_action(action)

    if USE_EXPLICIT_ACTION:
        action = get_action(MMS, F)

    return action


# create env
env = WavesEnv(**env_kwargs)
env.tmax = 1000
# env.n_steps_per_action = 1

# Create a mesh grid
K = 50578.0
if LOG_SCALE:
    x_mms = np.concatenate(([0], np.geomspace(1, 120 * K, 100)))
    y_f = np.concatenate(([0], np.geomspace(1, 120 * K, 100)))
else:
    x_mms = np.linspace(0, 120 * K, 100)
    y_f = np.linspace(0, 120 * K, 100)
X, Y = np.meshgrid(x_mms, y_f)

# Evaluate the function over the mesh grid
Z = np.zeros((len(x_mms), len(y_f)))  # initialize the output array
for i in range(len(x_mms)):
    for j in range(len(y_f)):
        Z[j, i] = f(x_mms[i], y_f[j], env)

# Create a 2D heatmap
fig, ax = plt.subplots()
# fig = plt.figure()

# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z)

# plt.show()
# import sys ; sys.exit()
pcm = ax.pcolormesh(X, Y, Z)

for _ in range(1):
    # run sim
    done = False
    state = env.reset()
    actions = []
    noises = []
    while not done:
        action, _ = model.predict(state, deterministic=True)
        if USE_EXPLICIT_ACTION:
            all_females = env.sim.y[2] * (1 + env.sim.gammas * env.sim.y[3] / env.sim.y[1])
            mms = env.sim.y[1] + env.sim.y[3]
            action = np.array([get_action(mms, all_females)], dtype=np.float64)

        if ACTION_NOISE_STD > 0:
            noise = np.random.normal(loc=0.0, scale=ACTION_NOISE_STD)
            noises.append(noise)
            action += noise

        state, reward, done, info = env.step(action, normalize_action=not USE_EXPLICIT_ACTION)
        actions.append(info['scaled_action'])
        # if env.sim.t > 200:
        #     done = True
    actions.append(actions[-1])

    actions = np.array(actions)[:, 0]  # (n_times, n_actions)
    t_lst = np.array(env.sim.t_lst)  # (n_times,)
    y_lst = np.array(env.sim.y_lst)  # (n_times, n_samples)

    gammas = 1
    mms = y_lst[:, 1] + y_lst[:, 3]
    all_females = y_lst[:, 2] * (1 + gammas * y_lst[:, 3] / y_lst[:, 1])

    n_steps_per_action = env_kwargs['config']['n_steps_per_action']
    # mms = mms[::n_steps_per_action]
    # all_females = all_females[::n_steps_per_action]

    ax.plot(mms, all_females, 'r', linewidth=1)
    ax.scatter(mms[0], all_females[0], s=30, marker='o', color='r', zorder=100)
    ax.scatter(mms[-1], all_females[-1], s=50, marker='x', color='r', zorder=100)
    ax.scatter(0.1, 0.1, s=0)  # to always show the full plot

    print("Initial state:", y_lst[0])
    print("Final state:", y_lst[-1])
    print("Max state:", np.max(y_lst, axis=0))
    print()

# Set the labels for the axes
ax.set_xlabel('Total males' + ' (log scale)' if LOG_SCALE else '')
ax.set_ylabel('Total females' + ' (log scale)' if LOG_SCALE else '')
ax.set_title('u(M+MS, F*(M+gamma_s*M_s)/M)')

if LOG_SCALE:
    ax.set_xscale('log')
    ax.set_yscale('log')

# Add a colorbar
cbar = fig.colorbar(pcm)
cbar.ax.set_title('Action')

plt.show()


# plt.figure()
# plt.plot(actions)
# plt.show()

# plt.figure()
# plt.plot(noises)
# plt.show()
