import numpy as np
from stable_baselines3 import PPO
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import json

from waves.env import WavesEnv
from waves.utils import parse_env_args

# parse args
parser = argparse.ArgumentParser()
parser.add_argument(
    "cp_path",
    type=str,
    help="Path to .zip checkpoint to load and use "
    "as a control. There must be a configs.json in the parent directory.",
)
parser.add_argument(
    "--plot",
    type=bool,
    help="If not False, we plot the control to visualize it rather than using it in an evaluation",
    default=False,
)

args = parser.parse_args()


# load trained model and config file
model_path = Path(args.cp_path)
model = PPO.load(str(model_path))
with open(str(model_path.parent.parent / "configs.json"), "r") as f:
    configs = json.load(f)
env_kwargs = parse_env_args(configs["args"])

env_kwargs["sim_kwargs"].update(
    {
        "K": 50000.0,
        "y0": lambda x: np.random.uniform(low=0.0, high=5 * 50000, size=4),
        "dt": 1e-2,
    }
)
env_kwargs["config"]["tmax"] = 1000.0
print(env_kwargs)

n_steps_per_action = env_kwargs["config"]["n_steps_per_action"]

# create env
env = WavesEnv(**env_kwargs)

# eval loop

all_ys = []
all_us = []

n_sims = 10
tmax = 820
for i in range(n_sims):
    print(f"{i+1}/{n_sims}")

    obs = env.reset()
    done = False
    u_lst = []
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        action = env.normalize_action(action)
        action = np.clip(action, 1000, 500000)  # + np.random.normal()
        obs, reward, done, info = env.step(action, normalize_action=False)
        u_lst += [float(action)] * n_steps_per_action

    ys = np.array(env.sim.y_lst)  # (n_times, 4): E, M, F, Ms
    ts = np.array(env.sim.t_lst)  # (n_times,)
    us = np.array([0] + u_lst)  # (n_times,)

    all_ys.append(np.copy(ys))
    all_us.append(np.copy(us))

all_ys = np.array(all_ys)  # shape (n_sims, n_times, 4) - E, M, F, Ms
all_us = np.array(all_us)  # shape (n_sims, n_times)

for t in [200, 400, 600, 800]:  # 800, 2000]:
    print(f"t = {t}")
    # find idx corresponding to that time
    idx = 0
    while ts[idx] < t:
        idx += 1
    # compute metrics
    EMF = (
        np.abs(all_ys[:, idx, 0])
        + np.abs(all_ys[:, idx, 1])
        + np.abs(all_ys[:, idx, 2])
    )
    print(f"\tavg |E|+|M|+|F| = {np.mean(EMF)}")
    print(f"\tvar |E|+|M|+|F| = {np.var(EMF)}")
    print(f"\tmax |E|+|M|+|F| = {np.max(EMF)}")
    MS = np.abs(all_ys[:, idx, 3])
    print(f"\tavg |Ms| = {np.mean(MS)}")
    print(f"\tvar |Ms| = {np.var(MS)}")
    print(f"\tmax |Ms| = {np.max(MS)}")

fig, axes = plt.subplots(nrows=5, figsize=(6, 8), sharex=True, dpi=300)
for k in range(5):
    ax = axes[k]
    # for step in [t * 100 for t in range(0, tmax + 1, 100)]:
    #     ax.scatter([step * sim.dt] * n_sims, all_ys[:, step, k], c='k', s=10)
    if k < 4:
        for j in range(n_sims):
            ax.plot(ts, all_ys[j, :, k], linewidth=1.0)  # 'k'
    else:
        for j in range(n_sims):
            ax.plot(ts, all_us[j,], linewidth=1.0)  # 'k'
        ax.set_xlabel("Time (days)")
    ax.set_ylabel(["E(t)", "M(t)", "F(t)", "M$_s$(t)", "u(t)"][k])
    ax.grid()
plt.tight_layout()
plt.savefig("distributions_control_with_mem.png")  # hardcoded without_mem as well

# plot 2
fig, axes = plt.subplots(nrows=3, figsize=(6, 4.8), sharex=True, dpi=300)
ax = axes[0]
for j in range(n_sims):
    ax.plot(ts, np.sum(all_ys[j, :, :3], axis=1), linewidth=1.0)  # 'k'
ax.set_ylabel("E(t)+M(t)+F(t)")
ax.grid()

ax = axes[1]
for j in range(n_sims):
    ax.plot(ts, all_ys[j, :, 3], linewidth=1.0)  # 'k'
ax.set_ylabel("M$_s$(t)")
ax.grid()

ax = axes[2]
for j in range(n_sims):
    ax.plot(ts, all_us[j,], linewidth=1.0)  # 'k'
ax.set_ylabel("u(t)")
ax.grid()
ax.set_xlabel("Time (days)")

plt.tight_layout()
plt.savefig("distributionsv2_control_with_mem.png")
