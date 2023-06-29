from models.sim_ode_discrete import SimODEDiscrete

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


sim = SimODEDiscrete(
    K=50578.0,
    y0=lambda x: np.random.uniform(low=0.0, high=510000, size=4),
    dt=1e-2,
)


def get_action(total_males, total_females):
    if total_females == 0 or (total_males != 0 and np.log(total_males / total_females) > 4):
        action = 5
    else:
        action = 300000  # limite 165435
    return action

all_ys = []

n_sims = 30
tmax = 500
for i in range(n_sims):
    print(f'{i+1}/{n_sims}')

    sim.reset()

    u_lst = []
    while sim.t <= tmax:
        total_males = sim.y[1] + sim.y[3]
        total_females = sim.y[2] * (1 + sim.gammas * sim.y[3] / sim.y[1])
        action = get_action(total_males, total_females)
        u_lst.append(action)
        sim.step(u=[action])

    ys = np.array(sim.y_lst)  # (n_times, 4): E, M, F, Ms
    ts = np.array(sim.t_lst)  # (n_times,)
    us = np.array([0] + u_lst)  # (n_times,)

    all_ys.append(np.copy(ys))

all_ys = np.array(all_ys)

fig, axes = plt.subplots(nrows=4, figsize=(12, 8))
for k in range(4):
    ax = axes[k]
    for step in [t * 100 for t in range(0, tmax + 1, 100)]:
        ax.scatter([step * sim.dt] * n_sims, all_ys[:, step, k], c='k', s=10)
    for j in range(n_sims):
        ax.plot(ts, all_ys[j, :, k], 'k', linewidth=0.5)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(['E(t)', 'M(t)', 'F(t)', 'Ms(t)'][k])
plt.tight_layout()
plt.show()