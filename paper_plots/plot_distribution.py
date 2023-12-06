from models.sim_ode_discrete import SimODEDiscrete

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


sim = SimODEDiscrete(
    K=50000.0,
    y0=lambda x: np.random.uniform(low=0.0, high=10*50000, size=4),
    dt=1e-2,
)


def get_action(total_males, total_females):
    if total_females == 0 or (total_males != 0 and np.log(total_males / total_females) > 4):
        action = 5
    else:
        action = 300000  # limite 165435
    return action

# def get_action(MMS, F, noise=False):
#    action = 0
#    if MMS < 200:
#        if F == 0 or np.log(F) < np.log(200) - 4:
#            action = 5
#        elif np.log(F) < np.log(200) - 3:
#            action = 300000 * (4 + np.log(F / 200))
#        else:
#            action = 300000
#    else:
#        if F == 0 or np.log(MMS) > np.log(F) + 4:
#            action = 5
#        elif np.log(MMS) > np.log(F) + 3:
#            action = 300000 * (4 - np.log(MMS / F))
#        else:
#            action = 300000
#    if noise:
#        action += np.random.normal(loc=0.0, scale=10.0)
#    return action


all_ys = []
all_us = []

n_sims = 100
tmax = 820
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
    all_us.append(np.copy(us))

all_ys = np.array(all_ys)  # shape (n_sims, n_times, 4) - E, M, F, Ms
all_us = np.array(all_us)  # shape (n_sims, n_times)

for t in [200, 400, 600, 800]:  # 800, 2000]:
    print(f't = {t}')
    # find idx corresponding to that time
    idx = 0
    while ts[idx] < t:
        idx += 1
    # compute metrics
    EMF = np.abs(all_ys[:, idx, 0]) + np.abs(all_ys[:, idx, 1]) + np.abs(all_ys[:, idx, 2])
    print(f'\tavg |E|+|M|+|F| = {np.mean(EMF)}')
    print(f'\tvar |E|+|M|+|F| = {np.var(EMF)}')
    print(f'\tmax |E|+|M|+|F| = {np.max(EMF)}')
    MS = np.abs(all_ys[:, idx, 3])
    print(f'\tavg |Ms| = {np.mean(MS)}')
    print(f'\tvar |Ms| = {np.var(MS)}')
    print(f'\tmax |Ms| = {np.max(MS)}')

fig, axes = plt.subplots(nrows=5, figsize=(6, 8), sharex=True, dpi=300)
for k in range(5):
    ax = axes[k]
    # for step in [t * 100 for t in range(0, tmax + 1, 100)]:
    #     ax.scatter([step * sim.dt] * n_sims, all_ys[:, step, k], c='k', s=10)
    if k < 4:
        for j in range(n_sims):
            ax.plot(ts, all_ys[j, :, k],  linewidth=1.0)  # 'k'
    else:
        for j in range(n_sims):
            ax.plot(ts, all_us[j, ],  linewidth=1.0)  # 'k'
        ax.set_xlabel('Time (days)')
    ax.set_ylabel(['E(t)', 'M(t)', 'F(t)', 'M$_s$(t)', 'u(t)'][k])
    ax.grid()
plt.tight_layout()
plt.savefig('distributions.png')
