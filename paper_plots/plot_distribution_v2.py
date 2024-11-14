import itertools
import sys
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")

from models.sim_ode_discrete import SimODEDiscrete


def get_action(total_males, total_females):
    if total_females == 0 or (total_males != 0 and np.log(total_males / total_females) > 4):
        action = 5
    else:
        action = 300000
    return action


all_ys = []
all_us = []

n_points_ic = 5
n_points_params = 3
tmax = 820
K = 50000.0

# 4 random ICs
# 6 random model params
N_total = n_points_ic**4 * n_points_params**6
print("total number of sweeps: ", N_total)


def run_simulation(params):
    y0_1, y0_2, y0_3, y0_4, betaE, nuE, deltaE, deltaF, deltaM, nu = params
    K = 50000.0
    tmax = 820

    sim = SimODEDiscrete(
        K=K,
        y0=lambda x: [y0_1, y0_2, y0_3, y0_4],
        dt=1e-2,
    )
    sim.betaE = betaE
    sim.nuE = nuE
    sim.deltaE = deltaE
    sim.deltaF = deltaF
    sim.deltaM = deltaM
    sim.nu = nu
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

    return ys, us


total_time = 0
N = 0
for y0_1 in np.linspace(K, 20 * K, n_points_ic):
    for y0_2 in np.linspace(K, 20 * K, n_points_ic):
        for y0_3 in np.linspace(K, 20 * K, n_points_ic):
            for y0_4 in np.linspace(K, 20 * K, n_points_ic):
                for betaE in np.linspace(7.46, 14.85, n_points_params):
                    for nuE in np.linspace(0.005, 0.25, n_points_params):
                        for deltaE in np.linspace(0.023, 0.046, n_points_params):
                            for deltaF in np.linspace(0.033, 0.046, n_points_params):
                                for deltaM in np.linspace(0.077, 0.139, n_points_params):
                                    for nu in np.linspace(0.47, 0.53, n_points_params):
                                        t0 = time.time_ns()

                                        sim = SimODEDiscrete(
                                            K=K,
                                            y0=lambda x: [y0_1, y0_2, y0_3, y0_4],
                                            dt=1e-2,
                                        )
                                        sim.betaE = betaE
                                        sim.nuE = nuE
                                        sim.deltaE = deltaE
                                        sim.deltaF = deltaF
                                        sim.deltaM = deltaM
                                        sim.nu = nu
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

                                        total_time += time.time_ns() - t0
                                        N += 1

                                        # print how many sims done/total number, and time elapsed/estimated time left
                                        time_left_s = total_time / 1e9 * (N_total - N) / N
                                        print(
                                            f"{N}/{N_total} done, {total_time / 1e9:.2f}s elapsed, {time_left_s/60.0:.0f}min = {time_left_s/3600.0:.0f}h left",
                                        )

all_ys = np.array(all_ys)  # shape (n_sims, n_times, 4) - E, M, F, Ms
all_us = np.array(all_us)  # shape (n_sims, n_times)
n_sims = len(all_ys)

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
plt.savefig("distributions_larger_ic.png")
