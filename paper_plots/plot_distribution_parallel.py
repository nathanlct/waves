import itertools
import sys
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")

from tqdm import tqdm

from models.sim_ode_discrete import SimODEDiscrete


def get_action(total_males, total_females):
    if total_females == 0 or (total_males != 0 and np.log(total_males / total_females) > 4):
        action = 5
    else:
        action = 300000
    return action


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

    return ys, us, ts


def main():
    n_points_ic = 2  # 5
    n_points_params = 1  # 3
    K = 50000.0

    # Generate all parameter combinations
    param_combinations = list(
        itertools.product(
            np.linspace(K, 15 * K, n_points_ic),
            np.linspace(K, 15 * K, n_points_ic),
            np.linspace(K, 15 * K, n_points_ic),
            np.linspace(K, 15 * K, n_points_ic),
            np.linspace(7.46, 14.85, n_points_params),
            np.linspace(0.005, 0.25, n_points_params),
            np.linspace(0.023, 0.046, n_points_params),
            np.linspace(0.033, 0.046, n_points_params),
            np.linspace(0.077, 0.139, n_points_params),
            np.linspace(0.47, 0.53, n_points_params),
        )
    )

    N_total = len(param_combinations)
    print("total number of sweeps: ", N_total)

    start_time = time.time()

    # Run simulations in parallel
    with Pool() as pool:
        results = list(tqdm(pool.imap(run_simulation, param_combinations), total=N_total))

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")

    all_ys = np.array([result[0] for result in results])  # shape (n_sims, n_times, 4)
    all_us = np.array([result[1] for result in results])  # shape (n_sims, n_times)
    all_ts = np.array([result[2] for result in results])  # shape (n_sims, n_times)
    n_sims = len(all_ys)

    # np.savez("distrib_parallel.npz", all_ys=all_ys, all_us=all_us, all_ts=all_ts)  # about 4MB per sweep...

    fig, axes = plt.subplots(nrows=5, figsize=(6, 8), sharex=True, dpi=300)
    for k in range(5):
        ax = axes[k]
        if k < 4:
            for j in range(n_sims):
                ax.plot(all_ts[j,], all_ys[j, :, k], linewidth=1.0)  # 'k'
        else:
            for j in range(n_sims):
                ax.plot(all_ts[j,], all_us[j,], linewidth=1.0)  # 'k'
            ax.set_xlabel("Time (days)")
        ax.set_ylabel(["E(t)", "M(t)", "F(t)", "M$_s$(t)", "u(t)"][k])
        ax.grid()
    plt.tight_layout()
    plt.savefig("distributions_larger_ic.png")


if __name__ == "__main__":
    main()
