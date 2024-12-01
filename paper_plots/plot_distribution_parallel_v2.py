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
    tmax = 1000

    sim = SimODEDiscrete(
        K=K,
        y0=lambda x: [y0_1, y0_2, y0_3, y0_4],
        dt=1e-3,
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

    ys = np.array(sim.y_lst)[::1000]  # (n_times, 4): E, M, F, Ms
    # ts = np.array(sim.t_lst)[::100]  # (n_times,)
    us = np.array([0] + u_lst)[::1000]  # (n_times,)

    return ys, us  # , ts


def main():
    K = 50000.0

    # first plot: sweep over initial conditions in [0, 20K] (previously [0, 10K] while training was [0, 5K])
    # keep system parameters constant

    betaE = 8
    nuE = 0.25
    deltaE = 0.03
    deltaF = 0.04
    deltaM = 0.1
    nu = 0.49

    # n_sweeps = 6  # 6
    # param_combinations = list(
    #     itertools.product(
    #         np.linspace(K, 20 * K, n_sweeps),
    #         np.linspace(K, 20 * K, n_sweeps),
    #         np.linspace(K, 20 * K, n_sweeps),
    #         np.linspace(K, 20 * K, n_sweeps),
    #         [betaE],
    #         [nuE],
    #         [deltaE],
    #         [deltaF],
    #         [deltaM],
    #         [nu],
    #     )
    # )

    # lets try sth random instead
    param_combinations = []

    for _ in range(1000):
        y0_1 = np.random.uniform(0.1, 20 * K)
        y0_2 = np.random.uniform(0.1, 20 * K)
        y0_3 = np.random.uniform(0.1, 20 * K)
        y0_4 = np.random.uniform(0.1, 20 * K)
        param_combinations.append([y0_1, y0_2, y0_3, y0_4, betaE, nuE, deltaE, deltaF, deltaM, nu])
    # and include the boundary points
    param_combinations.append([0.1, 0.1, 0.1, 0.1, betaE, nuE, deltaE, deltaF, deltaM, nu])
    param_combinations.append([20 * K, 20 * K, 20 * K, 20 * K, betaE, nuE, deltaE, deltaF, deltaM, nu])

    # param_combinations = list(
    #     itertools.product(
    #         np.linspace(K, 15 * K, n_points_ic),
    #         np.linspace(K, 15 * K, n_points_ic),
    #         np.linspace(K, 15 * K, n_points_ic),
    #         np.linspace(K, 15 * K, n_points_ic),
    #         np.linspace(7.46, 14.85, n_points_params),
    #         np.linspace(0.005, 0.25, n_points_params),
    #         np.linspace(0.023, 0.046, n_points_params),
    #         np.linspace(0.033, 0.046, n_points_params),
    #         np.linspace(0.077, 0.139, n_points_params),
    #         np.linspace(0.47, 0.53, n_points_params),
    #     )
    # )

    N_total = len(param_combinations)
    print("total number of sweeps: ", N_total)

    start_time = time.time()

    # Run simulations in parallel
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(run_simulation, param_combinations), total=N_total))

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")

    all_ys = np.array([result[0] for result in results])  # shape (n_sims, n_times, 4)
    all_us = np.array([result[1] for result in results])  # shape (n_sims, n_times)
    # all_ts = np.array([result[2] for result in results])  # shape (n_sims, n_times)
    ts_ys = np.linspace(0, 1000, all_ys.shape[1])
    ts_us = np.linspace(0, 1000, all_us.shape[1])
    n_sims = len(all_ys)

    # np.savez("distrib_parallel.npz", all_ys=all_ys, all_us=all_us, all_ts=all_ts)  # about 4MB per sweep...

    fig, axes = plt.subplots(nrows=5, figsize=(6, 8), sharex=True, dpi=300)
    for k in range(5):
        ax = axes[k]
        if k < 4:
            for j in range(n_sims):
                ax.plot(ts_ys, all_ys[j, :, k], linewidth=1.0)  # 'k'
        else:
            for j in range(n_sims):
                ax.plot(ts_us, all_us[j,], linewidth=1.0)  # 'k'
            ax.set_xlabel("Time (days)")
        ax.set_ylabel(["E(t)", "M(t)", "F(t)", "M$_s$(t)", "u(t)"][k])
        ax.grid()
    plt.tight_layout()
    plt.savefig("distributions_larger_ic_fixed_params.png")


if __name__ == "__main__":
    main()
