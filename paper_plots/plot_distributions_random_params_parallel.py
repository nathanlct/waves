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
    tmax = 800

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

    ys = np.array(sim.y_lst)[::1]  # (n_times, 4): E, M, F, Ms
    # ts = np.array(sim.t_lst)  # (n_times,)
    us = np.array([0] + u_lst)[::1]  # (n_times,)

    # print(round(sim.betaE, 2), round(np.sum(ys[-1][:3]), 2), " /// ", np.max(ys[-1]), ys[-1])

    val_200days = np.sum(ys[int(len(ys) * 0.25)][:3])  # |E|+|F|+|M|
    val_400days = np.sum(ys[int(len(ys) * 0.5)][:3])  # |E|+|F|+|M|
    val_600days = np.sum(ys[int(len(ys) * 0.75)][:3])  # |E|+|F|+|M|
    val_800days = np.sum(ys[-1][:3])  # |E|+|F|+|M|
    # print(ys.shape)

    if val_800days > 1:
        print(f"{betaE:.2f} & {nuE:.2f} & {deltaE:.2f} & {deltaF:.2f} & {deltaM:.2f} & {nu:.2f} & {val_800days:.2f}")

    return val_200days, val_400days, val_600days, val_800days


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

    n_sweeps = 8  # 10

    betaE_lst = np.linspace(
        7.46, 14.85, n_sweeps
    )  # causes pb: for largest value, M_s doesnt converge (stays constant = 3M) and same for E and M (stay constant = 40k) (Oviposition rate, ie how many eggs are laid by females: if too high, the max Ms(t) value is not enough to destroy the population?) -- but even making u_max very large (10M), while it reduces the convergence values, doesnt seem to converge to 0 still (equilibrium?)
    nuE_lst = np.linspace(0.005, 0.25, n_sweeps)  # no pb
    deltaE_lst = np.linspace(0.023, 0.046, n_sweeps)  # no pb
    deltaF_lst = np.linspace(0.033, 0.046, n_sweeps)  # no pb
    deltaM_lst = np.linspace(
        0.077, 0.139, n_sweeps
    )  # for the largest value, M_s doesnt converge (stays constant = 3M) (death rate of males: if too high, need to keep producing them?) -- same, large u_max doesnt help, it just keeps outputting the max action all the time, even when others states seem to have converged. could be OOD behavior, or the states havent converged enough and wont but it doesnt know it
    nu_lst = np.linspace(0.47, 0.53, n_sweeps)  # no pb

    n_sweeps = 32

    # possibly reduced ranges
    betaE_lst = np.linspace(0, 20, n_sweeps)[1:15]
    nuE_lst = np.linspace(0.0, 1.0, n_sweeps)[1:12]  # no pb
    deltaE_lst = np.linspace(0.0, 1.0, n_sweeps)[1:]  # no pb
    deltaF_lst = np.linspace(0.0, 0.1, n_sweeps)[10:]  # no pb
    deltaM_lst = np.linspace(0.0, 0.2, n_sweeps)[14:]
    nu_lst = np.linspace(0.01, 1.0, n_sweeps)[12:]  # no pb

    # params list
    sweep_params = [betaE_lst, nuE_lst, deltaE_lst, deltaF_lst, deltaM_lst, nu_lst]

    print(f"{betaE_lst=}")
    print(f"{nuE_lst=}")
    print(f"{deltaE_lst=}")
    print(f"{deltaF_lst=}")
    print(f"{deltaM_lst=}")
    print(f"{nu_lst=}")

    params_combinations = []

    # then, lets add some random points as well
    for _ in range(10000):
        params = [3.0 * K, 3.0 * K, 3.0 * K, 3.0 * K, betaE, nuE, deltaE, deltaF, deltaM, nu]
        for i in range(4):
            params[i] = np.random.uniform(0, 10 * K)
        for i in range(6):
            params[4 + i] = np.random.choice(sweep_params[i])
        params_combinations.append(params)

    N_total = len(params_combinations)
    print("total number of sweeps: ", N_total)

    start_time = time.time()

    # Run simulations in parallel
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(run_simulation, params_combinations), total=N_total))
    results = np.array(results)

    print(np.mean(results, axis=0))
    print(np.std(results, axis=0))
    print(np.max(results, axis=0))

    import sys

    sys.exit(0)

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
