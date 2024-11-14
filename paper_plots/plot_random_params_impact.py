import itertools
import sys
import time
from multiprocessing import Pool

import matplotlib.patches as patches
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

    print(round(sim.betaE, 2), round(np.sum(ys[-1][:3]), 2), " /// ", np.max(ys[-1]), ys[-1])

    params = (betaE, nuE, deltaE, deltaF, deltaM, nu)
    final_val = np.sum(ys[-1][:3])  # |E|+|F|+|M|

    return params, final_val


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

    train_params = [betaE, nuE, deltaE, deltaF, deltaM, nu]

    n_sweeps = 32  # 10

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

    # ranges for plots
    n_sweeps = 32  # 10

    betaE_lst = np.linspace(0, 20, n_sweeps)
    nuE_lst = np.linspace(0.0, 1.0, n_sweeps)  # no pb
    deltaE_lst = np.linspace(0.0, 1.0, n_sweeps)  # no pb
    deltaF_lst = np.linspace(0.0, 0.1, n_sweeps)  # no pb
    deltaM_lst = np.linspace(0.0, 0.2, n_sweeps)
    nu_lst = np.linspace(0.01, 1.0, n_sweeps)  # no pb

    # params list
    sweep_params = [betaE_lst, nuE_lst, deltaE_lst, deltaF_lst, deltaM_lst, nu_lst]

    fig, axes = plt.subplots(3, 2, figsize=(12, 7), dpi=300)

    for i in range(6):  # loop over the 6 params we want to test robustness against
        params_combinations = []
        for k in range(n_sweeps):
            params = [3.0 * K, 3.0 * K, 3.0 * K, 3.0 * K, betaE, nuE, deltaE, deltaF, deltaM, nu]
            params[4 + i] = sweep_params[i][k]
            params_combinations.append(params)

        N_total = len(params_combinations)
        print("total number of sweeps: ", N_total)

        start_time = time.time()

        # Run simulations in parallel
        with Pool() as pool:
            results = list(tqdm(pool.imap(run_simulation, params_combinations), total=N_total))

        # sort by the considered param
        results.sort(key=lambda x, i=i: x[0][i])
        results_keys = [x[0][i] for x in results]
        results_vals = [x[1] for x in results]

        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f}s")

        ax = axes[i // 2, i % 2]

        ax.plot(results_keys, results_vals, "o-", color="black", label="Parameters at inference time")

        # Change color of dots based on threshold
        threshold = 0.1
        first_above = False
        first_below = False
        for j in range(len(results_keys)):
            if results_vals[j] > threshold:
                ax.scatter(
                    results_keys[j],
                    results_vals[j],
                    color="orange",
                    zorder=999,
                    label=f"Non-converged values ($>$ {threshold})" if not first_above else None,
                )  # red dot if above threshold
                first_above = True
            else:
                ax.scatter(
                    results_keys[j],
                    results_vals[j],
                    color="green",
                    zorder=999,
                    label=f"Converged values ($\leq$ {threshold})" if not first_below else None,
                )  # green dot if below threshold
                first_below = True

        train_val = train_params[i]
        ax.axvline(x=train_val, color="red", linestyle="--", linewidth=2, label="Training parameter")

        label_names = [
            "$\\beta_E$ : Oviposition rate (Effective female fecundity)",
            "$\\nu_E$ : Eggs hatching rate",
            "$\\delta_E$ : Eggs death rate (aquatic phase)",
            "$\\delta_F$ : Fertilized females death rate",
            "$\\delta_M$ : Wild adult males death rate",
            "$\\nu$ : Probability of emergence (probability to give rise to a female)",
        ]
        ax.set_xlabel(label_names[i])
        ax.set_ylabel("$E+M+F$")
        ax.set_yscale("log")

        # ax.grid()
        if i == 1:
            ax.legend()

        # fig, axes = plt.subplots(nrows=5, figsize=(6, 8), sharex=True, dpi=300)
        # for k in range(5):
        #     ax = axes[k]
        #     if k < 4:
        #         for j in range(n_sims):
        #             ax.plot(ts_ys, all_ys[j, :, k], linewidth=1.0)  # 'k'
        #     else:
        #         for j in range(n_sims):
        #             ax.plot(ts_us, all_us[j,], linewidth=1.0)  # 'k'
        #         ax.set_xlabel("Time (days)")
        #     ax.set_ylabel(["E(t)", "M(t)", "F(t)", "M$_s$(t)", "u(t)"][k])
        #     ax.grid()

    plt.tight_layout()
    plt.savefig(f"random_params_impact_all_log.png")


if __name__ == "__main__":
    main()
