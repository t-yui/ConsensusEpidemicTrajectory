#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# running command with 5 sample trajectories and K basis functions:
# python simulation.py --J 5 --K 30 --R 1

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
from scipy.integrate import cumtrapz
from frechet_mean_optimization import run_SEIR_frechet_pipeline
from logzero import logger

# argparse
parser = argparse.ArgumentParser(description="Run SEIR Fréchet pipeline")
parser.add_argument("--J", type=int, default=5, help="Number of sample trajectories")
parser.add_argument(
    "--K", type=int, default=30, help="Number of B-spline basis functions"
)
parser.add_argument(
    "--R", type=int, default=1, help="Inverse Scaling Parameter for Gram Matrix"
)
args = parser.parse_args()
J = args.J
K = args.K
R = args.R
rho = 1/R

# settings
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)

N = 1e6
E0 = 1
I0 = 0
T = 720
n_time = 100
t_eval = np.linspace(0, T, n_time)
n_mc = 1

# baseline parameters
beta_point = 2.2 / 7.5
sigma_point = 1 / 5.2
gamma_point = 1 / 7.5


def simulate_SEIR(beta, sigma, gamma, N, E0, I0, T, t_eval):
    from scipy.integrate import solve_ivp

    def seir_model(t, y, beta, sigma, gamma, N):
        S, E, I, R = y
        dS = -beta * S * I / N
        dE = beta * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        return [dS, dE, dI, dR]

    S0 = N - E0 - I0
    R0 = 0
    y0 = [S0, E0, I0, R0]
    sol = solve_ivp(seir_model, [0, T], y0, args=(beta, sigma, gamma, N), t_eval=t_eval)
    return sol.t, sol.y


def align_trajectory(t, y, ref_peak):
    idx_peak = np.argmax(y)
    t_peak = t[idx_peak]
    shift = ref_peak - t_peak
    t_warped = t + shift
    y_aligned = np.interp(t, t_warped, y)
    return y_aligned, shift


# simulate point estimate trajectory
t_sim, sol_sim = simulate_SEIR(
    beta_point, sigma_point, gamma_point, N, E0, I0, T, t_eval
)
S_point, E_point, I_point, R_point = sol_sim


def sample_with_CI(mu, sd_left, sd_right, positive=True, eps=1e-4):
    z = np.random.normal()
    sd = sd_left if z < 0 else sd_right
    val = mu + sd * z
    return max(val, eps) if positive else val


def estimate_params_from_CI(point, lower, upper, z975=1.96):
    if not (lower < point < upper):
        raise ValueError("Require lower < point < upper.")
    sd_left = (point - lower) / z975
    sd_right = (upper - point) / z975
    return point, sd_left, sd_right


D_E_mu, D_E_sd_left, D_E_sd_right = estimate_params_from_CI(5.2, 4.1, 7.0)
D_I_mu, D_I_sd_left, D_I_sd_right = estimate_params_from_CI(7.5, 5.3, 19.0)
R0_mu, R0_sd_left, R0_sd_right = estimate_params_from_CI(2.2, 1.4, 3.9)

q_list = [1, 1.5, 2]
q_tag = {1: "q1", 1.5: "q15", 2: "q2"}
param_names = ["beta", "sigma", "gamma", "R0", "Incubation", "Infectious"]
fieldnames = ["iteration", "J", "K"] + [
    f"{p}_{q_tag[q]}" for q in q_list for p in param_names
]

rows = []
results_rep = None
trajectories_rep = None

for it in range(n_mc):
    logger.info("===== it = {0}".format(it))
    np.random.seed(it)
    trajectories = []
    for j in range(J):
        D_E = sample_with_CI(D_E_mu, D_E_sd_left, D_E_sd_right)
        D_I = sample_with_CI(D_I_mu, D_I_sd_left, D_I_sd_right)
        R0 = sample_with_CI(R0_mu, R0_sd_left, R0_sd_right)
        sigma = 1.0 / D_E
        gamma = 1.0 / D_I
        beta = R0 * gamma
        t, sol = simulate_SEIR(beta, sigma, gamma, N, E0, I0, T, t_eval)
        trajectories.append(
            {
                "t": t,
                "E_orig": sol[1].copy(),
                "I_orig": sol[2].copy(),
                "params": {
                    "beta": beta,
                    "sigma": sigma,
                    "gamma": gamma,
                    "R0": R0,
                    "D_E": D_E,
                    "D_I": D_I,
                },
            }
        )

    peak_times = [np.argmax(traj["I_orig"]) for traj in trajectories]
    ref_peak = np.median([traj["t"][idx] for traj, idx in zip(trajectories, peak_times)])

    for traj in trajectories:
        traj["E_aligned"], _ = align_trajectory(traj["t"], traj["E_orig"], ref_peak)
        traj["I_aligned"], _ = align_trajectory(traj["t"], traj["I_orig"], ref_peak)

    results = run_SEIR_frechet_pipeline(trajectories, t_eval, N, K=K, rho=rho)

    row = {"iteration": it + 1, "J": J, "K": K}
    for q in q_list:
        tag = q_tag[q]
        beta = results[q]["beta"]
        sigma = results[q]["sigma"]
        gamma = results[q]["gamma"]
        R0 = beta / gamma
        incubation = 1.0 / sigma
        infectious = 1.0 / gamma
        row[f"beta_{tag}"] = beta
        row[f"sigma_{tag}"] = sigma
        row[f"gamma_{tag}"] = gamma
        row[f"R0_{tag}"] = R0
        row[f"Incubation_{tag}"] = incubation
        row[f"Infectious_{tag}"] = infectious
    rows.append(row)

    if it == 0:
        results_rep = results
        trajectories_rep = trajectories

        # print estimated parameters
        print("\nEstimated Parameters Table (J={}, K={})".format(J, K))
        print("{:<3} {:<3} {:<8} {:<8} {:<8} {:<8} {:<12} {:<12}".format(
            "J", "q", "beta", "sigma", "gamma", "R0", "Incubation", "Infectious"))
        for q in [1, 1.5, 2]:
            beta = results[q]['beta']
            sigma = results[q]['sigma']
            gamma = results[q]['gamma']
            R0 = beta / gamma
            incubation = 1.0 / sigma
            infectious = 1.0 / gamma
            print("{:<3} {:<3} {:.4f}   {:.4f}   {:.4f}   {:.2f}   {:.2f}       {:.2f}".format(
                J, q, beta, sigma, gamma, R0, incubation, infectious))

        # visualization
        E_median = results[1]["E"]
        I_median = results[1]["I"]
        S_median = results[1]["S"]
        R_median = results[1]["R"]

        E_mean = results[2]["E"]
        I_mean = results[2]["I"]
        S_mean = results[2]["S"]
        R_mean = results[2]["R"]

        traj_list_I = []
        traj_list_E = []
        for i, traj in enumerate(trajectories):
            traj_list_I.append(traj["I_orig"])
            traj_list_E.append(traj["E_orig"])

        ## plot E
        plt.figure(figsize=(12, 6))
        for i, traj in enumerate(trajectories):
            plt.plot(
                traj["t"],
                traj["E_orig"],
                color="gray",
                alpha=0.8,
                linewidth=2.5,
                label="Individual trajectories" if i == 0 else "",
            )
        plt.plot(t_eval, np.mean(np.array(traj_list_E),axis=0), label="Simple Pointwise Mean", linewidth=5, color="green")
        plt.plot(t_eval, np.median(np.array(traj_list_E),axis=0), label="Simple Pointwise Median", linewidth=5, color="orange")
        plt.plot(t_eval, E_mean, label="Fréchet Mean ($q=2$)", linewidth=5, color="blue")
        plt.plot(t_eval, results[1.5]["E"], label="Fréchet $q$-Mean ($q=1.5$)", linewidth=5, color="purple")
        plt.plot(t_eval, E_median, label="Fréchet Median ($q=1$)", linewidth=5, color="red")
        plt.xlabel("Time")
        plt.ylabel("Exposed Population")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"SEIR_E_representative_J{J}_K{K}_R{R}.pdf", bbox_inches="tight")
        plt.close()

        ## plot I
        plt.figure(figsize=(12, 6))
        for i, traj in enumerate(trajectories):
            plt.plot(
                traj["t"],
                traj["I_orig"],
                color="gray",
                alpha=0.8,
                linewidth=2.5,
                label="Individual trajectories" if i == 0 else "",
            )
        plt.plot(t_eval, np.mean(np.array(traj_list_I),axis=0), label="Simple Pointwise Mean", linewidth=5, color="green")
        plt.plot(t_eval, np.median(np.array(traj_list_I),axis=0), label="Simple Pointwise Median", linewidth=5, color="orange")
        plt.plot(t_eval, I_mean, label="Fréchet Mean ($q=2$)", linewidth=5, color="blue")
        plt.plot(t_eval, results[1.5]["I"], label="Fréchet $q$-Mean ($q=1.5$)", linewidth=5, color="purple")
        plt.plot(t_eval, I_median, label="Fréchet Median ($q=1$)", linewidth=5, color="red")
        plt.xlabel("Days")
        plt.ylabel("Infectious Population")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"SEIR_I_representative_J{J}_K{K}_R{R}.pdf", bbox_inches="tight")
        plt.close()

        ## plot full compartments (q=1 only)
        plt.figure(figsize=(12, 6))
        plt.plot(t_eval, S_median, label="Susceptible (q=1)", linewidth=5, color="blue")
        plt.plot(t_eval, E_median, label="Exposed (q=1)", linewidth=5, color="green")
        plt.plot(t_eval, I_median, label="Infectious (q=1)", linewidth=5, color="red")
        plt.plot(t_eval, R_median, label="Recovered (q=1)", linewidth=5, color="purple")
        plt.plot(
            t_eval,
            S_point,
            label="Susceptible (Point Est.)",
            linestyle="--",
            linewidth=3,
            color="blue",
        )
        plt.plot(
            t_eval,
            E_point,
            label="Exposed (Point Est.)",
            linestyle="--",
            linewidth=3,
            color="green",
        )
        plt.plot(
            t_eval,
            I_point,
            label="Infectious (Point Est.)",
            linestyle="--",
            linewidth=3,
            color="red",
        )
        plt.plot(
            t_eval,
            R_point,
            label="Recovered (Point Est.)",
            linestyle="--",
            linewidth=3,
            color="purple",
        )
        plt.xlabel("Days")
        plt.ylabel("Population")
        plt.legend(loc="center left")
        plt.xlim(-5, 365)
        plt.tight_layout()
        plt.savefig(
            f"SEIR_full_representative_and_point_estimate_J{J}_K{K}_R{R}.pdf", bbox_inches="tight"
        )
        plt.close()


csv_name = f"SEIR_estimated_params_MC{n_mc}_J{J}_K{K}_R{R}.csv"
with open(csv_name, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

results = results_rep
trajectories = trajectories_rep

