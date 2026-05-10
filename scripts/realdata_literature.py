#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from frechet_mean_optimization import run_SEIR_frechet_pipeline


def simulate_SEIR(beta, sigma, gamma, N, E0, I0, T, t_eval):
    def seir_model(t, y):
        S, E, I, R = y
        dS = -beta * S * I / N
        dE = beta * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        return [dS, dE, dI, dR]

    S0 = N - E0 - I0
    y0 = [S0, E0, I0, 0.0]
    sol = solve_ivp(seir_model, [0, T], y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--T", type=float, default=720.0)
    parser.add_argument("--n_time", type=int, default=721)
    parser.add_argument("--delta_max", type=float, default=120.0)
    parser.add_argument("--rho", type=float, default=1.0)
    args = parser.parse_args()

    K = args.K
    T = float(args.T)
    n_time = int(args.n_time)

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
    E0 = 1.0
    I0 = 0.0
    t_eval = np.linspace(0.0, T, n_time)

    studies = [
        {"key": "Wu et al. (2020)",      "label": "Wu et al. (Lancet 2020)",
         "region": "Wuhan, China",   "model": "SEIR",
         "D_E": 6.0,  "D_I": 2.4,   "R0": 2.68},
        {"key": "Tang et al. (2020)",       "label": "Tang et al. (JCM 2020)",
         "region": "Wuhan, China",   "model": "SEIR with quarantine",
         "D_E": 7.0,  "D_I": 1.16,   "R0": 6.47},
        {"key": "Peirlinck et al. (2020)", "label": "Peirlinck et al. (CMBE 2020)",
         "region": "United States",   "model": "SEIR",
         "D_E": 2.56, "D_I": 17.82, "R0": 5.30},
        {"key": "Carcione et al. (2020)",   "label": "Carcione et al. (FPH 2020)",
         "region": "Lombardy, Italy", "model": "SEIRD",
         "D_E": 4.25, "D_I": 4.02,   "R0": 3.0},
        {"key": "Wan2020 et al. (2020)",       "label": "Wan et al. (JOGH 2020)",
         "region": "Wuhan, China",    "model": "SEIR",
         "D_E": 3.0,  "D_I": 14.0,  "R0": 1.44},
        {"key": "Dagpunar (2020)",   "label": "Dagpunar (IDM 2020)",
         "region": "United Kingdom", "model": "SEIR",
         "D_E": 4.5,  "D_I": 3.8,   "R0": 3.18},
    ]

    trajectories = []
    for st in studies:
        sigma = 1.0 / st["D_E"]
        gamma = 1.0 / st["D_I"]
        beta = st["R0"] * gamma
        t, sol = simulate_SEIR(beta, sigma, gamma, N, E0, I0, T, t_eval)
        trajectories.append(
            {
                "t": t,
                "E_orig": sol[1].copy(),
                "I_orig": sol[2].copy(),
                "source": st["label"],
                "params": {"beta": beta, "sigma": sigma, "gamma": gamma, **st},
            }
        )

    results = run_SEIR_frechet_pipeline(
        trajectories, t_eval, N, K=K, rho=args.rho, delta_max=args.delta_max
    )

    # --- Print inputs ---
    print("\nInput parameter sets (mapped to baseline SEIR)")
    header = "{:<22} {:<35} {:<28} {:<32} {:>5} {:>6} {:>6} {:>9} {:>9} {:>9}".format(
        "key", "label", "region", "model",
        "R0", "D_E", "D_I", "beta", "sigma", "gamma"
    )
    print(header)
    for st in studies:
        sigma = 1.0 / st["D_E"]
        gamma = 1.0 / st["D_I"]
        beta = st["R0"] * gamma
        print(
            "{:<22} {:<35} {:<28} {:<32} {:5.2f} {:6.2f} {:6.2f} {:9.4f} {:9.4f} {:9.4f}"
            .format(
                st["key"], st["label"][:35], st["region"][:28], st["model"][:32],
                st["R0"], st["D_E"], st["D_I"], beta, sigma, gamma,
            )
        )

    # --- Print outputs ---
    print(
        "\nEstimated representative-curve parameters "
        "(N={}, T={} days, J={})".format(int(N), int(T), len(trajectories))
    )
    print(
        "{:<3} {:<4} {:>9} {:>9} {:>9} {:>7} {:>11} {:>11}".format(
            "J", "q", "beta", "sigma", "gamma", "R0", "1/sigma", "1/gamma"
        )
    )
    for q in (1.0, 1.5, 2.0):
        beta = results[q]["beta"]
        sigma = results[q]["sigma"]
        gamma = results[q]["gamma"]
        R0 = beta / gamma
        print(
            "{:<3} {:<4} {:9.4f} {:9.4f} {:9.4f} {:7.2f} {:11.2f} {:11.2f}".format(
                len(trajectories), q, beta, sigma, gamma, R0, 1.0 / sigma, 1.0 / gamma,
            )
        )

    # --- Plots ---
    traj_E = np.stack([tr["E_orig"] for tr in trajectories], axis=0)
    traj_I = np.stack([tr["I_orig"] for tr in trajectories], axis=0)

    E_med, I_med = results[1.0]["E"], results[1.0]["I"]
    S_med, R_med = results[1.0]["S"], results[1.0]["R"]

    E_mean, I_mean = results[2.0]["E"], results[2.0]["I"]
    E_q, I_q = results[1.5]["E"], results[1.5]["I"]

    J = len(trajectories)
    individual_linestyles = ["-", ":", "--", (0, (5, 2)), "-.", (0, (3, 1, 1, 1))]

    # Exposed plot
    plt.figure(figsize=(12, 6))
    for i, tr in enumerate(trajectories):
        plt.plot(
            tr["t"], tr["E_orig"],
            color="gray", alpha=0.6, linewidth=2.0,
            linestyle=individual_linestyles[i % len(individual_linestyles)],
            label=tr["params"]["key"],
        )
    plt.plot(t_eval, np.mean(traj_E, axis=0),   label="Simple Pointwise Mean",   linewidth=3.0, color="green")
    plt.plot(t_eval, np.median(traj_E, axis=0), label="Simple Pointwise Median", linewidth=3.0, color="orange")
    plt.plot(t_eval, E_mean, label="Fr\u00e9chet Mean (q=2)",    linewidth=3.0, color="blue")
    plt.plot(t_eval, E_q,    label="Fr\u00e9chet q-Mean (q=1.5)", linewidth=3.0, color="purple")
    plt.plot(t_eval, E_med,  label="Fr\u00e9chet Median (q=1)",   linewidth=3.0, color="red")
    plt.xlabel("Days")
    plt.ylabel("Exposed Population")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        "SEIR_E_representative_realdata_J{}_K{}.pdf".format(J, K),
        bbox_inches="tight",
    )
    plt.close()

    # Infectious plot
    plt.figure(figsize=(12, 6))
    for i, tr in enumerate(trajectories):
        plt.plot(
            tr["t"], tr["I_orig"],
            color="gray", alpha=0.6, linewidth=2.0,
            linestyle=individual_linestyles[i % len(individual_linestyles)],
            label=tr["params"]["key"],
        )
    plt.plot(t_eval, np.mean(traj_I, axis=0),   label="Simple Pointwise Mean",   linewidth=3.0, color="green")
    plt.plot(t_eval, np.median(traj_I, axis=0), label="Simple Pointwise Median", linewidth=3.0, color="orange")
    plt.plot(t_eval, I_mean, label="Fr\u00e9chet Mean (q=2)",    linewidth=3.0, color="blue")
    plt.plot(t_eval, I_q,    label="Fr\u00e9chet q-Mean (q=1.5)", linewidth=3.0, color="purple")
    plt.plot(t_eval, I_med,  label="Fr\u00e9chet Median (q=1)",   linewidth=3.0, color="red")
    plt.xlabel("Days")
    plt.ylabel("Infectious Population")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        "SEIR_I_representative_realdata_J{}_K{}.pdf".format(J, K),
        bbox_inches="tight",
    )
    plt.close()

    # Full compartments recovered from the Fr\'echet median (q=1)
    plt.figure(figsize=(12, 6))
    plt.plot(t_eval, S_med, label="Susceptible (q=1)", linewidth=5, color="blue")
    plt.plot(t_eval, E_med, label="Exposed (q=1)",     linewidth=5, color="green")
    plt.plot(t_eval, I_med, label="Infectious (q=1)",  linewidth=5, color="red")
    plt.plot(t_eval, R_med, label="Recovered (q=1)",   linewidth=5, color="purple")
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.legend(loc="center right")
    plt.xlim(-5, T)
    plt.tight_layout()
    plt.savefig(
        "SEIR_full_representative_realdata_J{}_K{}.pdf".format(J, K),
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()

