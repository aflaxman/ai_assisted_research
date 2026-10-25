"""Run every experiment reported in Duffus et al. (2026), Pathogens 15(8):827,
and print side-by-side comparisons with the published values.

Usage:  python replicate_results.py [--ponds 10000] [--seed 0]

Produces:
  results/table1_replication.csv   (mortality sweep, paper Table 1)
  results/table2_replication.csv   (contact-rate sweep, paper Table 2)
  results/figure4_replication.png  (mean state counts vs. iteration)
  results/figure5to8_replication.png (mean curves with min/max bands)
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ranavirus_ca import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

# Published values (transcribed from the paper)
PAPER_TABLE1 = {  # omega -> (all_infected, all_dead, pond_count)
    0.075: (9.84, 71.12, 9832),
    0.100: (9.95, 54.41, 9904),
    0.125: (10.01, 44.39, 9799),
    0.150: (10.02, 37.64, 9626),
    0.175: (9.97, 37.79, 9368),
    0.200: (9.74, 29.39, 8950),
    0.225: (9.41, 26.85, 8392),
    0.250: (8.84, 24.76, 7670),
}
PAPER_TABLE2 = {  # psi -> (all_infected, all_dead, pond_count); mu = psi * sigma
    0.75: (11.01, 38.53, 9234),
    0.80: (10.91, 38.29, 9395),
    0.85: (10.63, 37.99, 9453),
    0.90: (10.35, 37.85, 9472),
    0.95: (10.23, 37.79, 9587),
}
SIGMA_U, SIGMA_H = 0.65, 0.75


def run_table1(n_ponds, seed):
    rows = []
    for i, (omega, paper) in enumerate(sorted(PAPER_TABLE1.items())):
        res = simulate(pu=SIGMA_U, ph=SIGMA_H, pd=omega, n_ponds=n_ponds, seed=seed + i)
        rows.append({
            "omega": omega,
            "all_infected_paper": paper[0], "all_infected_repl": round(res["all_infected_mean"], 2),
            "all_dead_paper": paper[1], "all_dead_repl": round(res["all_dead_mean"], 2),
            "pond_count_paper": paper[2], "pond_count_repl": res["dead_pond_count"],
        })
        print(f"  omega={omega:.3f}  all_infected {paper[0]:6.2f} vs {res['all_infected_mean']:6.2f}   "
              f"all_dead {paper[1]:6.2f} vs {res['all_dead_mean']:6.2f}   "
              f"ponds_dead {paper[2]:5d} vs {res['dead_pond_count']:5d}")
    return rows


def run_table2(n_ponds, seed):
    rows = []
    for i, (psi, paper) in enumerate(sorted(PAPER_TABLE2.items())):
        mu_u, mu_h = psi * SIGMA_U, psi * SIGMA_H
        res = simulate(pu=mu_u, ph=mu_h, pd=0.15, n_ponds=n_ponds, seed=seed + 100 + i)
        rows.append({
            "psi": psi, "mu_U": mu_u, "mu_H": mu_h,
            "all_infected_paper": paper[0], "all_infected_repl": round(res["all_infected_mean"], 2),
            "all_dead_paper": paper[1], "all_dead_repl": round(res["all_dead_mean"], 2),
            "pond_count_paper": paper[2], "pond_count_repl": res["dead_pond_count"],
        })
        print(f"  psi={psi:.2f}  all_infected {paper[0]:6.2f} vs {res['all_infected_mean']:6.2f}   "
              f"all_dead {paper[1]:6.2f} vs {res['all_dead_mean']:6.2f}   "
              f"ponds_dead {paper[2]:5d} vs {res['dead_pond_count']:5d}")
    return rows


STATE_STYLE = [  # (index, label, color) matching the original plots
    (0, "Susceptible", "green"),
    (1, "Ulcerative", "xkcd:dark yellow"),
    (2, "Hemorrhagic", "red"),
    (3, "Combined", "orange"),
    (4, "Dead", "black"),
]


def make_figure4(res, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(res["mean"].shape[0])
    for k, label, color in STATE_STYLE:
        ax.plot(x, res["mean"][:, k], color=color, label=label)
    ax.set_xlabel("Time (iterations)")
    ax.set_ylabel("# Frogs in given status")
    ax.set_title("Frog Status Count vs. Time (mean of %d ponds)" % res["params"]["n_ponds"])
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_band_figures(res, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (k, label, color) in zip(axes.flat, STATE_STYLE[1:]):
        x = range(1, res["mean"].shape[0])
        ax.plot(x, res["mean"][1:, k], color=color)
        ax.fill_between(x, res["min"][1:, k], res["max"][1:, k], alpha=0.4,
                        color="gray" if k == 4 else color)
        ax.set_title(f"{label}: mean with min/max band")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("# Frogs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ponds", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print(f"Baseline scenario (sigma_U={SIGMA_U}, sigma_H={SIGMA_H}, omega=0.15, "
          f"{args.ponds} ponds, 120 iterations)")
    base = simulate(pd=0.15, n_ponds=args.ponds, seed=args.seed)
    com_peak = base["mean"][:, 3].max()
    com_peak_at = base["mean"][:, 3].argmax()
    print(f"  mean iteration to all infected: {base['all_infected_mean']:.2f} (paper: 10.02)")
    print(f"  mean iteration to all dead:     {base['all_dead_mean']:.2f} (paper: 37.64)")
    print(f"  ponds fully depopulated:        {base['dead_pond_count']} (paper: 9626)")
    print(f"  peak of mean Combined curve:    {com_peak:.1f} at iteration {com_peak_at} (paper: ~70)")
    make_figure4(base, os.path.join(OUT, "figure4_replication.png"))
    make_band_figures(base, os.path.join(OUT, "figure5to8_replication.png"))

    print("\nTable 1 replication (mortality sweep; paper value vs. replication)")
    t1 = run_table1(args.ponds, args.seed)
    write_csv(t1, os.path.join(OUT, "table1_replication.csv"))

    print("\nTable 2 replication (contact-rate sweep; paper value vs. replication)")
    t2 = run_table2(args.ponds, args.seed)
    write_csv(t2, os.path.join(OUT, "table2_replication.csv"))

    print(f"\nOutputs written to {OUT}/")


if __name__ == "__main__":
    main()
