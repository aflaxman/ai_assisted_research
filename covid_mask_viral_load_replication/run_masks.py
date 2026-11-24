"""Masking results of Goyal et al. (2021): Figures 2-5, Supplementary figures 3-5.

Every table pairs what the paper's stated model gives with what the paper
printed (read off its figures by the scripts in ``digitize/``).  For the
quantities that are means, the reference is the exact expectation of the
paper's Monte Carlo over all 10,000 published parameter sets; ``se_3000`` is
the standard error the paper's 3,000-transmitter runs carry around it.

    uv run python run_masks.py            # about 12 minutes
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

import transmission as tr
import viral_load as vl

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

#: The paper's "x% of people wearing masks y% of the time" panels.
COVERAGES = {"25/25": (0.25, 0.25), "50/50": (0.50, 0.50), "75/75": (0.75, 0.75),
             "90/90": (0.90, 0.90), "100/100": (1.0, 1.0)}
LEVELS = np.round(np.arange(0.0, 1.01, 0.1), 1)
N_PAPER = 3000          # transmitters per scenario in the paper
N_BIG = 200_000         # transmitters per scenario here, for distributional summaries
READING_ERROR = 0.02    # how well a heat-map cell can be read off the published figures
CLUSTER_BINS = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-10", 6, 10),
                ("11-20", 11, 20), ("21-50", 21, 50), (">50", 51, np.inf)]


def draw(V, rng, n, **kwargs) -> tr.Draws:
    return tr.simulate(V[rng.integers(0, len(V), n)], rng, **kwargs)


# ---------------------------------------------------------------------------

def baseline(V, rng) -> pd.DataFrame:
    """No masks: the numbers the eLife paper and this paper quote."""
    reps = [tr.summarize(draw(V, rng, 10_000).r_individual) for _ in range(300)]
    reps = pd.DataFrame(reps)
    big = draw(V, rng, 2_000_000)
    exact = tr.expected_re(V, 0.0, 0.0, 0.0)
    rows = [("mean R0, exact expectation of the unrounded sum", exact, np.nan, "~1.8 (this paper, Figs 3 and 7)"),
            ("mean R0, individual values rounded", big.r_individual.mean(), reps.mean_R.std(), "1.6 (eLife Fig 4: 15,992 infections / 10,000)"),
            ("SD of individual R0", big.r_individual.std(), reps.sd_R.std(), "not reported"),
            ("% of infected people who transmit", reps.pct_transmit.mean(), reps.pct_transmit.std(), "~25% (eLife); 26.7% (Fig 7B); ~30% (text)")]
    quoted = {5: ("~10%", "~35%", "~85%"), 10: ("~6%", "~25%", "~70%"), 20: ("~2.5%", "~10%", "~44%")}
    for k, (a, b, c) in quoted.items():
        rows.append((f"% of infected people with R0 >= {k}", reps[f"pct_all_ge{k}"].mean(), reps[f"pct_all_ge{k}"].std(), a))
        rows.append((f"% of transmitters with R0 >= {k}", reps[f"pct_transmitters_ge{k}"].mean(), reps[f"pct_transmitters_ge{k}"].std(), b))
        rows.append((f"% of infections from people with R0 >= {k}", reps[f"share_infections_ge{k}"].mean(), reps[f"share_infections_ge{k}"].std(), c))
    return pd.DataFrame(rows, columns=["quantity", "replication", "sd_across_runs_of_10000", "published"])


def pairwise_claims() -> pd.DataFrame:
    """The transmission probabilities the text quotes for one exposure contact."""
    rows = []
    for label, logv, et, ee, quoted in [
        ("no masks, V = 10^7", 7.0, 0.0, 0.0, "TD50 is 10^7.5 (eLife)"),
        ("no masks, V = 10^7.5", 7.5, 0.0, 0.0, "50%"),
        ("no masks, V = 10^8", 8.0, 0.0, 0.0, "75% (eLife)"),
        ("both masked at 0.9, V = 10^8.5", 8.5, 0.9, 0.9, "<5% for V < 10^8.5"),
        ("both masked at 0.9, V = 10^9", 9.0, 0.9, 0.9, "~20%"),
        ("both masked at 0.5, V = 10^8", 8.0, 0.5, 0.5, "Fig 2D plots 0.23"),
        ("both masked at 0.5, V = 10^10", 10.0, 0.5, 0.5, "Fig 2D plots 0.49"),
    ]:
        rows.append({"case": label, "stated_model": float(tr.transmission_risk(10.0**logv, et, ee)),
                     "published": quoted})
    return pd.DataFrame(rows)


def exposure_response(V) -> pd.DataFrame:
    """How the population's transmission scales with exposure viral load.

    ``transmission_ratio`` is expected R_e when every exposure viral load is
    multiplied by ``exposure_factor``, relative to no reduction.  The elasticity
    is log(ratio) / log(factor): 1 would mean transmission proportional to
    exposure viral load.
    """
    base = tr.expected_re(V, 0.0, 0.0, 0.0)
    rows = []
    for factor in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01):
        ratio = tr.expected_re(V, 1.0, 1.0 - factor, 0.0) / base
        rows.append({"exposure_factor": factor, "transmission_ratio": ratio,
                     "elasticity": np.log(ratio) / np.log(factor)})
    return pd.DataFrame(rows)


def grid(V, rng, coverages, targeted: str | None = None, threshold: float = 10.0) -> pd.DataFrame:
    """R_e and the distribution of individual R over the efficacy grid.

    ``targeted`` adds Supplementary figure 5's rule that people with more than
    ``threshold`` exposure contacts per day are always masked, read either as
    "day" (on the days they have that many) or as "person" (anyone who has
    such a day at any time in the 31 days simulated).
    """
    extra = {"day": {"always_masked_above": threshold},
             "person": {"always_masked_if_ever_above": threshold}}.get(targeted, {})
    rows = []
    for label in coverages:
        sigma, vartheta = COVERAGES[label]
        for eps_t in LEVELS:
            for eps_e in LEVELS:
                d = draw(V, rng, N_BIG, compliance=sigma, adherence=vartheta, eps_t=eps_t,
                         eps_e=eps_e, **extra)
                r = d.r_individual
                row = {"coverage": label, "eps_T": eps_t, "eps_E": eps_e,
                       "Re_mc": r.mean(), "Re_mc_unrounded": d.total.mean(),
                       "sd_individual": r.std(), "se_3000": r.std() / np.sqrt(N_PAPER),
                       "pct_transmit": 100 * (r >= 1).mean(), "pct_all_ge5": 100 * (r >= 5).mean(),
                       "pct_transmitters_ge5": 100 * (r >= 5).sum() / max((r >= 1).sum(), 1)}
                if targeted is None:
                    row["Re_exact"] = tr.expected_re(V, sigma * vartheta, eps_t, eps_e)
                    # The reading under which Supplementary figure 2 is drawn:
                    # the exposed contact's mask multiplies risk by (1 - eps_E)
                    # and only the transmitter's mask lowers the viral load.
                    c = sigma * vartheta
                    row["Re_exact_asymmetric"] = (
                        (1 - c) * tr.expected_re(V, c, eps_t, 0.0) + c * (1 - eps_e) * tr.expected_re(V, c, eps_t, 0.0))
                rows.append(row)
    return pd.DataFrame(rows)


def pair_decomposition(V, rng) -> pd.DataFrame:
    """Figure 4: R_e by masking status of the pair, equal efficacy both sides."""
    rows = []
    for label in ["25/25", "50/50", "75/75", "90/90"]:
        sigma, vartheta = COVERAGES[label]
        for eps in [0.0, 0.3, 0.6, 0.9]:
            exact = tr.expected_re_by_pair(V, sigma * vartheta, eps, eps)
            d = draw(V, rng, N_BIG, compliance=sigma, adherence=vartheta, eps_t=eps, eps_e=eps)
            sd = d.by_pair().std(axis=0)
            for name, value, s in zip(tr.PAIRS, exact, sd):
                rows.append({"coverage": label, "efficacy": eps, "pair": name,
                             "Re_exact": value, "se_3000": s / np.sqrt(N_PAPER)})
    return pd.DataFrame(rows)


def cluster_contributions(V, rng) -> pd.DataFrame:
    """Supplementary figure 4: R_e by the size of the transmitter's cluster."""
    rows = []
    for label in ["25/25", "50/50", "75/75", "90/90"]:
        sigma, vartheta = COVERAGES[label]
        for eps in [0.0, 0.3, 0.6, 0.9]:
            r = draw(V, rng, N_BIG, compliance=sigma, adherence=vartheta, eps_t=eps, eps_e=eps).r_individual
            for name, lo, hi in CLUSTER_BINS:
                rows.append({"coverage": label, "efficacy": eps, "cluster_size": name,
                             "Re_contribution": r[(r >= lo) & (r <= hi)].sum() / len(r)})
    return pd.DataFrame(rows)


def transmission_viral_loads(V, rng) -> pd.DataFrame:
    """Figure 5: viral load on the days a transmitter infects someone, 75/75.

    A transmission event is a day on which the transmitter's rounded running
    total of secondary infections rises, the eLife code's definition of when a
    new infection is added.  ``weight = infections`` counts each event once
    per infection added instead.
    """
    rows = []
    sigma, vartheta = COVERAGES["75/75"]
    for eps in [0.0, 0.3, 0.6, 0.9]:
        d = draw(V, rng, 3 * N_BIG, compliance=sigma, adherence=vartheta, eps_t=eps, eps_e=eps)
        running = np.round(np.cumsum(d.infections, axis=1))
        added = np.diff(np.concatenate([np.zeros((len(running), 1)), running], axis=1), axis=1)
        for j, name in enumerate(tr.PAIRS):
            mask = (added > 0) & (d.pair == j)
            events_per_run = mask.sum() / len(running) * N_PAPER
            for panel, values in (("transmitter", d.v), ("exposure", d.v_exposure)):
                x = np.log10(values[mask])
                order = np.argsort(x)
                cw = np.cumsum(added[mask][order])
                cw /= cw[-1]
                row = {"panel": panel, "efficacy": eps, "pair": name, "events_per_3000": events_per_run}
                for q, lab in ((0.25, "q1"), (0.5, "median"), (0.75, "q3")):
                    row[lab] = np.quantile(x, q)
                    row[f"{lab}_infection_weighted"] = x[order][np.searchsorted(cw, q)]
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    V = vl.untreated()
    rng = np.random.default_rng(20210604)

    base = baseline(V, rng)
    base.to_csv(RESULTS / "baseline.tsv", sep="\t", index=False, float_format="%.3f")
    print(base.to_string(index=False), "\n")

    response = exposure_response(V)
    response.to_csv(RESULTS / "exposure_response.tsv", sep="\t", index=False, float_format="%.4f")
    print(response.round(3).to_string(index=False), "\n")

    pw = pairwise_claims()
    pw.to_csv(RESULTS / "fig2_pairwise_claims.tsv", sep="\t", index=False, float_format="%.4f")
    print(pw.to_string(index=False), "\n")

    fig3 = grid(V, rng, ["25/25", "50/50", "75/75", "100/100"])
    paper3 = pd.read_csv(DATA / "paper_fig3_digitized.tsv", sep="\t").rename(columns={"mean_R0": "Re_paper"})
    fig3 = fig3.merge(paper3[["coverage", "eps_T", "eps_E", "Re_paper"]], on=["coverage", "eps_T", "eps_E"])
    # The paper averages individual R0 after rounding each to an integer,
    # which lowers the mean slightly.  Estimate that shift as a line in R_e
    # from the big runs, where the same draws are available both ways.
    slope, intercept = np.polyfit(fig3.Re_exact, fig3.Re_mc - fig3.Re_mc_unrounded, 1)
    for name in ("Re_exact", "Re_exact_asymmetric"):
        fig3[name.replace("exact", "expected")] = fig3[name] + intercept + slope * fig3[name]
    # A cell is read off the figure to about 0.02 (its palette has steps of 0.033).
    noise = np.sqrt(fig3.se_3000**2 + READING_ERROR**2)
    fig3["z"] = (fig3.Re_paper - fig3.Re_expected) / noise
    fig3["z_asymmetric"] = (fig3.Re_paper - fig3.Re_expected_asymmetric) / noise
    fig3.to_csv(RESULTS / "fig3_re_grid.tsv", sep="\t", index=False, float_format="%.4f")
    summary = fig3.groupby("coverage", sort=False).agg(
        cells=("z", "size"), mean_z=("z", "mean"), sd_z=("z", "std"),
        chi2_stated=("z", lambda z: (z**2).sum()),
        chi2_asymmetric=("z_asymmetric", lambda z: (z**2).sum())).reset_index()
    summary.to_csv(RESULTS / "fig3_fit_summary.tsv", sep="\t", index=False, float_format="%.3f")
    print(summary.round(2).to_string(index=False), "\n")

    fig4 = pair_decomposition(V, rng)
    paper4 = pd.read_csv(DATA / "paper_fig4_digitized.tsv", sep="\t").rename(columns={"R0_contribution": "Re_paper"})
    fig4 = fig4.merge(paper4, on=["coverage", "efficacy", "pair"])
    fig4.to_csv(RESULTS / "fig4_pairs.tsv", sep="\t", index=False, float_format="%.4f")

    fig5 = transmission_viral_loads(V, rng)
    paper5 = pd.read_csv(DATA / "paper_fig5_digitized.tsv", sep="\t")[["panel", "efficacy", "pair", "q1", "median", "q3"]]
    fig5 = fig5.merge(paper5, on=["panel", "efficacy", "pair"], suffixes=("", "_paper"))
    fig5.to_csv(RESULTS / "fig5_viral_load.tsv", sep="\t", index=False, float_format="%.3f")

    clusters = cluster_contributions(V, rng)
    clusters.to_csv(RESULTS / "suppfig4_clusters.tsv", sep="\t", index=False, float_format="%.4f")

    keep = ["coverage", "eps_T", "eps_E", "Re_mc", "se_3000"]
    by_day = grid(V, rng, ["25/25", "50/50", "75/75", "100/100"], targeted="day")[keep]
    by_person = grid(V, rng, ["25/25", "50/50", "75/75", "100/100"], targeted="person")[keep]
    paper_s5 = pd.read_csv(DATA / "paper_suppfig5_digitized.tsv", sep="\t").rename(columns={"mean_R0": "Re_paper"})
    targeted = (by_day.rename(columns={"Re_mc": "Re_masked_on_busy_days", "se_3000": "se_3000_busy_days"})
                .merge(by_person.rename(columns={"Re_mc": "Re_masked_if_ever_busy", "se_3000": "se_3000_ever_busy"}),
                       on=["coverage", "eps_T", "eps_E"])
                .merge(fig3[["coverage", "eps_T", "eps_E", "Re_mc"]].rename(columns={"Re_mc": "Re_untargeted"}),
                       on=["coverage", "eps_T", "eps_E"])
                .merge(paper_s5[["coverage", "eps_T", "eps_E", "Re_paper"]], on=["coverage", "eps_T", "eps_E"]))
    for name, se in (("busy_days", "se_3000_busy_days"), ("ever_busy", "se_3000_ever_busy")):
        col = "Re_masked_on_busy_days" if name == "busy_days" else "Re_masked_if_ever_busy"
        targeted[f"z_{name}"] = (targeted.Re_paper - targeted[col]) / np.sqrt(targeted[se]**2 + READING_ERROR**2)
    targeted.to_csv(RESULTS / "suppfig5_targeted.tsv", sep="\t", index=False, float_format="%.4f")
    print(targeted.groupby("coverage", sort=False).agg(
        paper=("Re_paper", "mean"), busy_days=("Re_masked_on_busy_days", "mean"),
        ever_busy=("Re_masked_if_ever_busy", "mean"),
        chi2_busy_days=("z_busy_days", lambda z: (z**2).sum()),
        chi2_ever_busy=("z_ever_busy", lambda z: (z**2).sum())).round(2).to_string())
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
