#!/usr/bin/env python3
"""Synthesize the comparison: cross-family fit, duration, identifiability, the
subclinical-fraction hold-out, and the ACT3 screening result.

    uv run python scripts/analyze.py

Writes to data/derived/ and results/figures/:
  cohort_model_comparison.csv   WAIC on the shared infectious-converter curve,
                                derived infectious-disease duration, identifiability
  subclinical_holdout.csv       predicted subclinical fraction vs survey values
Figures: cumulative-incidence fits, prevalence predictions, ACT3 decline,
subclinical fraction.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
POST = ROOT / "data" / "derived" / "posteriors"
DERIVED = ROOT / "data" / "derived"
FIG = ROOT / "results" / "figures"

# Viet Nam prevalence-survey subclinical fraction (epidemery data), as a hold-out
# Prevalence-survey subclinical-fraction hold-out targets.  Viet Nam from the
# epidemery per-survey counts; South Africa from the 2017-19 national survey
# (CXR-only share, Moyo 2022); Kenya/India have no reachable country-specific
# value, so the Frascella 2021 pooled estimate (0.50, IQR 0.40-0.62) is the
# comparison.  See data/derived/survey_subclinical_targets.csv.
SURVEY_SUBFRAC = {"Viet Nam": (0.66, 0.70), "South Africa": (0.577, 0.577),
                  "India": (0.40, 0.62), "Kenya": (0.40, 0.62), "Netherlands": (0.40, 0.62)}


def shared_infectious_cohorts():
    """The infectious-band converter observations both families predict: NTI
    waves + Ferebee, as (year, events, at_risk) intervals."""
    import pandas as pd
    from tbmodels.data import load_converter_cohorts, load_ferebee_converter
    from tbmodels.likelihood import prep_converter
    df = pd.concat([load_converter_cohorts(), load_ferebee_converter()], ignore_index=True)
    return [c for c in prep_converter(df) if c["band"] == "infectious"]


def waic_shared(structure, samples, cohorts, max_draws=400):
    """WAIC on the shared infectious-converter curve (comparable across families).

    Vectorised over draws with jax.vmap + jit so the whole posterior sample is
    evaluated in one compiled pass (a per-draw Python loop is far too slow)."""
    import jax, jax.numpy as jnp
    from tbmodels import jaxcore as J
    from tbmodels.likelihood import converter_target
    names = list(structure.param_names)
    nd = len(np.asarray(samples[names[0]]))
    sel = np.linspace(0, nd - 1, min(max_draws, nd)).astype(int)
    stacked = jnp.asarray(np.stack([np.asarray(samples[k])[sel] for k in names], axis=1))  # (D, P)
    tgt = converter_target(structure, "infectious")
    isKL = structure.name in ("K", "L")

    def one_draw(theta):
        p = {names[i]: theta[i] for i in range(len(names))}
        row = []
        for co in cohorts:
            tgrid = jnp.asarray(np.concatenate([[0.0], co["year"]]))
            ci = J.jax_first_passage_KL(structure.name, p, tgrid) if isKL \
                else J.jax_first_passage(structure, p, tgrid, tgt)
            surv = 1.0 - ci
            q = jnp.clip((surv[:-1] - surv[1:]) / jnp.clip(surv[:-1], 1e-9, 1.0), 1e-9, 1 - 1e-9)
            ev, n = jnp.asarray(co["events"]), jnp.asarray(co["at_risk"])
            row.append(ev * jnp.log(q) + (n - ev) * jnp.log1p(-q))
        return jnp.concatenate(row)

    ll = np.asarray(jax.jit(jax.vmap(one_draw))(stacked))  # (D, obs)
    lppd = np.sum(np.log(np.mean(np.exp(ll - ll.max(0)), 0)) + ll.max(0))
    p_waic = np.sum(np.var(ll, axis=0, ddof=1))
    waic = -2 * (lppd - p_waic)
    return waic, p_waic, ll.shape[1]


def derived_duration(structure, samples, max_draws=400):
    """Infectious-disease duration (years) over the posterior (vmapped)."""
    import jax, jax.numpy as jnp
    from tbmodels.likelihood import duration_infectious
    names = list(structure.param_names)
    nd = len(np.asarray(samples[names[0]]))
    sel = np.linspace(0, nd - 1, min(max_draws, nd)).astype(int)
    stacked = jnp.asarray(np.stack([np.asarray(samples[k])[sel] for k in names], axis=1))

    def one(theta):
        return duration_infectious(structure, {names[i]: theta[i] for i in range(len(names))})
    try:
        vals = np.asarray(jax.jit(jax.vmap(one))(stacked))
        vals = vals[np.isfinite(vals)]
    except Exception:  # noqa: BLE001
        vals = np.array([])
    return np.percentile(vals, [50, 2.5, 97.5]) if len(vals) else (np.nan, np.nan, np.nan)


def main():
    from tbmodels.registry import all_structures
    structs = all_structures()
    cohorts = shared_infectious_cohorts()

    comp_rows = []
    for name, s in structs.items():
        f = POST / name / "samples.npz"
        if not f.exists():
            continue
        samples = {k: v for k, v in np.load(f).items()}
        # identifiability condition number + count of well-constrained parameters
        cond = np.nan; well = 0
        idf = POST / name / "identifiability.csv"
        if idf.exists():
            lines = idf.read_text().splitlines()
            cond = next((float(l.split(",")[1]) for l in lines if l.startswith("# condition_number")), np.nan)
            body = pd.read_csv(idf, comment="#")
            well = int((body["contraction"] > 0.3).sum())
        waic, pw, nobs = waic_shared(s, samples, cohorts)
        dmed, dlo, dhi = derived_duration(s, samples)
        comp_rows.append(dict(
            structure=name, family=s.family, n_params=len(s.param_names),
            waic_shared_converters=round(waic, 2), p_waic=round(pw, 2), n_obs=nobs,
            duration_infectious_med=round(dmed, 2), duration_lo=round(dlo, 2), duration_hi=round(dhi, 2),
            condition_number=round(cond, 1), well_constrained_params=well,
        ))
        print(f"{name:16s} WAIC={waic:7.1f} dur={dmed:.2f}y cond={cond:.0f} well={well}/{len(s.param_names)}")

    if comp_rows:
        comp_rows.sort(key=lambda r: r["waic_shared_converters"])
        with open(DERIVED / "cohort_model_comparison.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(comp_rows[0]))
            w.writeheader(); w.writerows(comp_rows)
        print(f"wrote {DERIVED / 'cohort_model_comparison.csv'}")

    # -- subclinical hold-out -------------------------------------------
    fits = DERIVED / "transmission_fits.csv"
    if fits.exists():
        sub_rows = []
        with open(fits) as fh:
            for r in csv.DictReader(fh):
                survey = SURVEY_SUBFRAC.get(r["country"])
                sub_rows.append(dict(
                    structure=r["structure"], family=r["family"], country=r["country"],
                    model_subclinical_fraction=r["subclinical_fraction"] or "n/a (single disease state)",
                    survey_subclinical_fraction=f"{survey[0]}-{survey[1]}" if survey else "",
                ))
        with open(DERIVED / "subclinical_holdout.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sub_rows[0]))
            w.writeheader(); w.writerows(sub_rows)
        print(f"wrote {DERIVED / 'subclinical_holdout.csv'}")

    make_figures(structs, cohorts)
    return 0


def make_figures(structs, cohorts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)

    # Figure: cumulative incidence to infectious disease vs Ferebee, median fits
    from tbmodels.data import load_ferebee
    fer = load_ferebee()
    years = fer["Year"].to_numpy()
    cum_obs = np.cumsum(fer["Calc_Count_5p"].to_numpy()) / fer["At_Risk_5p"].to_numpy()[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(years, cum_obs, "ko", label="Ferebee 1970 (observed)", zorder=5)
    tgrid = np.linspace(0, 10, 41)
    for name, s in structs.items():
        f = POST / name / "samples.npz"
        if not f.exists():
            continue
        med = {k: float(np.median(v)) for k, v in np.load(f).items()}
        try:
            ci = s.cumulative_incidence(med, tgrid,
                                        target=tuple(c.name for c in s.compartments
                                                     if getattr(c, "bact_positive", False)) or ("I",))
        except Exception:  # noqa: BLE001
            continue
        ls = "-" if s.family == "spectrum" else "--"
        ax.plot(tgrid, ci, ls, lw=1, alpha=0.7, label=name)
    ax.set_xlabel("years since infection"); ax.set_ylabel("cumulative incidence of infectious TB")
    ax.set_title("Progression from infection to infectious disease")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(FIG / "cumulative_incidence.png", dpi=130); plt.close(fig)
    print(f"wrote {FIG / 'cumulative_incidence.png'}")

    # Figure: ACT3 decline
    act3 = DERIVED / "act3_simulation.csv"
    if act3.exists():
        import pandas as pd
        a = pd.read_csv(act3)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        obs = a[a["round"].isin(["1", "2", "3", "4"])].drop_duplicates("round")
        ax.plot([1, 2, 3, 4], [389, 308, 176, 126], "ko-", label="ACT3 observed", zorder=5)
        for name, g in a[a["round"].isin(["1", "2", "3", "4"])].groupby("structure"):
            g = g.sort_values("round")
            fam = g["family"].iloc[0]
            ax.plot(g["round"].astype(int), g["model_prevalence"],
                    "-" if fam == "spectrum" else "--", lw=1, alpha=0.7, label=name)
        ax.set_xlabel("screening round (year)"); ax.set_ylabel("Xpert+ prevalence /100k")
        ax.set_title("ACT3: predicted vs observed prevalence decline (Viet Nam)")
        ax.legend(fontsize=6, ncol=2); fig.tight_layout()
        fig.savefig(FIG / "act3_decline.png", dpi=130); plt.close(fig)
        print(f"wrote {FIG / 'act3_decline.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
