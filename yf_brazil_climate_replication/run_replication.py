"""Reproduce every quantitative result in Sadeghieh et al. (2021).

Writes TSV tables into results/ and a JSON bundle the interactive HTML
document reads. Every simulation runs through the camdl binary.
"""

from __future__ import annotations

import json
import pathlib
import tomllib

import yf_model as M

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# Paper Table 4, "Average Temperature" column.
SCENARIOS = [
    ("RCP4.5", "2011-2040 (short-term)", 25.9),
    ("RCP4.5", "2041-2070 (medium-term)", 26.5),
    ("RCP4.5", "2071-2100 (long-term)", 26.9),
    ("RCP8.5", "2011-2040 (short-term)", 26.0),
    ("RCP8.5", "2041-2070 (medium-term)", 27.0),
    ("RCP8.5", "2071-2100 (long-term)", 28.3),
]
BASELINE_T = 23.0

# What the paper itself reports, for side-by-side comparison.
PAPER_TABLE4 = {
    "observed":  dict(peak=214, cum=1259, deaths=428, pkwk=5, dur=15),
    "fitted":    dict(peak=185, cum=1254, deaths=426, pkwk=4, dur=21),
    ("RCP4.5", 25.9): dict(peak=167, cum=1213, deaths=412, pkwk=4, dur=22),
    ("RCP4.5", 26.5): dict(peak=163, cum=1126, deaths=382, pkwk=4, dur=20),
    ("RCP4.5", 26.9): dict(peak=161, cum=1072, deaths=364, pkwk=4, dur=20),
    ("RCP8.5", 26.0): dict(peak=167, cum=1201, deaths=408, pkwk=4, dur=21),
    ("RCP8.5", 27.0): dict(peak=159, cum=1049, deaths=356, pkwk=4, dur=19),
    ("RCP8.5", 28.3): dict(peak=147, cum=867,  deaths=294, pkwk=4, dur=17),
}
PAPER_TABLE6 = {10: (868, 5997, 2038, 4, 28), 20: (772, 5310, 1805, 4, 28),
                30: (675, 4629, 1573, 4, 27), 40: (579, 3955, 1344, 4, 26),
                50: (483, 3288, 1117, 4, 25), 60: (387, 2626, 892, 4, 24),
                70: (291, 1971, 670, 4, 23),  80: (184, 1254, 426, 4, 21),
                90: (99, 678, 230, 4, 20)}
PAPER_TABLE5 = {("mu", 25): (0.33, 202, 5, 23), ("mu", 26): (0.36, 190, 4, 22),
                ("delta", 25): (0.41, 184, 4, 21), ("delta", 26): (0.45, 184, 4, 21),
                ("lambda2", 25): (0.26, 158, 4, 21), ("lambda2", 26): (0.29, 162, 4, 21)}


def load_mle(fit_glob: str) -> dict[str, float]:
    # camdl keys each fit directory by a hash of its inputs, so re-running with
    # changed bounds leaves older directories in place. Sort by mtime, not by
    # name: the hashes do not order chronologically.
    hits = sorted(ROOT.glob(f"results/fits/{fit_glob}/*/*/mle_params.toml"),
                  key=lambda f: f.stat().st_mtime)
    if not hits:
        raise SystemExit(f"no MLE found for {fit_glob} — run `camdl fit run` first")
    with open(hits[-1], "rb") as fh:
        raw = tomllib.load(fh)
    # Top-level scalars are the parameters; [provenance] is a nested table
    # and carries keys (chain, log_likelihood, ...) that are not parameters.
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def row(series, label):
    o = M.outcomes(series)
    return dict(label=label, peak=o["peak_incidence"], pkwk=o["peak_week"],
                cum=o["cumulative_incidence"], deaths=o["deaths"], dur=o["duration"])


def pct(new, old):
    return 100.0 * (new - old) / old if old else float("nan")


def main() -> None:
    bundle: dict[str, object] = {}
    obs = M.observed()
    bundle["observed"] = obs
    bundle["paper"] = {
        "table4": {str(k): v for k, v in PAPER_TABLE4.items()},
        "table5": {f"{k[0]}@{k[1]}": v for k, v in PAPER_TABLE5.items()},
        "table6": PAPER_TABLE6,
    }

    # ---- 1. the published parameter set, run as published -----------------
    pub = dict(M.PUBLISHED)
    s_pub = M.simulate(pub)
    bundle["published_run"] = {"series": s_pub, **row(s_pub, "published parameters")}

    # ---- 2. the two refits -------------------------------------------------
    mle_t1 = load_mle("fit_yf_mle_pinned-*")      # Table 1 mosquito parameters
    mle_t23 = load_mle("fit_yf_mle_t23-*")        # Table 3 polynomials at 23 C
    bundle["mle_table1"] = mle_t1
    bundle["mle_t23"] = mle_t23

    s_t1 = M.simulate(mle_t1)
    s_t23 = M.simulate(mle_t23)
    bundle["fit_table1"] = {"series": s_t1, **row(s_t1, "refit, Table 1 mosquito params")}
    bundle["fit_t23"] = {"series": s_t23, **row(s_t23, "refit, Table 3 polys at 23 C")}

    # ---- 3. Table 4: climate scenarios ------------------------------------
    # Variant A reproduces the paper's own procedure: a baseline that uses the
    # Table 1 mosquito parameters, compared against scenarios that use the
    # Table 3 polynomials. Variant B keeps one parameterisation throughout.
    table4 = []
    for variant, base_mle, base_T in (("A_paper_procedure", mle_t1, None),
                                      ("B_consistent", mle_t23, BASELINE_T)):
        base_params = dict(base_mle)
        base_series = M.simulate(base_params)
        b = row(base_series, "baseline")
        table4.append(dict(variant=variant, rcp="baseline", period="2017/18 fitted",
                           temp=BASELINE_T, **{k: v for k, v in b.items() if k != "label"},
                           d_peak=0.0, d_cum=0.0, d_dur=0.0, series=base_series["cases"][:30]))
        for rcp, period, temp in SCENARIOS:
            p = dict(base_params)
            p.update(M.temperature_params(temp))
            s = M.simulate(p)
            r = row(s, f"{rcp} {period}")
            table4.append(dict(
                variant=variant, rcp=rcp, period=period, temp=temp,
                **{k: v for k, v in r.items() if k != "label"},
                d_peak=pct(r["peak"], b["peak"]), d_cum=pct(r["cum"], b["cum"]),
                d_dur=pct(r["dur"], b["dur"]), series=s["cases"][:30]))
    bundle["table4"] = table4

    # ---- 4. Table 5: one-at-a-time temperature sensitivity ----------------
    table5 = []
    for name in ("mu", "delta", "lambda2"):
        for temp in (25, 26):
            p = dict(mle_t1)
            p[name] = M.temperature_params(temp)[name]
            s = M.simulate(p)
            r = row(s, f"{name}@{temp}")
            table5.append(dict(param=name, temp=temp, value=p[name],
                               **{k: v for k, v in r.items() if k != "label"}))
    bundle["table5"] = table5

    # ---- 5. Table 6: immunity / vaccination sweep -------------------------
    table6 = []
    for pct_im in range(10, 100, 10):
        p = dict(mle_t1)
        p["immune"] = pct_im / 100.0
        s = M.simulate(p)
        r = row(s, f"{pct_im}% immune")
        table6.append(dict(immune=pct_im, **{k: v for k, v in r.items() if k != "label"},
                           series=s["cases"][:30]))
    bundle["table6"] = table6

    # ---- 5b. immunity sweep under the paper's own (near-zero) feedback ----
    # The paper's Table 6 is almost exactly linear in the susceptible fraction,
    # which only happens when R0 ~ 0. Re-running the sweep with the PUBLISHED
    # beta2 and a beta1 rescaled to match the reported case total reproduces
    # that linearity, and pins down which published value is inconsistent.
    recon = dict(M.PUBLISHED)
    recon["omega"] = M.OMEGA_CFR
    recon["beta1"] = M.PUBLISHED["beta1"] * 1254.0 / 30174.6
    bundle["reconstruction"] = {"params": recon, "r0": M.r0(recon),
                                **row(M.simulate(recon), "beta1 rescaled, else published")}
    table6_lowfeedback = []
    for pct_im in range(10, 100, 10):
        p = dict(recon)
        p["immune"] = pct_im / 100.0
        s = M.simulate(p)
        r = row(s, f"{pct_im}% immune")
        table6_lowfeedback.append(dict(immune=pct_im,
                                       **{k: v for k, v in r.items() if k != "label"}))
    bundle["table6_lowfeedback"] = table6_lowfeedback

    # ---- 5c. R0 under each parameterisation -------------------------------
    bundle["r0"] = {
        "published": M.r0(M.PUBLISHED),
        "reconstruction": M.r0(recon),
        "refit_table1": M.r0(mle_t1),
        "refit_t23": M.r0(mle_t23),
    }

    # ---- 6. identifiability: the beta1 ridge ------------------------------
    ridge = []
    # c is capped at 10: camdl rounds `count`-kind initial values to whole
    # numbers, so a y0 below 1 no longer scales and the invariance breaks
    # for numerical, not structural, reasons.
    for c in (0.01, 0.1, 1.0, 10.0):
        p = dict(mle_t1)
        p["beta1"] = mle_t1["beta1"] * c
        p["beta2"] = mle_t1["beta2"] / c
        p["y0"] = mle_t1["y0"] / c
        p["z0"] = mle_t1["z0"] / c
        s = M.simulate(p)
        r = row(s, f"beta1 x {c}")
        ridge.append(dict(scale=c, beta1=p["beta1"], beta2=p["beta2"], y0=p["y0"],
                          **{k: v for k, v in r.items() if k != "label"}))
    bundle["ridge"] = ridge

    # ---- write ------------------------------------------------------------
    (OUT / "replication_bundle.json").write_text(json.dumps(bundle, indent=1))

    def write_tsv(name, rows, cols):
        with open(OUT / name, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(
                    f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])
                    for c in cols) + "\n")

    write_tsv("table4_climate_scenarios.tsv", table4,
              ["variant", "rcp", "period", "temp", "peak", "pkwk", "cum", "deaths",
               "dur", "d_peak", "d_cum", "d_dur"])
    write_tsv("table5_sensitivity.tsv", table5,
              ["param", "temp", "value", "peak", "pkwk", "dur", "cum", "deaths"])
    write_tsv("table6_immunity.tsv", table6,
              ["immune", "peak", "cum", "deaths", "pkwk", "dur"])
    write_tsv("table6_immunity_lowfeedback.tsv", table6_lowfeedback,
              ["immune", "peak", "cum", "deaths", "pkwk", "dur"])
    write_tsv("ridge_identifiability.tsv", ridge,
              ["scale", "beta1", "beta2", "y0", "peak", "pkwk", "cum", "deaths", "dur"])

    print("=== published parameters, run as published ===")
    print(" ", bundle["published_run"])
    print("=== refit (Table 1 mosquito params) ===", mle_t1)
    print(" ", {k: v for k, v in bundle["fit_table1"].items() if k != "series"})
    print("=== refit (Table 3 polys @23C) ===", mle_t23)
    print(" ", {k: v for k, v in bundle["fit_t23"].items() if k != "series"})
    print("\nwrote", OUT / "replication_bundle.json")


if __name__ == "__main__":
    main()
