"""Drill-down: workers in Seattle in industries with on-site cafeterias /
public-health-influenced menus — hospitals (NAICS 62), education (NAICS 61),
public administration (NAICS 92).

For each target industry:
  - total workers at Seattle workplaces (from WAC)
  - demographic composition (age, sex, race, ethnicity, education, earnings)
  - estimated breakdown by where workers LIVE (from OD weighted by each Seattle
    block's industry share).

The home-origin estimate is an approximation: it assumes that within a single
census block, the home-location mix of industry-i workers matches the home-
location mix of all workers in that block. For hospital / school / agency
campuses this is reasonable because such blocks are typically dominated by
one employer.
"""

from pathlib import Path
import pandas as pd

import lib

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

TARGETS = {
    "Health Care & Social Assistance (CNS16 / NAICS 62)": "CNS16",
    "Educational Services (CNS15 / NAICS 61)":            "CNS15",
    "Public Administration (CNS20 / NAICS 92)":           "CNS20",
}

DEMO_COLS = (list(lib.AGE) + list(lib.EARN) + list(lib.SEX)
             + list(lib.RACE) + list(lib.ETH) + list(lib.EDU))


def industry_demographics(wac_sea: pd.DataFrame) -> pd.DataFrame:
    """One row per target industry: total + demographic shares."""
    rows = []
    all_workers = wac_sea["C000"].sum()
    for name, col in TARGETS.items():
        # Block-weighted demographics: rescale each block's demo counts by
        # the industry share at that block, then sum.
        share = (wac_sea[col] / wac_sea["C000"]).fillna(0)
        ind_total = wac_sea[col].sum()
        row = {"industry": name,
               "total_workers": int(ind_total),
               "share_of_all_seattle_workers_pct": round(ind_total / all_workers * 100, 1)}
        for d in DEMO_COLS:
            est = (wac_sea[d] * share).sum()
            row[d] = round(est / ind_total * 100, 1) if ind_total else 0.0
        rows.append(row)
    df = pd.DataFrame(rows).set_index("industry")
    # also include "All Seattle workers" baseline for comparison
    baseline = {"total_workers": int(all_workers),
                "share_of_all_seattle_workers_pct": 100.0}
    for d in DEMO_COLS:
        baseline[d] = round(wac_sea[d].sum() / all_workers * 100, 1)
    df.loc["All Seattle workers (baseline)"] = baseline
    return df


def home_origin_by_industry(wac_sea: pd.DataFrame, od: pd.DataFrame,
                            seattle_blocks: set[str]) -> pd.DataFrame:
    """For each target industry, estimate worker counts by home origin bucket.

    Block-weighted: for each Seattle work block b, scale OD inflows to b by
    f_b(i) = WAC[b, i] / WAC[b, C000]."""
    od_in = od[od["w_geocode"].isin(seattle_blocks)].copy()
    od_in["home"] = od_in["h_geocode"].apply(
        lambda g: lib.classify_block(g, seattle_blocks))

    # Map each Seattle work block to its industry share
    rows = []
    for name, col in TARGETS.items():
        share = (wac_sea.set_index("w_geocode")[col]
                 / wac_sea.set_index("w_geocode")["C000"]).fillna(0)
        od_in["ind_workers"] = od_in["S000"] * od_in["w_geocode"].map(share).fillna(0)
        grp = od_in.groupby("home")["ind_workers"].sum().round().astype(int)
        order = ["Seattle", "King Co. (outside Seattle)",
                 "WA (outside King Co.)", "Out of state"]
        grp = grp.reindex([o for o in order if o in grp.index]).fillna(0).astype(int)
        total = int(grp.sum())
        grp_pct = (grp / total * 100).round(1)
        for home in grp.index:
            rows.append({
                "industry": name,
                "home_origin": home,
                "workers_est": int(grp[home]),
                "share_pct": float(grp_pct[home]),
            })
        rows.append({
            "industry": name, "home_origin": "Outside Seattle",
            "workers_est": int(total - grp.get("Seattle", 0)),
            "share_pct": round((total - grp.get("Seattle", 0)) / total * 100, 1)})
        rows.append({
            "industry": name, "home_origin": "Outside King Co.",
            "workers_est": int(grp.get("WA (outside King Co.)", 0)
                               + grp.get("Out of state", 0)),
            "share_pct": round((grp.get("WA (outside King Co.)", 0)
                                + grp.get("Out of state", 0)) / total * 100, 1)})
        rows.append({"industry": name, "home_origin": "TOTAL",
                     "workers_est": total, "share_pct": 100.0})
    return pd.DataFrame(rows)


def demographics_by_industry_and_home(wac_sea: pd.DataFrame, od: pd.DataFrame,
                                      seattle_blocks: set[str]) -> pd.DataFrame:
    """For each (industry, home-origin), estimate demographic shares.

    Approach (block-weighted):
      worker_count[i, H, d] ≈ Σ_b  OD[h∈H, w=b, S000] · (WAC[b, i]/WAC[b, C000])
                                                       · (WAC[b, d]/WAC[b, C000])
    This separability assumption (independence of demographic d and home H
    within a block, conditional on industry) is approximate; it gives the
    direction and rough magnitude of differences."""
    od_in = od[od["w_geocode"].isin(seattle_blocks)].copy()
    od_in["home"] = od_in["h_geocode"].apply(
        lambda g: lib.classify_block(g, seattle_blocks))
    wac_idx = wac_sea.set_index("w_geocode")
    rows = []
    for name, ind_col in TARGETS.items():
        share_ind = (wac_idx[ind_col] / wac_idx["C000"]).fillna(0)
        for d in DEMO_COLS:
            share_d = (wac_idx[d] / wac_idx["C000"]).fillna(0)
            weight = (share_ind * share_d).to_dict()
            od_in["w"] = od_in["S000"] * od_in["w_geocode"].map(weight).fillna(0)
            grp = od_in.groupby("home")["w"].sum()
            # also baseline for the industry overall (no home filter):
            denom_per_home = (od_in.assign(w2=od_in["S000"] * od_in["w_geocode"].map(share_ind).fillna(0))
                              .groupby("home")["w2"].sum())
            for home in grp.index:
                if denom_per_home[home] > 0:
                    rows.append({
                        "industry": name,
                        "home_origin": home,
                        "demographic": d,
                        "label": ({**lib.AGE, **lib.EARN, **lib.SEX,
                                   **lib.RACE, **lib.ETH, **lib.EDU}).get(d, d),
                        "share_pct": round(grp[home] / denom_per_home[home] * 100, 1),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    print("loading geography + LODES...")
    seattle = lib.load_seattle_blocks()
    wac = lib.load_wac()
    od  = lib.load_od()
    wac_sea = wac[wac["w_geocode"].isin(seattle)].copy()
    print(f"  Seattle workplace blocks with jobs: {(wac_sea['C000']>0).sum():,}")

    print("\n--- industry demographics (Seattle workplaces) ---")
    demo = industry_demographics(wac_sea)
    demo.to_csv(OUT / "04_industry_demographics.csv")
    print(demo[["total_workers", "share_of_all_seattle_workers_pct"]].to_string())

    print("\n--- home origin of workers in each target industry ---")
    home = home_origin_by_industry(wac_sea, od, seattle)
    home.to_csv(OUT / "05_industry_home_origin.csv", index=False)
    pivot = home.pivot(index="industry", columns="home_origin", values="share_pct")
    print(pivot.to_string())

    print("\n--- demographics within each (industry × home origin) ---")
    fine = demographics_by_industry_and_home(wac_sea, od, seattle)
    fine.to_csv(OUT / "06_industry_by_home_demographics.csv", index=False)
    print(f"  wrote {len(fine):,} rows to 06_industry_by_home_demographics.csv")

    print(f"\nResults in {OUT}/")


if __name__ == "__main__":
    main()
