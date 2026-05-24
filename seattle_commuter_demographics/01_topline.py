"""Top-level counts and demographics for three groups:

  A. Residents of Seattle (workers living in Seattle, from RAC)
  B. Workers in Seattle (jobs located in Seattle, from WAC)
  C. Workers in Seattle who LIVE outside Seattle / outside King County (from OD)

Outputs CSVs to results/.
"""

from pathlib import Path
import pandas as pd

import lib

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def summarize(df: pd.DataFrame, label_maps: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Roll up a counts-by-block frame into total + each label map's totals."""
    cols_to_sum = ["C000"] + [c for m in label_maps.values() for c in m]
    totals = df[cols_to_sum].sum().to_frame(name="workers").astype(int)
    totals["share_pct"] = (totals["workers"] / totals.loc["C000", "workers"] * 100).round(1)
    # add readable labels
    label = {}
    for m in label_maps.values():
        label.update(m)
    label["C000"] = "All workers"
    totals.insert(0, "category", [label.get(i, i) for i in totals.index])
    return totals


def od_breakdown(od: pd.DataFrame, seattle_blocks: set[str]) -> pd.DataFrame:
    """For workers WORKING in Seattle, break down by home origin (Seattle / King Co / WA / OOS)
    and by LODES segments available in OD (age, earnings, super-sector industry)."""
    work_in_sea = od[od["w_geocode"].isin(seattle_blocks)].copy()
    work_in_sea["home"] = work_in_sea["h_geocode"].apply(
        lambda g: lib.classify_block(g, seattle_blocks))

    seg_cols = ["S000", "SA01", "SA02", "SA03",
                "SE01", "SE02", "SE03",
                "SI01", "SI02", "SI03"]
    grp = work_in_sea.groupby("home")[seg_cols].sum().astype(int)
    # row order
    order = ["Seattle", "King Co. (outside Seattle)",
             "WA (outside King Co.)", "Out of state"]
    grp = grp.reindex([o for o in order if o in grp.index])
    grp.loc["Total"] = grp.sum()
    grp.loc["Workers from outside Seattle"] = grp.loc[
        [r for r in grp.index if r not in ("Seattle", "Total", "Workers from outside Seattle")]
    ].sum()
    grp.loc["Workers from outside King Co."] = grp.loc[
        [r for r in grp.index if r in ("WA (outside King Co.)", "Out of state")]
    ].sum()
    grp = grp.rename(columns={
        "S000": "total", "SA01": "age≤29", "SA02": "age30–54", "SA03": "age≥55",
        "SE01": "earn≤$1.25k", "SE02": "earn$1.25–3.33k", "SE03": "earn>$3.33k",
        "SI01": "Goods producing", "SI02": "Trade/Trans/Util", "SI03": "All other services",
    })
    return grp


def main() -> None:
    print("loading geography...")
    seattle = lib.load_seattle_blocks()
    print(f"  Seattle has {len(seattle):,} census blocks")

    print("loading LODES files...")
    wac = lib.load_wac()
    rac = lib.load_rac()
    od  = lib.load_od()
    print(f"  WAC rows: {len(wac):,}   RAC rows: {len(rac):,}   OD rows: {len(od):,}")

    # ---- A. Seattle residents (workers living in Seattle) ----
    rac_sea = rac[rac["h_geocode"].isin(seattle)]
    rac_kingco = rac[rac["h_geocode"].str.startswith(lib.KING_COUNTY_FIPS)]
    # ---- B. Workers in Seattle (jobs located in Seattle) ----
    wac_sea = wac[wac["w_geocode"].isin(seattle)]
    wac_kingco = wac[wac["w_geocode"].str.startswith(lib.KING_COUNTY_FIPS)]

    label_maps = {"Age": lib.AGE, "Earnings": lib.EARN, "Sex": lib.SEX,
                  "Race": lib.RACE, "Ethnicity": lib.ETH, "Education": lib.EDU,
                  "Industry": lib.NAICS}

    sea_res = summarize(rac_sea, label_maps);  sea_res.to_csv(OUT / "01_seattle_residents.csv")
    sea_wrk = summarize(wac_sea, label_maps);  sea_wrk.to_csv(OUT / "02_seattle_workers.csv")
    kc_res  = summarize(rac_kingco, label_maps); kc_res.to_csv(OUT / "01b_kingco_residents.csv")
    kc_wrk  = summarize(wac_kingco, label_maps); kc_wrk.to_csv(OUT / "02b_kingco_workers.csv")

    # ---- C. Where Seattle workers live (from OD) ----
    od_sea = od_breakdown(od, seattle)
    od_sea.to_csv(OUT / "03_seattle_workers_by_home.csv")

    # Combined topline table for the README
    head = pd.DataFrame({
        "Seattle residents (workers living in Seattle)": sea_res["workers"],
        "Workers in Seattle (jobs located in Seattle)":   sea_wrk["workers"],
        "King Co. residents (workers)":                    kc_res["workers"],
        "Workers in King Co.":                              kc_wrk["workers"],
    })
    head.insert(0, "category", sea_res["category"])
    head.to_csv(OUT / "00_topline_counts.csv")

    print("\n=== TOP LINE ===")
    print(head.loc[["C000"]].to_string())
    print("\n=== WORKERS IN SEATTLE BY HOME LOCATION ===")
    print(od_sea[["total"]].to_string())
    total = od_sea.loc["Total", "total"]
    in_sea = od_sea.loc["Seattle", "total"]
    out_sea = od_sea.loc["Workers from outside Seattle", "total"]
    out_kc  = od_sea.loc["Workers from outside King Co.", "total"]
    print(f"\nShare of Seattle workers who live OUTSIDE Seattle: {out_sea/total*100:.1f}%")
    print(f"Share of Seattle workers who live OUTSIDE King Co.: {out_kc/total*100:.1f}%")
    print(f"\nResults written to {OUT}/")


if __name__ == "__main__":
    main()
