#!/usr/bin/env python3
"""Assemble every input, posterior and simulated output into a single DuckDB file.

    uv run python scripts/build_db.py

Produces results/tb_model_structures.duckdb with one table per dataset, each
carrying a `source` column where the loader provides one.  Re-runnable: the file
is rebuilt from scratch each time.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "results" / "tb_model_structures.duckdb"
GBD = ROOT / "data" / "gbd"
DERIVED = ROOT / "data" / "derived"
POST = DERIVED / "posteriors"


def main() -> int:
    import tbmodels.data as D
    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()
    con = duckdb.connect(str(DB))

    def put(name: str, df: pd.DataFrame):
        con.register("_tmp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _tmp")
        con.unregister("_tmp")
        print(f"  {name}: {len(df)} rows")

    # -- inputs ----------------------------------------------------------
    print("inputs:")
    put("input_converter_cohorts", D.load_converter_cohorts())
    put("input_ferebee_converter", D.load_ferebee_converter())
    put("input_spectrum_transitions", D.load_spectrum_transitions())
    put("input_tiemersma_survival", D.load_tiemersma_survival())
    put("input_prevalence_surveys", D.load_prevalence_surveys())
    put("input_duration_targets", D.load_duration_targets())
    put("input_menzies_review", D.load_menzies_review())
    put("input_sutherland", D.load_sutherland())

    # -- GBD / UNAIDS targets -------------------------------------------
    print("gbd:")
    for csv in sorted(GBD.glob("*.csv")):
        name = "gbd_" + csv.stem.replace("-", "_")
        put(name, pd.read_csv(csv))

    # -- posteriors (summaries + draws) ---------------------------------
    print("posteriors:")
    summ = []
    draws = []
    for d in sorted(POST.glob("*")):
        if not d.is_dir():
            continue
        s = pd.read_csv(d / "summary.csv")
        s.insert(0, "structure", d.name)
        summ.append(s)
        npz = d / "samples.npz"
        if npz.exists():
            arr = np.load(npz)
            dd = pd.DataFrame({k: arr[k] for k in arr.files})
            dd.insert(0, "draw", np.arange(len(dd)))
            dd.insert(0, "structure", d.name)
            draws.append(dd.melt(id_vars=["structure", "draw"], var_name="param", value_name="value"))
    if summ:
        put("posterior_summary", pd.concat(summ, ignore_index=True))
    if draws:
        put("posterior_draws", pd.concat(draws, ignore_index=True))

    # -- derived results -------------------------------------------------
    print("results:")
    for csv in [ROOT / "results" / "reproduction_spectrum.csv",
                DERIVED / "transmission_fits.csv",
                DERIVED / "act3_simulation.csv",
                DERIVED / "cohort_model_comparison.csv",
                DERIVED / "subclinical_holdout.csv"]:
        if csv.exists():
            put("result_" + csv.stem, pd.read_csv(csv))

    tables = con.execute("SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall()
    print(f"\n{DB} built with {len(tables)} tables:")
    for (t,) in tables:
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t} ({n})")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
