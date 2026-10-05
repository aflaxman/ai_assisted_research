"""One-off conversion of public inputs into the slim files committed in data/.

Inputs (downloaded manually, not committed because of size):
  <raw>/cod_ab/ukr_admin2.geojson, <raw>/cod_ab/ukr_admin4.geojson
      HDX "Ukraine - Subnational Administrative Boundaries" (COD-AB v05, 2025-09-01)
      https://data.humdata.org/dataset/cod-ab-ukr
  <raw>/ukr_admpop_2022.xlsx
      HDX "Ukraine: Subnational Population Statistics" 2022 workbook (oblast totals
      and the 30 cities with >= 100k residents, each tagged with its raion p-code)
      https://data.humdata.org/dataset/cod-ps-ukr

Output: data/ukraine_adm2.geojson with one row per admin-2 unit (raion or city
with special status) and a population proxy.  Raion populations are not in the
public workbook, so each oblast total is split as: the 100k+ cities keep their
own population, and the remainder is shared among the oblast's raions in
proportion to a settlement score (village = 1, urban-type settlement = 8,
city = 30, with the 100k+ cities removed from the score).  Crimea and
Sevastopol are flagged included = False (no registry data since 2014).

Usage: uv run python prepare_data.py <raw_dir>
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

raw = Path(sys.argv[1])
out = Path(__file__).parent / "data" / "ukraine_adm2.geojson"

g = gpd.read_file(raw / "cod_ab" / "ukr_admin2.geojson")
g = g[["adm2_pcode", "adm2_name", "adm1_pcode", "adm1_name", "area_sqkm", "geometry"]].rename(
    columns={"adm2_pcode": "pcode", "adm2_name": "name", "adm1_pcode": "oblast_pcode", "adm1_name": "oblast"})
g["geometry"] = g.geometry.simplify(0.01, preserve_topology=True)

xlsx = raw / "ukr_admpop_2022.xlsx"
adm1 = pd.read_excel(xlsx, sheet_name="ukr_admpop_adm1_2022")[["ADM1_PCODE", "T_TL"]]
adm1 = adm1.set_index("ADM1_PCODE")["T_TL"]
cities = pd.read_excel(xlsx, sheet_name="ukr_admpop_100k_cities_2022")[["admin1Pcode", "admin2Pcode", "T_TL"]]
city_pop = cities.groupby("admin2Pcode")["T_TL"].sum()
n_city = cities.groupby("admin2Pcode").size()

a4 = gpd.read_file(raw / "cod_ab" / "ukr_admin4.geojson", columns=["adm4_type", "adm2_pcode"], read_geometry=False)
a4["wt"] = a4["adm4_type"].map({"Village": 1.0, "Settlement": 8.0, "City": 30.0}).fillna(1.0)
score = a4.groupby("adm2_pcode")["wt"].sum()

g["city_pop"] = g["pcode"].map(city_pop).fillna(0.0)
g["score"] = g["pcode"].map(score).fillna(1.0) - 30.0 * g["pcode"].map(n_city).fillna(0)
g["score"] = g["score"].clip(lower=1.0)
pop = np.zeros(len(g))
for ob, idx in g.groupby("oblast_pcode").groups.items():
    rows = g.loc[idx]
    total = float(adm1.get(ob, np.nan))
    if len(rows) == 1:                       # Kyiv city, Sevastopol
        pop[g.index.get_indexer(idx)] = total
        continue
    remainder = max(total - rows["city_pop"].sum(), 0.0)
    share = rows["score"] / rows["score"].sum()
    pop[g.index.get_indexer(idx)] = rows["city_pop"].values + remainder * share.values
g["pop"] = np.round(pop).astype(int)
g["included"] = ~g["oblast_pcode"].isin(["UA01", "UA85"])
g = g.drop(columns=["score"]).sort_values("pcode").reset_index(drop=True)
g.to_file(out, driver="GeoJSON")
print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB), {len(g)} units, {g.included.sum()} included")
print("total population proxy (included):", int(g.loc[g.included, "pop"].sum()))
print(g.sort_values("pop", ascending=False)[["name", "oblast", "pop", "city_pop"]].head(12).to_string())
