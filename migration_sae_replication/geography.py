"""Ukraine district (admin-2) geography used throughout the replication.

Loads data/ukraine_adm2.geojson (see prepare_data.py) and derives:
* centroid coordinates (km, ETRS89-LAEA) and the pairwise distance matrix,
* the spatial graph used for Sigma_r: Dupuis links districts whose centroids are
  within 2.5 h drive time, tuned to a median of ~5 neighbours with no isolates.
  Without a routing engine we use centroid distance with the threshold chosen by
  the same criterion (median degree 5, no isolated district),
* border contiguity (for the "neighbor" migration pattern),
* district sets: major urban centres, conflict-affected (frontline) districts,
  hotspot/coldspot anchors, and approximate oblast-level IDP destination shares.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np

DATA = Path(__file__).parent / "data" / "ukraine_adm2.geojson"

# raions containing the eight largest government-controlled cities (2022 pop >= 465k)
URBAN_PCODES = ["UA8000", "UA6312", "UA5110", "UA1202", "UA4606", "UA2306", "UA1206", "UA4806"]
# Donetsk, Luhansk and Kherson oblasts plus the frontline raions of Kharkiv and Zaporizhzhia oblasts
CONFLICT_OBLASTS = ["UA14", "UA44", "UA65"]
CONFLICT_RAIONS = ["Kupianskyi", "Iziumskyi", "Chuhuivskyi", "Melitopolskyi", "Berdianskyi",
                   "Polohivskyi", "Vasylivskyi"]
HOTSPOTS = ["Kharkivskyi", "Chernihivskyi", "Kropyvnytskyi"]
COLDSPOTS = ["Lvivskyi", "Zhytomyrskyi", "Odeskyi"]
# approximate share of IDPs hosted by oblast (rounded from IOM General Population
# Survey rounds, 2023); only used to weight "crisis arrivals" destinations
IDP_SHARE = {
    "UA12": 0.12, "UA63": 0.11, "UA80": 0.10, "UA32": 0.07, "UA23": 0.06, "UA51": 0.05,
    "UA53": 0.05, "UA46": 0.05, "UA14": 0.04, "UA05": 0.03, "UA71": 0.03, "UA18": 0.02,
    "UA59": 0.02, "UA68": 0.02, "UA35": 0.02, "UA48": 0.02, "UA26": 0.02, "UA21": 0.02,
    "UA74": 0.015, "UA61": 0.015, "UA73": 0.015, "UA56": 0.015, "UA07": 0.015, "UA65": 0.01,
    "UA44": 0.0,
}


@dataclass
class Geography:
    gdf: gpd.GeoDataFrame          # included districts only, in file order
    coords: np.ndarray             # (r, 2) centroid x, y in km
    dist: np.ndarray               # (r, r) centroid distances in km
    adjacency: np.ndarray          # (r, r) 0/1 spatial graph for Sigma_r
    contiguity: np.ndarray         # (r, r) 0/1 shared-border graph
    threshold_km: float
    urban: np.ndarray              # boolean (r,)
    conflict: np.ndarray           # boolean (r,)
    hot: np.ndarray                # boolean (r,)
    cold: np.ndarray               # boolean (r,)
    idp_weight: np.ndarray         # (r,) destination weights for crisis arrivals

    @property
    def r(self):
        return len(self.gdf)

    @property
    def pop(self):
        return self.gdf["pop"].to_numpy(dtype=float)


def distance_threshold(dist, target_median=5):
    """Smallest threshold giving median degree >= target and no isolated district."""
    cands = np.unique(np.round(dist[np.triu_indices_from(dist, 1)], 0))
    for t in cands:
        adj = (dist <= t) & (dist > 0)
        deg = adj.sum(1)
        if deg.min() >= 1 and np.median(deg) >= target_median:
            return float(t)
    raise RuntimeError("no threshold found")


def load_geography(path=DATA, included_only=True) -> Geography:
    gdf = gpd.read_file(path)
    if included_only:
        gdf = gdf[gdf["included"]].reset_index(drop=True)
    proj = gdf.to_crs("EPSG:3035")
    cent = proj.geometry.centroid
    coords = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()]) / 1000.0
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    thr = distance_threshold(dist)
    adjacency = ((dist <= thr) & (dist > 0)).astype(float)
    # border contiguity: polygons that intersect after a 500 m buffer (robust to simplification)
    buffered = proj.geometry.buffer(500)
    sidx = buffered.sindex
    contiguity = np.zeros((len(gdf), len(gdf)))
    for i, geom in enumerate(buffered):
        for j in sidx.query(geom, predicate="intersects"):
            if i != j:
                contiguity[i, j] = 1.0
    contiguity = np.maximum(contiguity, contiguity.T)
    pc = gdf["pcode"].to_numpy()
    name = gdf["name"].to_numpy()
    ob = gdf["oblast_pcode"].to_numpy()
    urban = np.isin(pc, URBAN_PCODES)
    conflict = np.isin(ob, CONFLICT_OBLASTS) | np.isin(name, CONFLICT_RAIONS)
    hot = np.isin(name, HOTSPOTS)
    cold = np.isin(name, COLDSPOTS)
    pop = gdf["pop"].to_numpy(dtype=float)
    share = np.array([IDP_SHARE.get(o, 0.0) for o in ob])
    idp_w = np.zeros(len(gdf))
    for o in np.unique(ob):
        m = (ob == o) & ~conflict
        if m.sum() and share[m].sum() > 0:
            idp_w[m] = share[m][0] * pop[m] / pop[m].sum()
    idp_w = idp_w / idp_w.sum()
    return Geography(gdf, coords, dist, adjacency, contiguity, thr, urban, conflict, hot, cold, idp_w)


if __name__ == "__main__":
    geo = load_geography()
    deg = geo.adjacency.sum(1)
    print(f"{geo.r} districts; distance threshold {geo.threshold_km:.0f} km; "
          f"degree median {np.median(deg):.0f}, min {deg.min():.0f}, max {deg.max():.0f}")
    cdeg = geo.contiguity.sum(1)
    print(f"contiguity degree median {np.median(cdeg):.0f}, min {cdeg.min():.0f}, max {cdeg.max():.0f}")
    print("urban:", geo.gdf.loc[geo.urban, "name"].tolist())
    print("conflict:", geo.conflict.sum(), geo.gdf.loc[geo.conflict, "name"].tolist())
    print("hot:", geo.gdf.loc[geo.hot, "name"].tolist(), "cold:", geo.gdf.loc[geo.cold, "name"].tolist())
    print("pop total", geo.pop.sum(), "min/median/max", geo.pop.min(), np.median(geo.pop), geo.pop.max())
