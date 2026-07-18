"""M2 — extract one OlmoEarth embedding per admin unit.

Pipeline shape (analog of the funded project):
  admin polygons (GADM L2, Malawi) -> low-cloud Sentinel-2 chip per unit
  -> OlmoEarth embedding (mean-pooled) -> one row per unit.

The "region" for the held-out-region test is the parent district (GADM NAME_1),
mirroring Admin-2 units grouped under a higher admin unit / held-out country.

Also records a stand-in target aggregated to the same units:
  * ndvi_mean  — mean NDVI from the SAME Sentinel-2 chip (zero extra download).
CAVEAT: NDVI (and anything here) is a PROXY to prove the pipeline runs and that
embeddings carry spatial signal. It says NOTHING about micronutrient deficiency.

Usage:
  uv run python extract_embeddings.py [--limit N] [--variant BASE|TINY|NANO]
Requires proxy env: CURL_CA_BUNDLE, GDAL_HTTP_PROXY (see README).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds
from shapely.geometry import Point, shape

from olmoearth_pretrain_minimal import ModelID
from olmoe import embed_sample, load_encoder, make_s2_sample

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
# 2021 dry season: low cloud AND pre-baseline-04.00 (avoids the -1000 L2A offset).
DATE_RANGE = "2021-05-01/2021-09-30"
TARGET_DISTRICTS = {"Blantyre", "Zomba", "Chiradzulu"}  # compact adjacent southern cluster
# drop non-land polygons (water / protected areas have no useful surface signal)
DROP_TOKENS = ("Lake", "NationalPark", "GameReserve", "ForestReserve", "N.P", "Marsh")

# model S2 band order -> Earth Search v1 asset key
ASSET_FOR_BAND = {
    "B02": "blue", "B03": "green", "B04": "red", "B08": "nir",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3", "B8A": "nir08",
    "B11": "swir16", "B12": "swir22", "B01": "coastal", "B09": "nir09",
}
S2_BAND_ORDER = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
B04_IDX, B08_IDX = S2_BAND_ORDER.index("B04"), S2_BAND_ORDER.index("B08")


def load_units(path="data/gadm41_MWI_2.json"):
    fc = json.load(open(path))
    units = []
    for f in fc["features"]:
        p = f["properties"]
        if p.get("NAME_1") not in TARGET_DISTRICTS:
            continue
        name2 = p.get("NAME_2") or ""
        if any(tok.lower() in name2.lower() for tok in DROP_TOKENS):
            continue
        geom = shape(f["geometry"])
        units.append({
            "admin_id": p["GID_2"],
            "name": name2,
            "region": p["NAME_1"],  # parent district = held-out "region"
            "geom": geom,
            "lon": geom.centroid.x,
            "lat": geom.centroid.y,
        })
    return units


def sample_points(geom, k, seed):
    """Centroid plus up to k-1 random interior points."""
    pts = [(geom.centroid.x, geom.centroid.y)]
    if k <= 1:
        return pts
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = geom.bounds
    tries = 0
    while len(pts) < k and tries < 500:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if geom.contains(Point(x, y)):
            pts.append((x, y))
        tries += 1
    return pts


def get_items(bbox, max_cloud=10, max_items=12):
    cat = Client.open(STAC_URL)
    s = cat.search(
        collections=[COLLECTION], bbox=bbox, datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=max_items,
    )
    items = list(s.items())
    return [{"item": it, "foot": shape(it.geometry), "ds": {}} for it in items]


def open_band(entry, band):
    """Cache-open one band COG of an item; returns rasterio dataset."""
    href = entry["item"].assets[ASSET_FOR_BAND[band]].href
    if band not in entry["ds"]:
        entry["ds"][band] = rasterio.open(href)
    return entry["ds"][band]


def read_timestep(entry, lon, lat, size):
    """Read one (size, size, 12) raw-DN chip (single date) centered on (lon, lat)."""
    half = size * 10 / 2.0  # 10 m/px
    bands = np.zeros((size, size, 12), dtype=np.float32)
    for bi, band in enumerate(S2_BAND_ORDER):
        ds = open_band(entry, band)
        xs, ys = warp_transform("EPSG:4326", ds.crs, [lon], [lat])
        win = from_bounds(xs[0] - half, ys[0] - half, xs[0] + half, ys[0] + half, ds.transform)
        bands[:, :, bi] = ds.read(
            1, window=win, out_shape=(size, size),
            resampling=Resampling.bilinear, boundless=True, fill_value=0,
        ).astype(np.float32)
    return bands


def read_chip(entry_list, lon, lat, size, T):
    """Build a (size, size, T, 12) multitemporal chip from up to T covering scenes.

    The model requires T >= 2. If only one scene covers the point we repeat it.
    Returns (chip, months) where months are the acquisition months (0-11).
    """
    chosen = entry_list[:T] if len(entry_list) >= T else entry_list + [entry_list[-1]] * (T - len(entry_list))
    steps = [read_timestep(e, lon, lat, size) for e in chosen]
    months = [e["item"].datetime.month - 1 for e in chosen]
    return np.stack(steps, axis=2), months  # (S, S, T, 12)


def find_entries(entries, lon, lat):
    """All candidate scenes (cloud-sorted) whose footprint covers the point."""
    pt = Point(lon, lat)
    return [e for e in entries if e["foot"].contains(pt)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max units (0 = all)")
    ap.add_argument("--variant", default="BASE", choices=["BASE", "TINY", "NANO"])
    ap.add_argument("--chip", type=int, default=96, help="chip size in px @10m")
    ap.add_argument("--k", type=int, default=1, help="points sampled per unit")
    ap.add_argument("--t", type=int, default=3, help="timesteps per chip (model needs >=3)")
    ap.add_argument("--out", default="outputs/embeddings.csv")
    args = ap.parse_args()

    units = load_units()
    if args.limit:
        units = units[: args.limit]
    print(f"units: {len(units)} across regions {sorted({u['region'] for u in units})}")

    minx = min(u["geom"].bounds[0] for u in units)
    miny = min(u["geom"].bounds[1] for u in units)
    maxx = max(u["geom"].bounds[2] for u in units)
    maxy = max(u["geom"].bounds[3] for u in units)
    print(f"AOI bbox: {minx:.3f},{miny:.3f},{maxx:.3f},{maxy:.3f}")

    t0 = time.perf_counter()
    entries = get_items([minx, miny, maxx, maxy])
    print(f"STAC: {len(entries)} candidate items "
          f"(cloud%: {[round(e['item'].properties['eo:cloud_cover'],3) for e in entries]})")

    model = load_encoder(getattr(ModelID, f"OLMOEARTH_V1_{args.variant}"))
    print(f"model: OLMOEARTH_V1_{args.variant} loaded ({time.perf_counter()-t0:.1f}s)")

    rows, embed_t = [], 0.0
    for i, u in enumerate(units):
        covering = find_entries(entries, u["lon"], u["lat"])
        if not covering:
            print(f"  [{i+1}/{len(units)}] {u['name']:<22} SKIP (no covering scene)")
            continue
        pts = sample_points(u["geom"], args.k, seed=i)
        vecs, ndvis = [], []
        for (lon, lat) in pts:
            chip, months = read_chip(covering, lon, lat, args.chip, args.t)  # (S,S,T,12)
            red, nir = chip[:, :, 0, B04_IDX], chip[:, :, 0, B08_IDX]  # NDVI from first date
            ndvis.append(float(((nir - red) / (nir + red + 1e-6)).mean()))
            te = time.perf_counter()
            v = embed_sample(model, make_s2_sample(chip, months))
            embed_t += time.perf_counter() - te
            vecs.append(v)
        emb = np.mean(vecs, axis=0)
        dates = "|".join(sorted({str(e["item"].datetime.date()) for e in covering[: args.t]}))
        rows.append({
            "admin_id": u["admin_id"], "name": u["name"], "region": u["region"],
            "lon": u["lon"], "lat": u["lat"], "n_chips": len(pts),
            "date": dates, "scene": covering[0]["item"].id,
            "ndvi_mean": float(np.mean(ndvis)), "emb": emb,
        })
        print(f"  [{i+1}/{len(units)}] {u['name']:<22} {u['region']:<11} "
              f"ndvi={np.mean(ndvis):+.3f} pts={len(pts)} T={args.t} dates={dates}")

    for e in entries:  # close COGs
        for ds in e["ds"].values():
            ds.close()

    # write tidy table: admin_id, meta, ndvi, emb_000..emb_{D-1}
    D = len(rows[0]["emb"])
    cols = ["admin_id", "name", "region", "lon", "lat", "n_chips", "date", "scene", "ndvi_mean"]
    header = cols + [f"emb_{j:03d}" for j in range(D)]
    with open(args.out, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            base = [str(r[c]) for c in cols]
            f.write(",".join(base + [f"{x:.6f}" for x in r["emb"]]) + "\n")

    total = time.perf_counter() - t0
    n = len(rows)
    print(f"\nwrote {n} units x {D}-dim embeddings -> {args.out}")
    print(f"embed dim: {D} | variant: {args.variant} | chip: {args.chip}px k={args.k}")
    print(f"wall-clock total: {total:.1f}s | embed compute: {embed_t:.1f}s "
          f"| per-unit: {total/max(n,1):.1f}s (incl. imagery I/O)")


if __name__ == "__main__":
    main()
