"""Helpers for the animal-movement demo (see animal_movement_demo.ipynb).

Pipeline shape (analog of a wildlife/livestock GPS-collar study):
  real Sentinel-2 landscape  ->  simulated 2-state animal track  ->
  OlmoEarth embeddings at points  ->  (a) resource-selection "used vs available"
  and (b) sharp-turn vs straight classification.

Movement-ecology grounding:
  * Area-restricted search (ARS): foraging concentrates in resource-rich patches
    with SHORTER steps and HIGHER turning; transit is fast and directed. We model
    two states (forage/transit) whose switching is biased by local NDVI, so the
    landscape genuinely drives both where the animal dwells and where it turns.
  * This mirrors step-/resource-selection functions (SSF/RSF) and HMM movement-
    state models, with OlmoEarth embeddings standing in for hand-picked habitat
    covariates.

=====================================================================
CAVEATS. (1) The GPS track is SIMULATED, not a real collar. (2) The landscape is
real Sentinel-2, but the NDVI->behavior coupling is a MODELING CHOICE — recovering
it validates the method, it does not discover a fact about real animals. (3) This
is a demo of plumbing + method, not an ecological finding.
=====================================================================
"""

from __future__ import annotations

import time

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

from olmoe import embed_sample, make_s2_sample

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
DATE_RANGE = "2021-05-01/2021-09-30"  # dry season; pre-2022 (no L2A offset)
S2_BAND_ORDER = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
ASSET_FOR_BAND = {
    "B02": "blue", "B03": "green", "B04": "red", "B08": "nir",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3", "B8A": "nir08",
    "B11": "swir16", "B12": "swir22", "B01": "coastal", "B09": "nir09",
}
B04_IDX, B08_IDX = S2_BAND_ORDER.index("B04"), S2_BAND_ORDER.index("B08")


# --------------------------------------------------------------------------- #
# 1. Load one real Sentinel-2 landscape window (downloaded once, sliced locally)
# --------------------------------------------------------------------------- #
def load_landscape(center_lon, center_lat, size_px=640, n_dates=3, max_cloud=10):
    """Read a (H, W, T, 12) raw-DN Sentinel-2 stack centered on a point.

    Downloaded once into memory; the demo then slices chips locally (no per-point
    network reads). Returns (stack, meta) where meta has the UTM affine + crs so
    pixel <-> lon/lat is possible.
    """
    half = size_px * 10 / 2.0
    cat = Client.open(STAC_URL)
    s = cat.search(
        collections=[COLLECTION], intersects={"type": "Point", "coordinates": [center_lon, center_lat]},
        datetime=DATE_RANGE, query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}], max_items=n_dates,
    )
    items = list(s.items())
    if not items:
        raise RuntimeError("no low-cloud scene found for this point/date range")

    t0 = time.perf_counter()
    steps, dates, transform, crs = [], [], None, None
    for it in items[:n_dates]:
        bands = np.zeros((size_px, size_px, 12), dtype=np.float32)
        for bi, band in enumerate(S2_BAND_ORDER):
            with rasterio.open(it.assets[ASSET_FOR_BAND[band]].href) as ds:
                if crs is None:
                    crs = ds.crs
                    xs, ys = warp_transform("EPSG:4326", crs, [center_lon], [center_lat])
                    cx, cy = xs[0], ys[0]
                win = from_bounds(cx - half, cy - half, cx + half, cy + half, ds.transform)
                if transform is None:
                    transform = ds.window_transform(win)
                bands[:, :, bi] = ds.read(
                    1, window=win, out_shape=(size_px, size_px),
                    resampling=Resampling.bilinear, boundless=True, fill_value=0).astype(np.float32)
        steps.append(bands)
        dates.append(str(it.datetime.date()))
    stack = np.stack(steps, axis=2)  # (H, W, T, 12)
    meta = {"transform": transform, "crs": crs, "dates": dates,
            "months": [int(d.split("-")[1]) - 1 for d in dates],
            "scene": items[0].id, "load_s": time.perf_counter() - t0}
    return stack, meta


def ndvi_map(stack):
    """(H, W) mean NDVI over time from a (H, W, T, 12) stack."""
    red = stack[:, :, :, B04_IDX]
    nir = stack[:, :, :, B08_IDX]
    return ((nir - red) / (nir + red + 1e-6)).mean(axis=2)


# --------------------------------------------------------------------------- #
# 2. Simulate a 2-state (forage/transit) correlated random walk over the NDVI
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_track(ndvi, n_steps=120, margin=40, habitat_coupling=6.0,
                   persistence=0.75, seed=0):
    """Two-state correlated random walk whose state is biased by local NDVI.

    forage state: short steps + wide turning (ARS, sharp turns) in greener cells.
    transit state: long steps + concentrated turning (directed).
    habitat_coupling: how strongly NDVI pulls the animal into the forage state
        (set 0 for a habitat-independent control -> turns unrelated to landscape).

    Returns a dict of per-fix arrays (x, y in pixel units; state; step; turn).
    """
    rng = np.random.default_rng(seed)
    H, W = ndvi.shape
    lo, hi = np.percentile(ndvi, 5), np.percentile(ndvi, 95)
    ndvi_n = np.clip((ndvi - lo) / (hi - lo + 1e-6), 0, 1)

    x, y = W / 2.0, H / 2.0           # start near center (col, row)
    theta = rng.uniform(-np.pi, np.pi)
    state = "transit"
    xs, ys, states, steps, turns = [x], [y], [state], [0.0], [0.0]

    for _ in range(n_steps):
        r, c = int(np.clip(round(y), 0, H - 1)), int(np.clip(round(x), 0, W - 1))
        p_forage = _sigmoid(habitat_coupling * (ndvi_n[r, c] - 0.5))
        # sticky Markov switching, target-biased by habitat
        if rng.random() > persistence:
            state = "forage" if rng.random() < p_forage else "transit"

        if state == "forage":
            step = rng.gamma(2.0, 1.5)          # ~30 m mean (10 m px)
            turn = rng.vonmises(0.0, 0.3)       # wide -> frequent sharp turns
        else:
            step = rng.gamma(5.0, 4.0)          # ~200 m mean
            turn = rng.vonmises(0.0, 6.0)       # concentrated -> straight

        theta_new = theta + turn
        nx, ny = x + step * np.cos(theta_new), y + step * np.sin(theta_new)
        # keep inside the usable window (leave room for chips); reflect if needed
        if not (margin <= nx <= W - margin and margin <= ny <= H - margin):
            theta_new = np.arctan2(H / 2.0 - y, W / 2.0 - x) + rng.normal(0, 0.3)
            nx, ny = x + step * np.cos(theta_new), y + step * np.sin(theta_new)
            nx, ny = np.clip(nx, margin, W - margin), np.clip(ny, margin, H - margin)

        x, y, theta = nx, ny, theta_new
        xs.append(x); ys.append(y); states.append(state)
        steps.append(step); turns.append(turn)

    return {"x": np.array(xs), "y": np.array(ys), "state": np.array(states),
            "step": np.array(steps), "turn": np.array(turns)}


def available_points(ndvi, n, margin=40, seed=1):
    """Random 'available' (background) points the animal did NOT visit."""
    rng = np.random.default_rng(seed)
    H, W = ndvi.shape
    xs = rng.uniform(margin, W - margin, n)
    ys = rng.uniform(margin, H - margin, n)
    return xs, ys


def grid_points(ndvi, n_side=8, margin=40):
    """Regular grid of points for a coarse suitability map."""
    H, W = ndvi.shape
    gx = np.linspace(margin, W - margin, n_side)
    gy = np.linspace(margin, H - margin, n_side)
    xx, yy = np.meshgrid(gx, gy)
    return xx.ravel(), yy.ravel()


# --------------------------------------------------------------------------- #
# 3. Embeddings at points (sliced from the in-memory landscape)
# --------------------------------------------------------------------------- #
def embed_points(model, stack, xs, ys, months, chip=64):
    """OlmoEarth embedding for each (x, y) pixel location; slices local chips."""
    H, W, _, _ = stack.shape
    h = chip // 2
    out = []
    for x, y in zip(xs, ys):
        c, r = int(round(x)), int(round(y))
        c = min(max(c, h), W - h); r = min(max(r, h), H - h)
        sub = stack[r - h:r + h, c - h:c + h, :, :]  # (chip, chip, T, 12)
        out.append(embed_sample(model, make_s2_sample(sub, months)))
    return np.array(out)


def sample_ndvi(ndvi, xs, ys):
    """Local NDVI at each point (a simple baseline covariate)."""
    H, W = ndvi.shape
    r = np.clip(np.round(ys).astype(int), 0, H - 1)
    c = np.clip(np.round(xs).astype(int), 0, W - 1)
    return ndvi[r, c]
