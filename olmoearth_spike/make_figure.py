"""Make outputs/figure.png — a one-glance visual of the spike.

Three panels tell the whole story:
  A. the geospatial output — mean NDVI per admin unit (the pipeline produced real
     per-unit EO features from boundaries + Sentinel-2);
  B. held-out-district prediction — OlmoEarth embedding -> NDVI (ridge,
     leave-one-region-out), showing embeddings carry transferable spatial signal;
  C. the feasibility headline — on CPU the model is cheap; imagery I/O dominates.

Regenerate:  uv run --group viz python make_figure.py

CAVEAT: NDVI is an open PROXY target to prove plumbing + signal, NOT
micronutrient deficiency.
"""

from __future__ import annotations

import csv
import json
from math import cos, radians

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from shapely.geometry import shape
from sklearn.model_selection import GroupKFold, cross_val_predict

from fit_model import load, ridge

GADM = "data/gadm41_MWI_2.json"
REGION_COLORS = {"Blantyre": "#4C72B0", "Chiradzulu": "#DD8452", "Zomba": "#55A868"}


def unit_polygons():
    feats = {f["properties"]["GID_2"]: f["geometry"] for f in json.load(open(GADM))["features"]}
    rows = list(csv.DictReader(open("outputs/embeddings.csv")))
    out = []
    for r in rows:
        geom = shape(feats[r["admin_id"]])
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        out.append({"region": r["region"], "ndvi": float(r["ndvi_mean"]),
                    "lon": float(r["lon"]), "lat": float(r["lat"]), "polys": polys})
    return out


def main():
    X, targets, region, lon, lat = load()
    y = targets["ndvi"]
    # leave-one-region-out predictions (== FEASIBILITY leave-1-region number)
    gkf = GroupKFold(n_splits=len(set(region)))
    pred = cross_val_predict(ridge(), X, y, cv=gkf, groups=region)
    from sklearn.metrics import mean_absolute_error, r2_score
    r2, mae = r2_score(y, pred), mean_absolute_error(y, pred)

    units = unit_polygons()
    fig = plt.figure(figsize=(14, 4.7), dpi=150)
    fig.suptitle("OlmoEarth feasibility spike — pipeline runs end-to-end on CPU "
                 "(proxy target: NDVI, not micronutrient deficiency)",
                 fontsize=12, fontweight="bold")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05], wspace=0.28,
                          left=0.04, right=0.97, bottom=0.11, top=0.86)

    # ---- Panel A: choropleth of per-unit NDVI ----
    axA = fig.add_subplot(gs[0, 0])
    norm = Normalize(min(u["ndvi"] for u in units), max(u["ndvi"] for u in units))
    cmap = plt.cm.YlGn
    for u in units:
        for poly in u["polys"]:
            xs, ys = poly.exterior.xy
            axA.fill(xs, ys, facecolor=cmap(norm(u["ndvi"])), edgecolor="white", linewidth=0.5)
    axA.set_aspect(1.0 / cos(radians(np.mean(lat))))
    axA.set_title("A. Sentinel-2 mean NDVI per admin unit\n(22 units, 3 Malawi districts)", fontsize=9.5)
    axA.set_xlabel("lon"); axA.set_ylabel("lat"); axA.tick_params(labelsize=7)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axA, fraction=0.046, pad=0.03)
    cb.set_label("mean NDVI", fontsize=8); cb.ax.tick_params(labelsize=7)

    # ---- Panel B: held-out-district predicted vs actual ----
    axB = fig.add_subplot(gs[0, 1])
    for reg, c in REGION_COLORS.items():
        m = region == reg
        axB.scatter(y[m], pred[m], s=42, c=c, edgecolor="k", linewidth=0.4, label=reg, zorder=3)
    lo, hi = min(y.min(), pred.min()) - 0.02, max(y.max(), pred.max()) + 0.02
    axB.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, zorder=1)
    axB.set_xlim(lo, hi); axB.set_ylim(lo, hi); axB.set_aspect("equal")
    axB.set_xlabel("actual NDVI"); axB.set_ylabel("predicted NDVI")
    axB.set_title("B. Embedding → held-out district NDVI\nridge, leave-one-region-out", fontsize=9.5)
    axB.tick_params(labelsize=7)
    axB.legend(fontsize=7, loc="upper left", framealpha=0.9)
    axB.text(0.97, 0.05, f"R² = {r2:+.2f}\nMAE = {mae:.3f}", transform=axB.transAxes,
             ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.6"))

    # ---- Panel C: compute breakdown (the feasibility headline) ----
    axC = fig.add_subplot(gs[0, 2])
    model_s, io_s = 7.1, 292.3 - 7.1  # measured, 22-unit BASE run
    io_pct = 100 * io_s / (model_s + io_s)
    wedges, _ = axC.pie(
        [model_s, io_s], colors=["#3182bd", "#de2d26"], startangle=90,
        counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5))
    axC.text(0, 0.12, f"{io_pct:.0f}%", ha="center", va="center", fontsize=20, fontweight="bold")
    axC.text(0, -0.16, "imagery I/O", ha="center", va="center", fontsize=9)
    axC.set_title("C. On CPU the model is cheap;\nimagery I/O dominates wall-clock", fontsize=9.5)
    axC.legend(wedges, [f"model compute ({model_s:.0f}s)", f"imagery I/O ({io_s:.0f}s)"],
               fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False)
    facts = ("CPU-only • no GPU • BASE 89M / 768-dim\n"
             "weights: anonymous HF download\n"
             "imagery: public STAC, no key\n"
             "embeddings deterministic")
    axC.text(0.5, -0.30, facts, transform=axC.transAxes, ha="center", va="top",
             fontsize=8.2, bbox=dict(boxstyle="round", fc="#f0f0f0", ec="0.7"))

    fig.savefig("outputs/figure.png", bbox_inches="tight", facecolor="white")
    print(f"wrote outputs/figure.png  (LORO R2={r2:+.3f}, MAE={mae:.3f})")


if __name__ == "__main__":
    main()
