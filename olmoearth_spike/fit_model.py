"""M3 — downstream mini-model: OlmoEarth embeddings -> stand-in target.

Fits ridge and gradient boosting from per-unit embeddings to a stand-in target,
and evaluates with (a) 5-fold CV, (b) leave-one-unit-out, and (c)
leave-one-region-out (GroupKFold by district) to rehearse the funded project's
held-out-country transportability test.

=====================================================================
CAVEAT — READ THIS. The target here is an OPEN, EO-derivable PROXY (mean
NDVI + a synthetic smooth spatial field). It exists ONLY to prove the pipeline
runs and that embeddings carry spatial signal. It is NOT micronutrient
deficiency and says NOTHING about predicting deficiency. NDVI is additionally
derived from the same Sentinel-2 bands that feed the embedding, so predicting it
mostly tests that embeddings preserve band/landcover information.
=====================================================================
"""

from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMB_CSV = "outputs/embeddings.csv"


def load():
    rows = list(csv.DictReader(open(EMB_CSV)))
    D = sum(c.startswith("emb_") for c in rows[0])
    X = np.array([[float(r[f"emb_{j:03d}"]) for j in range(D)] for r in rows])
    lon = np.array([float(r["lon"]) for r in rows])
    lat = np.array([float(r["lat"]) for r in rows])
    ndvi = np.array([float(r["ndvi_mean"]) for r in rows])
    region = np.array([r["region"] for r in rows])
    # synthetic smooth spatial field: pure sanity target, independent of imagery.
    # (Tests whether embeddings encode geographic position — they may not, since
    #  no lat/lon was fed to the model. Either result is informative.)
    z = np.sin(2.0 * (lon - lon.mean())) + np.cos(2.0 * (lat - lat.mean()))
    return X, {"ndvi": ndvi, "synthetic_spatial": z}, region, lon, lat


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))


def gbm():
    return GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=0)


def scored(y, pred):
    return {"r2": round(float(r2_score(y, pred)), 3),
            "mae": round(float(mean_absolute_error(y, pred)), 4)}


def evaluate(X, y, region, model_fn, name):
    out = {}
    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    out["cv5"] = scored(y, cross_val_predict(model_fn(), X, y, cv=kf))
    # leave-one-unit-out
    out["loo"] = scored(y, cross_val_predict(model_fn(), X, y, cv=LeaveOneOut()))
    # leave-one-region-out (GroupKFold by district == held-out "region")
    gkf = GroupKFold(n_splits=len(set(region)))
    pred_g = cross_val_predict(model_fn(), X, y, cv=gkf, groups=region)
    out["loro"] = scored(y, pred_g)
    # per-region held-out detail
    out["loro_by_region"] = {}
    for reg in sorted(set(region)):
        m = region == reg
        out["loro_by_region"][reg] = {"n": int(m.sum()), **scored(y[m], pred_g[m])}
    return out


def main():
    X, targets, region, lon, lat = load()
    n, D = X.shape
    print(f"units={n}  embed_dim={D}  regions={sorted(set(region))}")
    print("CAVEAT: proxy target (NDVI / synthetic), NOT micronutrient deficiency.\n")

    coords = np.c_[lon, lat]  # trivial spatial baseline features
    results = {"n_units": n, "embed_dim": D, "targets": {}}

    for tname, y in targets.items():
        print(f"===== target: {tname}  (range {y.min():.3f}..{y.max():.3f}) =====")
        dummy = scored(y, cross_val_predict(DummyRegressor(strategy="mean"), X, y,
                                            cv=KFold(5, shuffle=True, random_state=0)))
        # baseline: ridge on lon/lat only — does the embedding beat raw position?
        coord_ridge = scored(y, cross_val_predict(ridge(), coords, y,
                                                  cv=KFold(5, shuffle=True, random_state=0)))
        r_res = evaluate(X, y, region, ridge, "ridge")
        g_res = evaluate(X, y, region, gbm, "gbm")
        results["targets"][tname] = {
            "baseline_mean": dummy, "baseline_lonlat_ridge": coord_ridge,
            "ridge_emb": r_res, "gbm_emb": g_res,
        }
        print(f"  baseline (predict mean)      MAE={dummy['mae']:.4f}  R2={dummy['r2']}")
        print(f"  baseline (lon/lat ridge)     MAE={coord_ridge['mae']:.4f}  R2={coord_ridge['r2']}")
        print(f"  ridge(emb)  5-fold           R2={r_res['cv5']['r2']:+.3f}  MAE={r_res['cv5']['mae']:.4f}")
        print(f"  ridge(emb)  leave-1-unit-out R2={r_res['loo']['r2']:+.3f}  MAE={r_res['loo']['mae']:.4f}")
        print(f"  ridge(emb)  leave-1-REGION   R2={r_res['loro']['r2']:+.3f}  MAE={r_res['loro']['mae']:.4f}")
        print(f"  gbm(emb)    5-fold           R2={g_res['cv5']['r2']:+.3f}  MAE={g_res['cv5']['mae']:.4f}")
        print(f"  gbm(emb)    leave-1-REGION   R2={g_res['loro']['r2']:+.3f}  MAE={g_res['loro']['mae']:.4f}")
        print(f"  ridge(emb) per held-out region: "
              + ", ".join(f"{k}(n={v['n']}):R2={v['r2']:+.2f}"
                          for k, v in r_res["loro_by_region"].items()))
        print()

    json.dump(results, open("outputs/m3_results.json", "w"), indent=2)
    print("wrote outputs/m3_results.json")
    print("\nINTERPRETATION: 5-fold/LOO R2 gauges whether embeddings linearly encode")
    print("the proxy; leave-1-region R2 rehearses transportability (predicting an")
    print("unseen district). Both are plumbing/signal checks, NOT deficiency skill.")


if __name__ == "__main__":
    main()
