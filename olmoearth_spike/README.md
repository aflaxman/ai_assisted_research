# olmoearth_spike — feasibility spike

Thin end-to-end vertical slice testing whether **Ai2's OlmoEarth** can produce
per-admin-unit image embeddings for an areal-prediction workflow. The question is
**"does the pipeline run, on what compute, with what data friction"** — not
accuracy.

**Read [`FEASIBILITY.md`](FEASIBILITY.md) for the go/no-go memo** and
[`docs/RECON.md`](docs/RECON.md) for the model recon.

![Spike results: per-unit NDVI choropleth, held-out-district prediction, and the compute breakdown](outputs/figure.png)

*(A) The pipeline produced real per-unit EO features. (B) Embeddings predict a
held-out district's NDVI (ridge, leave-one-region-out R²=+0.18). (C) On CPU the
model is a sliver of wall-clock — imagery I/O dominates. Regenerate with
`uv run --group viz python make_figure.py`.*

> ⚠️ **Proxy target, not real data.** The downstream target is mean NDVI (an open
> EO stand-in) plus a synthetic spatial field — used **only** to prove the
> pipeline runs and that embeddings carry spatial signal. It says nothing about
> micronutrient deficiency, the real project's outcome.

## Pipeline

```
GADM Malawi ADM2 polygons ─┐
                           ├─► low-cloud Sentinel-2 chip per unit (Earth Search STAC)
                           │        │
                           │        ├─► OlmoEarth-v1-Base ─► mean-pool ─► 768-dim vector/unit
                           │        └─► mean NDVI (stand-in target)
                           ▼
              ridge / GBM regressor  ─►  5-fold + leave-one-unit-out + leave-one-region-out
```

## Reproduce

Prerequisites: `uv` (0.8+), Python 3.11, network access to Hugging Face,
`geodata.ucdavis.edu` (GADM), and `earth-search.aws.element84.com` +
`sentinel-cogs.s3.us-west-2.amazonaws.com` (imagery).

```bash
cd olmoearth_spike
uv sync                       # pinned env; CPU-only torch. ~30 s, ~1.1 GB venv

# M1 — smoke test: load BASE + NANO, embed, check determinism
uv run python smoke_test.py

# Boundaries (GADM Malawi ADM2). geoBoundaries is preferable (CC-BY) but its
# GitHub-hosted files were blocked by this session's proxy; GADM works here.
mkdir -p data && curl -sSL \
  "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_MWI_2.json.zip" \
  -o data/gadm_mwi2.zip && (cd data && unzip -o gadm_mwi2.zip)

# M2 — extract one embedding per admin unit.
# COG reads go through GDAL/curl, so point them at the session proxy + CA bundle:
export CURL_CA_BUNDLE=/root/.ccr/ca-bundle.crt GDAL_HTTP_PROXY="$HTTPS_PROXY"
uv run python extract_embeddings.py --variant BASE --chip 64 --k 1 --t 3
#   -> outputs/embeddings.csv   (admin_id, region, ndvi_mean, emb_000..emb_767)
#   (on an unrestricted network the two export lines are unnecessary)

# M3 — downstream model + held-out evaluation
uv run python fit_model.py
#   -> outputs/m3_results.json  + printed R2/MAE table
```

`uv.lock` is committed, so the environment is fully pinned and reproducible.

## Files

| File | Purpose |
|---|---|
| `docs/RECON.md` | M0 — model/API/license recon, read from source before coding |
| `pyproject.toml`, `uv.lock` | pinned env (CPU torch via the pytorch-cpu index) |
| `olmoe.py` | shared loader + correct mean-pool embedding helper |
| `smoke_test.py` | M1 — load + run + determinism, with timings/peak RSS |
| `extract_embeddings.py` | M2 — boundaries → Sentinel-2 → per-unit embeddings |
| `fit_model.py` | M3 — ridge/GBM + 5-fold/LOO/leave-one-region eval |
| `outputs/` | `embeddings.csv`, `m3_results.json`, `m1_smoke_test.log` |
| `FEASIBILITY.md` | M4 — the deliverable: GO/NO-GO memo |

## Key facts (measured on 4-vCPU / 15 GB / no-GPU)

- Runs **CPU-only**; BASE forward ≈14 s on a 128×128×12 chip, ≈0.3 s on 64×64×3;
  peak RSS ≈1.9 GB. Embeddings **deterministic**.
- Model weights download **anonymously** from Hugging Face (no token/gating).
- Imagery from a **public STAC, no account/key**. **Imagery I/O — not the model —
  is the wall-clock bottleneck** (~13 s/unit here vs 0.3 s compute).
- Weights are under the **OlmoEarth Artifact License** (custom; permits research +
  embeddings-as-features; forbids military/surveillance and extractive-industry
  uses — none of which apply here).

## Notable gotchas (grounded in source)

- Encoder returns `{'tokens_and_masks': TokensAndMasks}`; its built-in
  `pool_unmasked_tokens` raises a shape error on the single-modality path, so we
  mean-pool the token tensor directly (see `olmoe.py`).
- The model needs **T ≥ 3** timesteps (undocumented); T=1 and T=2 raise IndexError.
- Sentinel-2 must be fed as **raw DN in [0, 10000]** in the exact band order
  `['B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12','B01','B09']`
  (not wavelength-sorted). We use 2021 scenes to avoid the post-2022 L2A offset.
- README of the loader tells you to `uv sync --extra torch-cpu`, but the published
  `pyproject.toml` (0.0.6) defines no such extra — we pin CPU torch ourselves.
