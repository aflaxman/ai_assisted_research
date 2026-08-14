# M0 — Recon: OlmoEarth for admin-unit embedding extraction

*Feasibility spike. Goal: decide whether Ai2's OlmoEarth can run end-to-end to
produce image embeddings usable as areal-prediction covariates. This doc records
what I read before touching code, so later milestones don't invent APIs.*

Sources read (2026-07-18):

- Loader repo `allenai/olmoearth_pretrain_minimal` (README + source, cloned `depth 1` of `main`)
- Full repo `allenai/olmoearth_pretrain` (LICENSE)
- HF collection `allenai/olmoearth` and model card `allenai/OlmoEarth-v1-Base`
- Tech report references (arXiv 2511.13655 for v1; v1.1 / v1.2 tech reports)

---

## 1. Artifact / variant I'll use

Two Python packages exist. I'll use the **minimal** one — it is exactly the
"load pretrained weights and run the encoder" surface this spike needs, with no
training/eval deps.

- `olmoearth-pretrain-minimal` — on PyPI (v0.0.6) and GitHub. Loads configs +
  weights from Hugging Face, builds the model, runs the encoder. Code is
  **Apache-2.0**.
- `olmoearth_pretrain` — the full data/training/eval stack. Not needed here.

**Variants** (encoder params, from the minimal README table):

| Family | Nano | Tiny | Small | Base | Large |
|--------|------|------|-------|------|-------|
| v1     | 1.4M | 6.2M | —     | 89M  | 308M  |
| v1.1   | 1.7M | 12.5M| —     | 114M | —     |
| v1.2   | 1.7M | 12.5M| 35.6M | 114M | —     |

`ModelID` enum values map 1:1 to HF repos `allenai/OlmoEarth-<variant>`
(`model_loader.py:43-62`).

**Plan for this box (4 vCPU, 15 GB RAM, no GPU):**

- **Smoke test (M1):** smallest that works — `OLMOEARTH_V1_NANO` (1.4M), then
  `TINY`.
- **Embedding run (M2):** use `BASE` (89M) *only if* it loads and runs on CPU in
  reasonable wall-clock; otherwise fall back to `TINY`/`NANO` and say so. The
  spike measures plumbing, not accuracy, so a small variant is acceptable and
  the memo will note that BASE is the variant a funded project would use.

I'll pin **v1** for the spike (the tech-report-documented baseline). v1.2 is
newer and marketed for "image feature extraction," but v1 is sufficient to prove
the pipeline and keeps this grounded in the most-documented artifact.

## 2. How it's loaded and run (grounded in source)

```python
from olmoearth_pretrain_minimal import ModelID, load_model_from_id
model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=True)
model.eval()
```

`load_model_from_id` downloads `config.json` + `weights.pth` via
`huggingface_hub.hf_hub_download` from `allenai/<variant>`, builds the model from
the config, and loads the state dict on CPU (`map_location="cpu"`)
(`model_loader.py:65-132`). **No HF token or gated-access acceptance appears in
the loader** — downloads are anonymous `hf_hub_download`. (To be confirmed at
runtime in M1; if HF starts gating these repos, that's an M1 blocker.)

Running the encoder (from the README "Sample Code", verified against source):

```python
output = model.encoder(sample, patch_size=8, input_res=10, fast_pass=True)
```

- `sample` is a `MaskedOlmoEarthSample` (see §3).
- The encoder returns a **`TokensAndMasks`** NamedTuple, which carries the token
  embeddings plus masks and **built-in pooling methods**
  (`nn/flexi_vit.py:75-296`):
  - `pool_unmasked_tokens(pooling_type=PoolingType.MEAN|MAX, spatial_pooling=False)`
    → one vector per sample (instance-wise pool over tokens).
  - `pool_spatially(...)`, `pool_instance_wise(...)` for other aggregations.
- So **aggregating a chip to a single embedding vector is a first-class op** — I
  don't have to hand-roll pooling. For "one vector per admin unit" I'll mean-pool
  each chip's tokens, then average chips within a unit.
- `fast_pass=True` selects the inference path (`Encoder.forward` branches on
  `fast_pass or not self.training`, `nn/flexi_vit.py:1896-1899`).

**Embedding dimensionality:** set by `embedding_size` in each variant's
`config.json` — not hard-coded in the loader. I'll read it off the output shape
at runtime (M1) rather than guess.

## 3. Exact input spec

**Modalities.** Trained on three satellite modalities — **Sentinel-2 L2A,
Sentinel-1, Landsat** — plus six derived maps (OpenStreetMap, ESA WorldCover,
USDA CDL, SRTM DEM, WRI Canopy Height, WorldCereal). For this spike I'll feed a
**single modality, Sentinel-2 L2A**, which is the easiest to fetch cloud-free
from public STAC catalogs. The encoder accepts a subset of modalities (the sample
in the README uses S2 alone).

**Band order (must match exactly; `nn/.../constants.py`, echoed in README):**

- `SENTINEL2_L2A` (12): `['B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12','B01','B09']`
- `SENTINEL1` (2): `['vv','vh']`
- `LANDSAT` (11): `['B8','B1','B2','B3','B4','B5','B6','B7','B9','B10','B11']`
- `WORLDCOVER` (1): `['B1']`; `SRTM` (1): `['srtm']`

Note the S2 order is **not** wavelength-sorted — B08 comes before B05, and B01/B09
come last. Getting this wrong silently produces garbage embeddings, so the M2
fetch must reorder bands to this exact list.

**Tensor shapes:**

- Multitemporal (S2/S1/Landsat): `(batch, height, width, time, bands)`
- Single-temporal (WorldCover/SRTM): `(batch, height, width, bands)`
- The README sample uses `(1, 128, 128, 12, 12)` — a 128×128 chip, 12 timesteps,
  12 bands.

**Timestamps:** a `(batch, time, 3)` long tensor; column 1 holds the month index
(`timestamps[:, :, 1] = arange(T)`). Columns appear to be (year, month, day)-ish
indices used for temporal position encoding; month is the one the sample sets.

**Masks:** each modality needs a companion `<modality>_mask` of shape
`(batch, H, W, time)` (long); all-zeros = "nothing masked / all valid".

**Normalization:** `Normalizer(std_multiplier=2.0).normalize(Modality.SENTINEL2_L2A, arr)`
before building the sample. Norm stats ship in the package
(`data/norm_configs/predefined.json`, `computed.json`). So I do **not** need to
compute dataset statistics — I use the model's own.

**Resolution / patching:** `input_res=10` (S2 native 10 m/px) and `patch_size=8`
in the sample call. So a 128×128 chip at 10 m ≈ a 1.28 km × 1.28 km tile,
tokenized into 16×16 patches per timestep.

## 4. License and fitness for intended use

Two distinct licenses:

- **Code** (`olmoearth_pretrain_minimal`): **Apache-2.0** (verified — read the
  full LICENSE file in the clone).
- **Weights**: **OlmoEarth Artifact License** — a *custom* Ai2 "responsible-AI"
  license (`allenai/olmoearth_pretrain/LICENSE`), used for the model card too.

What the Artifact License permits/forbids (read from the raw LICENSE):

- **Permits** use, reproduction, modification, display, distribution of the
  artifacts; explicitly permits **derivative works and using model outputs**
  (transfer of patterns, synthetic data from outputs) — i.e. **using embeddings
  as features is allowed**. Free of charge.
- **Commercial-neutral:** restrictions are on *use type*, not on profit.
- **Prohibited uses:** (a) weapons/military/intelligence/surveillance/policing;
  (b) extractive industry — "removal of raw materials from the earth" incl.
  drilling, mining, deforestation.
- Also invokes Ai2's Responsible Use Guidelines.

**Fitness verdict:** Predicting Admin-1/2 micronutrient-deficiency prevalence for
a public-health project is squarely permitted research use and hits **none** of
the prohibited categories. ✅ The one thing a funded project should confirm: this
is a bespoke license, not OSI-approved, so an institutional legal skim is prudent
before productionizing — but nothing about the intended use conflicts with it.

## 5. Stated hardware requirements

Not stated explicitly in README or model card. Signals:

- Package ships CPU (`torch-cpu`) and GPU (`torch-cu128`) install paths → CPU
  inference is a supported, intended mode.
- Weights load with `map_location="cpu"`.
- Encoder param counts: Nano 1.4M / Tiny 6.2M are trivially CPU-runnable; Base
  89M / Large 308M are CPU-runnable for inference but slower.

I'll **measure** actual CPU wall-clock and peak RAM in M1/M2 rather than rely on
a stated spec (there isn't one). That measurement is a core deliverable — it
drives the GPU-vs-CPU budget question in the memo.

## 6. Known discrepancy to work around (M1)

The README says installation **must** use `uv sync --extra torch-cpu` (or
`torch-cu128`). But the **published `pyproject.toml` (v0.0.6) defines no such
extras** — it lists `torch>=2.7` as a plain dependency and only a `dev` extra.
So the documented install command would fail against the current package.

**Workaround:** I'll build my own pinned `uv` project that depends on
`olmoearth-pretrain-minimal==0.0.6` and pins **CPU-only torch** via the PyTorch
CPU wheel index (keeps downloads modest — the default PyPI torch wheel bundles
CUDA libs, ~GB). The README's "Sample Code" block is the official example and
I'll run it verbatim as the smoke test.

---

## GATE decision — M0

- Weights openly downloadable from Hugging Face with **no gating/token** observed
  in the loader (confirm at runtime). ✅
- Code is Apache-2.0; usable and modifiable. ✅
- Weights license **permits** research use, derivative works, and
  embeddings-as-features for public-health prediction; prohibited-use categories
  don't apply. ✅
- Input spec, loader API, and pooling are all documented and grounded in source.
  ✅

**PROCEED to M1.** Open risks carried forward: (1) confirm anonymous HF download
works; (2) confirm CPU install/run works and measure cost; (3) confirm the S2
band-order and normalization path at runtime.
