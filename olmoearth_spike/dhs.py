"""Helpers for the DHS geo-displacement demo (see dhs_displacement_demo.ipynb).

Question: DHS survey coordinates are randomly displaced (up to 2 km urban, 5 km
rural, 10 km for 1% of rural clusters) for confidentiality. If we read an
OlmoEarth embedding at the *displaced* point, we sample the wrong landscape and
lose signal. Does pooling embeddings over a displacement-sized *buffer* recover
it? (The matched buffer is guaranteed to contain the true location.)

Design that keeps compute modest: embed a dense GRID over one real Sentinel-2
window ONCE, then every cluster / displacement / buffer is a cheap grid lookup or
average — no re-embedding per experiment.

=====================================================================
CAVEAT. The survey clusters and their outcome are SYNTHETIC; the target is an
open, EO-derivable stand-in (local NDVI), NOT a real DHS indicator. This demo
tests the geo-displacement *mechanism* and the buffer remedy, so a student can
reuse the pattern with real (restricted) DHS data. It is not an ecological or
epidemiological result.
=====================================================================
"""

from __future__ import annotations

import numpy as np

from olmoe import embed_sample, make_s2_sample

PX_PER_KM = 100  # 10 m/pixel

# DHS displacement maxima (Burgert et al. 2013), for reference/labels:
DHS_MAX_KM = {"urban": 2.0, "rural": 5.0, "rural_1pct": 10.0}


def embedding_grid(model, stack, months, step=24, chip=64):
    """Embed a regular grid of chips over the landscape once.

    Returns (E, rows, cols): E is (n_rows, n_cols, D); rows/cols are pixel centers.
    """
    H, W, _, _ = stack.shape
    h = chip // 2
    rows = np.arange(h, H - h + 1, step)
    cols = np.arange(h, W - h + 1, step)
    vecs = []
    for r in rows:
        for c in cols:
            sub = stack[r - h:r + h, c - h:c + h, :, :]
            vecs.append(embed_sample(model, make_s2_sample(sub, months)))
    D = len(vecs[0])
    E = np.array(vecs).reshape(len(rows), len(cols), D)
    return E, rows, cols


def ndvi_grid(ndvi, rows, cols, chip=64):
    """Mean NDVI of each grid cell's chip (the landscape 'truth' the target uses)."""
    h = chip // 2
    g = np.zeros((len(rows), len(cols)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            g[i, j] = ndvi[r - h:r + h, c - h:c + h].mean()
    return g


def _nearest(rows, cols, r, c):
    return int(np.argmin(np.abs(rows - r))), int(np.argmin(np.abs(cols - c)))


def make_clusters(rows, cols, n=180, box_px=120, seed=0):
    """Synthetic survey clusters at TRUE locations in a central box (pixel coords)."""
    rng = np.random.default_rng(seed)
    cr = (rows[0] + rows[-1]) / 2.0
    cc = (cols[0] + cols[-1]) / 2.0
    r = rng.uniform(cr - box_px, cr + box_px, n)
    c = rng.uniform(cc - box_px, cc + box_px, n)
    return np.c_[r, c]  # (n, 2) as (row, col)


def displace(clusters, max_km, seed=0):
    """Apply DHS-style displacement: random angle, distance uniform in [0, max]."""
    rng = np.random.default_rng(seed)
    n = len(clusters)
    ang = rng.uniform(0, 2 * np.pi, n)
    dist = rng.uniform(0, max_km * PX_PER_KM, n)     # uniform in distance, per DHS
    out = clusters.copy()
    out[:, 0] += dist * np.sin(ang)
    out[:, 1] += dist * np.cos(ang)
    return out


def latent_factor(E):
    """A standardized latent landscape factor the embedding captures by construction.

    We use the top principal component of the grid embeddings. This makes "the
    embedding encodes the outcome" true by design, which is the whole premise of
    using OlmoEarth as a covariate — so the experiment isolates the *displacement*
    effect rather than confounding it with how well embeddings encode a given
    index. Returns a (n_rows, n_cols) field.
    """
    from sklearn.decomposition import PCA

    nr, nc, d = E.shape
    pc1 = PCA(n_components=1).fit_transform(E.reshape(-1, d)).reshape(nr, nc)
    return (pc1 - pc1.mean()) / (pc1.std() + 1e-9)


def sample_field(field, rows, cols, pts, noise_sd=0.25, seed=0):
    """Stand-in DHS outcome: the latent factor at each TRUE location + noise."""
    rng = np.random.default_rng(seed)
    y = np.array([field[_nearest(rows, cols, r, c)] for r, c in pts])
    return y + rng.normal(0, noise_sd, len(pts))


def point_features(pts, E, rows, cols):
    """Embedding at the grid cell nearest each point (what you'd naively use)."""
    return np.array([E[_nearest(rows, cols, r, c)] for r, c in pts])


def buffer_features(pts, E, rows, cols, radius_px):
    """Mean embedding over grid cells within radius_px of each point (the remedy)."""
    RR, CC = np.meshgrid(rows, cols, indexing="ij")
    out = []
    for r, c in pts:
        mask = (RR - r) ** 2 + (CC - c) ** 2 <= radius_px ** 2
        if not mask.any():                       # fall back to nearest cell
            out.append(E[_nearest(rows, cols, r, c)])
        else:
            out.append(E[mask].mean(axis=0))
    return np.array(out)
