"""Helpers for the DHS transfer-learning demo (see dhs_transfer_demo.ipynb).

Idea: micronutrient labels are scarce, but stunting/wasting/wealth/under-5
mortality are in essentially every geocoded DHS. Pretrain a compact representation
on those ABUNDANT indicators (multi-task, on frozen OlmoEarth embeddings), then
transfer it to the SCARCE micronutrient target. A multi-task bottleneck retains
the shared drivers, so a small target head generalizes from few labels.

=====================================================================
CAVEAT. Indicators and target are SYNTHETIC (built from real OlmoEarth
embeddings) so we can control what the tasks share. This demonstrates the
transfer *mechanism* and honest expectations, not real epidemiology.
=====================================================================
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def latent_factors(E, k=6, seed=0):
    """k standardized 'landscape drivers' = embeddings projected on random
    orthonormal directions. Random (not top-PC) directions on purpose: the
    outcome-relevant axes need not be the highest-variance ones, which is why
    unsupervised PCA compression can miss them."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((E.shape[1], k)))
    Z = E @ Q[:, :k]
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-9)


def ridge_predictor_representation(X, Y, alphas):
    """RECOMMENDED shared representation: one optimally-regularized ridge predictor
    per abundant indicator, stacked. Returns W (p x n_tasks) so that X @ W gives
    the 'predicted abundant indicators' as a compact feature set.

    Closed-form and per-task CV-regularized. In our tests a gradient-trained shared
    bottleneck (see train_trunk) under-regularized the low-variance drivers of one
    task and failed to carry it to the held-out region; giving each task its own
    CV-chosen shrinkage fixes that.
    """
    from sklearn.linear_model import RidgeCV

    return np.stack([RidgeCV(alphas=alphas).fit(X, Y[:, j]).coef_ for j in range(Y.shape[1])], axis=1)


class _Trunk(nn.Module):
    def __init__(self, d_in, bottleneck, n_tasks, hidden=None):
        super().__init__()
        if hidden:   # optional nonlinear encoder (more capacity, less transportable)
            self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck))
        else:        # default: LINEAR shared bottleneck = reduced-rank multi-task regression
            self.enc = nn.Linear(d_in, bottleneck)
        self.heads = nn.Linear(bottleneck, n_tasks)

    def forward(self, x):
        z = self.enc(x)
        return self.heads(z), z


def train_trunk(X, Y, bottleneck=8, hidden=None, epochs=800, lr=2e-3, wd=1e-3, seed=0):
    """Multi-task pretraining: X (embeddings) -> shared bottleneck -> abundant tasks Y.

    Default is a LINEAR bottleneck (reduced-rank multi-task ridge): it learns the
    outcome-relevant subspace from the abundant labels and transports across
    space like ridge does. Set `hidden` for a nonlinear encoder.
    Returns `encode`, a function mapping new embeddings to the frozen bottleneck.
    """
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6      # standardize inside the trunk
    Xn = (Xt - mu) / sd
    model = _Trunk(X.shape[1], bottleneck, Y.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred, _ = model(Xn)
        lossf(pred, Yt).backward()
        opt.step()
    model.eval()

    def encode(Xnew):
        with torch.no_grad():
            xn = (torch.tensor(Xnew, dtype=torch.float32) - mu) / sd
            _, z = model(xn)
        return z.numpy()

    return encode
