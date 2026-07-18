"""Shared OlmoEarth helpers used by the smoke test and the embedding extractor.

Grounded in the installed `olmoearth-pretrain-minimal==0.0.6` API (see docs/RECON.md).

Design note on pooling: `model.encoder(...)` returns
`{'tokens_and_masks': TokensAndMasks}`. For a single Sentinel-2 modality the S2
token tensor is (batch, Hp, Wp, time, band_sets, embed_dim). The package's own
`TokensAndMasks.pool_unmasked_tokens` raises a shape error on this single-modality
path (its output mask lacks the time axis), so we mean-pool the token tensor
directly over the spatial/temporal/band-set axes. Because we feed complete,
low-cloud chips (no missing pixels), every token is valid and a plain mean is the
standard, deterministic choice.
"""

from __future__ import annotations

import numpy as np
import torch

from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
)

# Exact band order the model expects for Sentinel-2 L2A (NOT wavelength-sorted).
S2_BAND_ORDER = list(Modality.SENTINEL2_L2A.band_order)
_NORMALIZER = Normalizer(std_multiplier=2.0)


def load_encoder(model_id: ModelID):
    """Load a pretrained OlmoEarth model in eval mode (weights from Hugging Face)."""
    model = load_model_from_id(model_id, load_weights=True)
    model.eval()
    return model


def make_s2_sample(chip: np.ndarray, months: list[int] | None = None) -> MaskedOlmoEarthSample:
    """Build a normalized single-Sentinel-2 sample from a chip.

    Args:
        chip: float array (H, W, T, 12) with bands in S2_BAND_ORDER (reflectance).
        months: length-T month indices (0-11) for temporal position encoding;
            defaults to 0..T-1 when the true acquisition months are unknown.
    """
    assert chip.ndim == 4 and chip.shape[-1] == 12, f"expected (H,W,T,12), got {chip.shape}"
    h, w, t, _ = chip.shape
    arr = chip[None].astype(np.float32)  # (1, H, W, T, 12)
    arr = _NORMALIZER.normalize(Modality.SENTINEL2_L2A, arr)
    timestamps = torch.zeros(1, t, 3, dtype=torch.long)
    months = months if months is not None else list(range(t))
    timestamps[0, :, 1] = torch.tensor(months[:t], dtype=torch.long)
    return MaskedOlmoEarthSample(
        timestamps=timestamps,
        sentinel2_l2a=torch.from_numpy(arr).float(),
        sentinel2_l2a_mask=torch.zeros(1, h, w, t, dtype=torch.long),
    )


@torch.no_grad()
def embed_sample(model, sample: MaskedOlmoEarthSample, patch_size=8, input_res=10) -> np.ndarray:
    """Return one mean-pooled embedding vector (embed_dim,) for a chip sample."""
    out = model.encoder(sample, patch_size=patch_size, input_res=input_res, fast_pass=True)
    tokens = out["tokens_and_masks"].sentinel2_l2a  # (B, Hp, Wp, T, band_sets, D)
    pooled = tokens.mean(dim=(1, 2, 3, 4))  # (B, D)
    return pooled.squeeze(0).cpu().numpy()
