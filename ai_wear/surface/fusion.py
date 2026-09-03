"""Multi-view fusion: weighted mean + outlier clipping + coverage/confidence.

Accumulation happens in surface.projection. Here we finalize the per-texel AI
field from (sum_mask, sum_weight) and produce coverage / confidence maps for QC.
"""

from __future__ import annotations

import numpy as np


def finalize_ai_field(sum_mask: np.ndarray, sum_weight: np.ndarray,
                      count: np.ndarray) -> dict:
    """Robust weighted mean with outlier clipping. Returns dict of UV-domain arrays."""
    res = int(np.sqrt(sum_mask.size))
    sm = sum_mask.reshape(res, res)
    sw = sum_weight.reshape(res, res)
    cnt = count.reshape(res, res)
    valid = sw > 1e-6
    ai_field = np.zeros((res, res), dtype=np.float32)
    ai_field[valid] = np.clip(sm[valid] / sw[valid], 0.0, 1.0)
    # Outlier clipping: any texel whose accumulated value exceeds 1.5x the local
    # mean of its 8-neighbors is pulled toward that mean (rejects one bad view).
    neighbor = _local_mean(ai_field, valid)
    outlier = valid & (np.abs(ai_field - neighbor) > 0.5) & (cnt >= 1)
    ai_field = np.where(outlier, 0.5 * (ai_field + neighbor), ai_field)
    coverage = (cnt > 0).astype(np.float32)
    confidence = np.clip(sw / (sw.max() + 1e-9), 0.0, 1.0).astype(np.float32)
    return {
        "ai_field": ai_field.astype(np.float32),
        "valid": valid,
        "coverage": coverage,
        "confidence": confidence,
        "coverage_ratio": float(coverage.sum()) / float(res * res),
        "texel_coverage_ratio": float(valid.sum()) / float(max(res * res, 1)),
    }


def finalize_rgb_field(sum_rgb: np.ndarray, sum_weight: np.ndarray) -> dict:
    """Weighted-mean RGB texture from accumulated per-view worn-image samples.

    sum_rgb: (res*res, 3) float32; sum_weight: (res*res,) float32. Mirrors
    finalize_ai_field's weighted mean (no per-channel outlier clip — the facing
    weight already downweights grazing views, and a clip could distort color).
    """
    res = int(np.sqrt(sum_weight.size))
    sw = sum_weight.reshape(res, res)
    sr = sum_rgb.reshape(res, res, 3)
    valid = sw > 1e-6
    rgb = np.zeros((res, res, 3), dtype=np.float32)
    if valid.any():
        rgb[valid] = np.clip(
            sr[valid] / np.maximum(sw[valid][:, None], 1e-9), 0.0, 1.0)
    coverage = (sw > 0).astype(np.float32)
    return {
        "rgb": rgb,
        "weight": sw,
        "valid": valid,
        "coverage": coverage,
        "coverage_ratio": float(coverage.sum()) / float(res * res),
    }


def prepare_worn_overlay(encoded_rgb: np.ndarray, ai_field: np.ndarray,
                         rgb_valid: np.ndarray,
                         mask_low: float = 0.06,
                         mask_high: float = 0.55,
                         dark_limit: float = 0.06,
                         light_limit: float = 0.28) -> dict:
    """Turn fused view residuals into a safe material-overlay texture.

    ``finalize_rgb_field`` only says where a camera observed the surface.  That
    coverage is *not* a wear mask: on a well-covered object it is one almost
    everywhere.  Using it as the WornTex alpha made Wear Amount=100 apply the
    AI view's global brightness shift to the whole model.

    Alpha is therefore derived from the independently fused AI wear mask.
    A smooth contrast window rejects the low-level tone/noise floor while
    preserving strong scratches and edge chips.  The signed RGB residual is
    also bounded: image-edit models often relight the entire object, and an
    unbounded view-space residual is not valid material albedo.

    Returns ``rgb`` in the same unsigned residual encoding (0.5 is neutral) and
    a continuous ``alpha`` wear-mask envelope.  The shader still multiplies
    this alpha by the monotonic WearTime gate, so 30 is a subset of 60 of 100,
    while even 100 remains confined to AI-observed wear instead of whitening
    every camera-covered texel.
    """
    rgb = np.asarray(encoded_rgb, dtype=np.float32)
    mask = np.asarray(ai_field, dtype=np.float32)
    valid = np.asarray(rgb_valid, dtype=bool)

    width = max(float(mask_high) - float(mask_low), 1e-6)
    x = np.clip((mask - float(mask_low)) / width, 0.0, 1.0)
    alpha = (x * x * (3.0 - 2.0 * x)).astype(np.float32)
    alpha *= valid.astype(np.float32)

    # Decode the stored 0.5 + 0.5*delta representation, reject implausible
    # relighting, then encode again for the Non-Color shader texture.
    delta = 2.0 * (rgb - 0.5)
    delta = np.clip(delta, -float(dark_limit), float(light_limit))
    safe_rgb = np.clip(0.5 + 0.5 * delta, 0.0, 1.0).astype(np.float32)
    safe_rgb[~valid] = 0.5
    return {"rgb": safe_rgb, "alpha": alpha}


def _local_mean(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pad = np.pad(np.where(valid, field, 0.0), 1)
    vpad = np.pad(valid.astype(np.float32), 1)
    acc = (pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:]
           + pad[1:-1, 0:-2] + pad[1:-1, 2:] + pad[2:, 0:-2]
           + pad[2:, 1:-1] + pad[2:, 2:])
    n = (vpad[0:-2, 0:-2] + vpad[0:-2, 1:-1] + vpad[0:-2, 2:]
         + vpad[1:-1, 0:-2] + vpad[1:-1, 2:] + vpad[2:, 0:-2]
         + vpad[2:, 1:-1] + vpad[2:, 2:])
    out = np.zeros_like(field)
    m = n > 0
    out[m] = acc[m] / n[m]
    return out
