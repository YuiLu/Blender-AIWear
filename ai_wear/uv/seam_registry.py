"""UV seam registry, QA metric, fusion, dilation.

A topology edge shared by two faces is a UV seam when the UV coordinates at its
two vertices differ between the two faces. The registry stores the paired UV
coordinates on both sides so we can later sample a UV-domain field on each side
at the same 3D parameter t and force continuity (fusion) plus padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SeamPair:
    edge_index: int
    v0: int
    v1: int
    uv_a0: np.ndarray  # (2,)
    uv_a1: np.ndarray
    uv_b0: np.ndarray
    uv_b1: np.ndarray
    # Unit directions pointing from the seam into each adjacent UV face.  They
    # let fusion reconcile a narrow strip on both islands instead of painting
    # only the edge texel, which otherwise remains visible as a bright halo.
    uv_a_inward: Optional[np.ndarray] = None
    uv_b_inward: Optional[np.ndarray] = None


def _face_inward_uv(face, uv_layer, start: np.ndarray,
                    end: np.ndarray) -> np.ndarray:
    centre = np.mean([
        np.array([loop[uv_layer].uv.x, loop[uv_layer].uv.y], dtype=np.float64)
        for loop in face.loops
    ], axis=0)
    tangent = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(tangent))
    if length < 1e-12:
        return np.zeros(2, dtype=np.float64)
    inward = np.array([-tangent[1], tangent[0]], dtype=np.float64) / length
    midpoint = 0.5 * (np.asarray(start) + np.asarray(end))
    if float(np.dot(inward, centre - midpoint)) < 0.0:
        inward *= -1.0
    return inward


def build_seam_registry(obj, uv_layer_name: Optional[str]) -> List[SeamPair]:
    import bmesh
    if obj.type != "MESH":
        return []
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    uv_lay = bm.loops.layers.uv.get(uv_layer_name or "") or bm.loops.layers.uv.active
    if uv_lay is None:
        bm.free()
        return []
    eps = 1e-5
    seams: List[SeamPair] = []
    for edge in bm.edges:
        faces = edge.link_faces
        if len(faces) != 2:
            continue
        v0, v1 = edge.verts
        loops_a = {l.vert: l for l in faces[0].loops}
        loops_b = {l.vert: l for l in faces[1].loops}
        if v0 not in loops_a or v1 not in loops_a or v0 not in loops_b or v1 not in loops_b:
            continue
        la0, la1 = loops_a[v0][uv_lay].uv, loops_a[v1][uv_lay].uv
        lb0, lb1 = loops_b[v0][uv_lay].uv, loops_b[v1][uv_lay].uv
        a0 = np.array([la0.x, la0.y]); a1 = np.array([la1.x, la1.y])
        b0 = np.array([lb0.x, lb0.y]); b1 = np.array([lb1.x, lb1.y])
        c1 = (np.allclose(a0, b0, atol=eps) and np.allclose(a1, b1, atol=eps))
        c2 = (np.allclose(a0, b1, atol=eps) and np.allclose(a1, b0, atol=eps))
        if not (c1 or c2):
            inward_a = _face_inward_uv(faces[0], uv_lay, a0, a1)
            inward_b = _face_inward_uv(faces[1], uv_lay, b0, b1)
            seams.append(SeamPair(
                edge.index, v0.index, v1.index, a0, a1, b0, b1,
                inward_a, inward_b))
    bm.free()
    return seams


def _bilinear_sample(field: np.ndarray, uvs: np.ndarray, res: int) -> np.ndarray:
    # Texel i is centred at (i + 0.5) / res.  Sampling with u*res (the old
    # implementation) shifts every lookup by half a texel and reads farther
    # outside an island exactly where seam fusion needs the most precision.
    x = np.clip(uvs[:, 0] * res - 0.5, 0.0, res - 1.0)
    y = np.clip(uvs[:, 1] * res - 0.5, 0.0, res - 1.0)
    x0 = np.floor(x).astype(np.int64); x1 = np.minimum(x0 + 1, res - 1)
    y0 = np.floor(y).astype(np.int64); y1 = np.minimum(y0 + 1, res - 1)
    fx = (x - x0).astype(np.float32); fy = (y - y0).astype(np.float32)
    a = field[y0, x0]; b = field[y0, x1]; c = field[y1, x0]; d = field[y1, x1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + d * fx * fy)


def seam_qa(field: np.ndarray, registry: List[SeamPair], res: int,
            samples: int = 16) -> dict:
    if not registry:
        return {"count": 0, "mean": 0.0, "p95": 0.0, "worst": 0.0}
    diffs = []
    for sp in registry:
        t = np.linspace(0.05, 0.95, samples)
        ua = sp.uv_a0[None] * (1 - t[:, None]) + sp.uv_a1[None] * t[:, None]
        ub = sp.uv_b0[None] * (1 - t[:, None]) + sp.uv_b1[None] * t[:, None]
        va = _bilinear_sample(field, ua, res)
        vb = _bilinear_sample(field, ub, res)
        diffs.extend(np.abs(va - vb).tolist())
    d = np.asarray(diffs)
    return {
        "count": len(registry),
        "mean": float(d.mean()) if d.size else 0.0,
        "p95": float(np.percentile(d, 95)) if d.size else 0.0,
        "worst": float(d.max()) if d.size else 0.0,
    }


def fuse_seam(field: np.ndarray, registry: List[SeamPair], res: int,
              diffuse_texels: int = 8, tol: Optional[float] = None,
              valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Symmetrically reconcile corresponding strips on both sides of a seam.

    The previous implementation sampled only the boundary, took ``max`` and
    monotonically raised nearby texels.  That has incompatible meanings for the
    three fields using it: a larger alpha means more wear, a larger
    WearThreshold means *later* wear, and RGB stores a signed residual around
    neutral 0.5.  It could reduce the exact A/B difference while creating the
    bright seam band visible on the ceiling-fan base.

    This implementation samples both islands at equal edge parameter and equal
    inward texel distance, averages them symmetrically, and fades that common
    profile back to the original signal.  Accumulation is order-independent so
    intersecting seams cannot overwrite one another according to mesh order.
    """
    original = np.asarray(field, dtype=np.float32)
    if not registry:
        return original.copy()
    tolerance = 0.02 if tol is None else float(tol)
    width = max(1, int(diffuse_texels))
    accum = np.zeros_like(original, dtype=np.float32)
    weights = np.zeros_like(original, dtype=np.float32)

    def _accumulate(uvs: np.ndarray, values: np.ndarray, strength: float) -> None:
        inside = ((uvs[:, 0] >= 0.0) & (uvs[:, 0] <= 1.0)
                  & (uvs[:, 1] >= 0.0) & (uvs[:, 1] <= 1.0))
        xs = np.clip(np.floor(uvs[:, 0] * res).astype(np.int64), 0, res - 1)
        ys = np.clip(np.floor(uvs[:, 1] * res).astype(np.int64), 0, res - 1)
        if valid is not None:
            inside &= valid[ys, xs]
        if not inside.any():
            return
        xs = xs[inside]; ys = ys[inside]
        vals = values[inside].astype(np.float32)
        np.add.at(accum, (ys, xs), vals * strength)
        np.add.at(weights, (ys, xs), strength)

    for sp in registry:
        edge_texels = max(
            float(np.linalg.norm(sp.uv_a1 - sp.uv_a0)),
            float(np.linalg.norm(sp.uv_b1 - sp.uv_b0))) * res
        sample_count = max(2, int(np.ceil(edge_texels * 2.0)) + 1)
        t = np.linspace(0.0, 1.0, sample_count)
        edge_a = sp.uv_a0[None] * (1.0 - t[:, None]) + sp.uv_a1[None] * t[:, None]
        edge_b = sp.uv_b0[None] * (1.0 - t[:, None]) + sp.uv_b1[None] * t[:, None]
        has_profile = sp.uv_a_inward is not None and sp.uv_b_inward is not None
        max_distance = width if has_profile else 0
        inward_a = (np.asarray(sp.uv_a_inward) if has_profile
                    else np.zeros(2, dtype=np.float64))
        inward_b = (np.asarray(sp.uv_b_inward) if has_profile
                    else np.zeros(2, dtype=np.float64))
        for distance in range(max_distance + 1):
            ua = edge_a + inward_a[None] * (distance / res)
            ub = edge_b + inward_b[None] * (distance / res)
            va = _bilinear_sample(original, ua, res)
            vb = _bilinear_sample(original, ub, res)
            active = np.abs(va - vb) > tolerance
            if not active.any():
                continue
            merged = 0.5 * (va + vb)
            strength = float(1.0 - distance / (max_distance + 1.0))
            _accumulate(ua[active], merged[active], strength)
            _accumulate(ub[active], merged[active], strength)

    touched = weights > 1e-8
    target = original.copy()
    target[touched] = accum[touched] / weights[touched]
    strength = np.clip(weights, 0.0, 1.0)
    return original * (1.0 - strength) + target * strength


def dilate(field: np.ndarray, valid: np.ndarray, steps: int) -> tuple:
    """Grow valid texels outward by `steps`, filling empties with neighbor values.

    Fixes bilinear/mipmap bleed at island borders.
    """
    if steps <= 0:
        return field.copy(), valid.copy()
    f = field.astype(np.float32).copy()
    v = valid.copy()
    res = f.shape[0]
    for _ in range(steps):
        up = np.roll(f, 1, axis=0); vu = np.roll(v, 1, axis=0); vu[0, :] = False
        dn = np.roll(f, -1, axis=0); vd = np.roll(v, -1, axis=0); vd[-1, :] = False
        lf = np.roll(f, 1, axis=1); vl = np.roll(v, 1, axis=1); vl[:, 0] = False
        rt = np.roll(f, -1, axis=1); vr = np.roll(v, -1, axis=1); vr[:, -1] = False
        acc = (np.where(vu, up, 0) + np.where(vd, dn, 0)
               + np.where(vl, lf, 0) + np.where(vr, rt, 0))
        cnt = (vu.astype(np.float32) + vd.astype(np.float32)
               + vl.astype(np.float32) + vr.astype(np.float32))
        has_nbr = cnt > 0
        fill = np.where(has_nbr, acc / np.maximum(cnt, 1), f)
        f = np.where(v, f, fill)
        v = v | (has_nbr & ~v)
    return f, v


def fuse_seam_rgb(field_rgb: np.ndarray, registry: List[SeamPair], res: int,
                  diffuse_texels: int = 8,
                  valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Symmetrically fuse both sides of an (res,res,C) texture.

    fuse_seam/dilate are single-channel (they stamp scalars; np.where on
    (H,W,C) would broadcast-fail), so loop channels reusing the tested
    per-channel code. Averaging each channel independently == averaging the
    color, so the merged RGB is correct.
    """
    if not registry or field_rgb.ndim != 3:
        return field_rgb.copy()
    out = np.empty_like(field_rgb)
    for c in range(field_rgb.shape[2]):
        out[..., c] = fuse_seam(
            field_rgb[..., c], registry, res, diffuse_texels, valid=valid)
    return out


def dilate_rgb(field_rgb: np.ndarray, valid: np.ndarray, steps: int) -> tuple:
    """Grow valid texels outward for an (res,res,C) texture + (res,res) valid.

    Valid growth is channel-independent (depends only on `valid` + neighbors,
    not field values), so each channel is dilated from the SAME original valid
    and the grown valid is identical across channels — we return any one.
    Starting each channel from the original `valid` (not a shared grown-v) is
    what keeps the per-channel fill consistent.
    """
    if steps <= 0:
        return field_rgb.copy(), valid.copy()
    out = np.empty_like(field_rgb)
    v_out = valid.copy()
    for c in range(field_rgb.shape[2]):
        chan, v_c = dilate(field_rgb[..., c].copy(), valid.copy(), steps)
        out[..., c] = chan
        v_out = v_c  # channel-independent; identical each iteration
    return out, v_out


def visualize_seams(obj, registry: List[SeamPair]) -> int:
    """Select seam edges and tag a custom property for viewport/inspection."""
    import bpy
    if obj.type != "MESH":
        return 0
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    idx = {s.edge_index for s in registry}
    bm.edges.ensure_lookup_table()
    # deselect all edges
    for e in bm.edges:
        e.select_set(e.index in idx)
    bm.select_flush_mode()
    # store as custom property on the object for downstream UI
    obj["ai_wear_seam_count"] = len(registry)
    bm.to_mesh(obj.data)
    bm.free()
    return len(registry)
