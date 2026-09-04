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
            seams.append(SeamPair(edge.index, v0.index, v1.index, a0, a1, b0, b1))
    bm.free()
    return seams


def _bilinear_sample(field: np.ndarray, uvs: np.ndarray, res: int) -> np.ndarray:
    x = np.clip(uvs[:, 0] * res, 0, res - 1.001)
    y = np.clip(uvs[:, 1] * res, 0, res - 1.001)
    x0 = np.floor(x).astype(np.int64); x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64); y1 = y0 + 1
    fx = (x - x0).astype(np.float32); fy = (y - y0).astype(np.float32)
    a = field[y0, x0]; b = field[y0, x1]; c = field[y1, x0]; d = field[y1, x1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + d * fx * fy)


def _stamp_blend(field: np.ndarray, x: int, y: int, value: float, radius: int) -> None:
    """Blend `value` into a radius-px disc around (x, y) with a linear falloff.

    The centre takes `value`, the rim keeps the original texel, so a fused seam
    fades in smoothly instead of stamping a hard ``(2r+1)`` block (which read as
    a mosaic at low resolution). Writes in place.
    """
    res = field.shape[0]
    x0 = max(0, x - radius); x1 = min(res, x + radius + 1)
    y0 = max(0, y - radius); y1 = min(res, y + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    w = np.clip(1.0 - dist / max(1.0, float(radius)), 0.0, 1.0)
    region = field[y0:y1, x0:x1]
    # Monotonic: only ever RAISE a texel toward `value`. A plain blend can lower
    # a texel already raised by an earlier stamp (when a later sample's merged
    # value is smaller), eroding the wear peak we are trying to preserve.
    blended = region * (1.0 - w) + value * w
    field[y0:y1, x0:x1] = np.maximum(region, blended)


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
              diffuse_texels: int = 8, tol: Optional[float] = None) -> np.ndarray:
    """Fuse each seam: max-preserve + threshold-gate + distance-falloff blend.

    A UV seam is two copies of the same 3D edge, so the two sides should agree.
    Taking the *max* (not the mean) keeps the ridge wear peak — a seam is usually
    a convex edge where wear is highest, and averaging the two sides would halve
    it. The gate only touches texels where the sides actually disagree
    (``|va-vb| > tol``), leaving already-continuous seams and clean faces alone;
    the falloff blend fades the fix in rather than stamping a hard block.
    """
    out = field.copy()
    if not registry:
        return out
    radius = max(1, diffuse_texels // 4)
    if tol is None:
        tol = 0.02
    t = np.linspace(0.05, 0.95, max(8, res // 16))
    for sp in registry:
        ua = sp.uv_a0[None] * (1 - t[:, None]) + sp.uv_a1[None] * t[:, None]
        ub = sp.uv_b0[None] * (1 - t[:, None]) + sp.uv_b1[None] * t[:, None]
        va = _bilinear_sample(out, ua, res)
        vb = _bilinear_sample(out, ub, res)
        merged = np.maximum(va, vb)  # keep the peak wear, don't average it away
        active = np.abs(va - vb) > tol
        for i in range(len(t)):
            if not active[i]:
                continue
            xi = int(np.clip(ua[i, 0] * res, 0, res - 1))
            yi = int(np.clip(ua[i, 1] * res, 0, res - 1))
            _stamp_blend(out, xi, yi, float(merged[i]), radius)
            xi = int(np.clip(ub[i, 0] * res, 0, res - 1))
            yi = int(np.clip(ub[i, 1] * res, 0, res - 1))
            _stamp_blend(out, xi, yi, float(merged[i]), radius)
    return out


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
                  diffuse_texels: int = 8) -> np.ndarray:
    """Robust-average both sides of each seam for an (res,res,C) texture.

    fuse_seam/dilate are single-channel (they stamp scalars; np.where on
    (H,W,C) would broadcast-fail), so loop channels reusing the tested
    per-channel code. Averaging each channel independently == averaging the
    color, so the merged RGB is correct.
    """
    if not registry or field_rgb.ndim != 3:
        return field_rgb.copy()
    out = np.empty_like(field_rgb)
    for c in range(field_rgb.shape[2]):
        out[..., c] = fuse_seam(field_rgb[..., c], registry, res, diffuse_texels)
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
