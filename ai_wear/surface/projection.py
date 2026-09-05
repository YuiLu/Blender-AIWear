"""Projection, screen-space mask extraction, software z-buffer, accumulation.

Projection uses the camera's inverted world matrix (exact Blender convention:
objects in front have negative camera-space Z) plus a manual perspective
formula from lens/sensor. Depth and screen-mask are sampled with the same
convention, so the visibility test is self-consistent — no multilayer-EXR
loading, no Blender depth-pass ambiguity.

Per view we accumulate a weighted contribution into the mesh/UV surface field:
  visible = (texel_depth <= zbuffer + eps)        # not occluded
  facing  = max(0, dot(N, view_dir))^gamma         # front-facing
  w       = visible * facing * confidence
  sum_mask[x] += w * m ;  sum_weight[x] += w
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .. import utils


# --- camera ------------------------------------------------------------------

def camera_view(camera) -> np.ndarray:
    """World→camera-space 4x4 (Blender convention; in-front has z<0)."""
    return np.array(camera.matrix_world.inverted(), dtype=np.float32)


def _fov(sensor_w: float, lens: float, res_x: int, res_y: int) -> Tuple[float, float]:
    fov_x = 2.0 * math.atan(sensor_w / (2.0 * lens))
    aspect = res_x / max(1, res_y)
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect)
    return fov_x, fov_y


def project_points(view: np.ndarray, P: np.ndarray, lens: float,
                   sensor_w: float, res_x: int, res_y: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project world points P (N,3) → (px, py, depth, in_front)."""
    n = P.shape[0]
    Ph = np.empty((n, 4), dtype=np.float32)
    Ph[:, :3] = P
    Ph[:, 3] = 1.0
    cam = Ph @ view.T  # (N,4) camera space
    z = cam[:, 2]
    in_front = z < -1e-5
    neg_z = np.where(in_front, -z, 1.0)
    _, fov_y = _fov(sensor_w, lens, res_x, res_y)
    tan_half_x = math.tan(math.atan(sensor_w / (2.0 * lens)))
    tan_half_y = math.tan(fov_y / 2.0)
    ndc_x = (cam[:, 0] / neg_z) / tan_half_x
    ndc_y = (cam[:, 1] / neg_z) / tan_half_y
    px = (ndc_x * 0.5 + 0.5) * res_x
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * res_y
    depth = neg_z  # positive distance
    return px, py, depth, in_front


def bilinear_sample_points(arr: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Sample a 2D array (H,W) or (H,W,C) at float pixel coords (top-left origin)."""
    H, W = arr.shape[:2]
    x = np.clip(px, 0, W - 1.001)
    y = np.clip(py, 0, H - 1.001)
    x0 = np.floor(x).astype(np.int64); x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64); y1 = y0 + 1
    fx = (x - x0).astype(np.float32); fy = (y - y0).astype(np.float32)
    a = arr[y0, x0]; b = arr[y0, x1]; c = arr[y1, x0]; d = arr[y1, x1]
    if arr.ndim == 3:
        # broadcast the per-pixel weights over the channel axis: (k,1)*(k,C)
        fx = fx[:, None]; fy = fy[:, None]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + d * fx * fy)


# --- software z-buffer -------------------------------------------------------

def _edge(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def rasterize_screen_depth(verts_world: np.ndarray, tri_vert: np.ndarray,
                            view: np.ndarray, lens: float, sensor_w: float,
                            res_x: int, res_y: int) -> Tuple[np.ndarray, np.ndarray]:
    """Front-most depth per screen pixel (software z-buffer). Returns (depth, coverage)."""
    ntri = tri_vert.shape[0]
    # project all verts once
    vw = verts_world[tri_vert.reshape(-1)]  # (ntri*3,3)
    px, py, depth, in_front = project_points(view, vw, lens, sensor_w, res_x, res_y)
    px = px.reshape(ntri, 3); py = py.reshape(ntri, 3)
    depth = depth.reshape(ntri, 3); in_front = in_front.reshape(ntri, 3)

    depth_buf = np.full((res_y, res_x), np.inf, dtype=np.float32)
    coverage = np.zeros((res_y, res_x), dtype=bool)
    eps = 1e-3
    pix_eps = 1e-3

    for t in range(ntri):
        if not in_front[t].all():
            continue
        x0p, y0p = px[t, 0], py[t, 0]
        x1p, y1p = px[t, 1], py[t, 1]
        x2p, y2p = px[t, 2], py[t, 2]
        area = _edge(x0p, y0p, x1p, y1p, x2p, y2p)
        if abs(area) < 1e-8:
            continue
        xmin = max(0, int(min(x0p, x1p, x2p) - 1))
        xmax = min(res_x - 1, int(max(x0p, x1p, x2p) + 1))
        ymin = max(0, int(min(y0p, y1p, y2p) - 1))
        ymax = min(res_y - 1, int(max(y0p, y1p, y2p) + 1))
        if xmax < xmin or ymax < ymin:
            continue
        xs = np.arange(xmin, xmax + 1, dtype=np.int64) + 0.5
        ys = np.arange(ymin, ymax + 1, dtype=np.int64) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        PX = gx.ravel(); PY = gy.ravel()
        w0 = _edge(x1p, y1p, x2p, y2p, PX, PY) / area
        w1 = _edge(x2p, y2p, x0p, y0p, PX, PY) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -pix_eps) & (w1 >= -pix_eps) & (w2 >= -pix_eps)
        if not inside.any():
            continue
        d0, d1, d2 = depth[t]
        d = w0 * d0 + w1 * d1 + w2 * d2
        mask = inside.reshape(len(ys), len(xs))
        sub = depth_buf[ymin:ymax + 1, xmin:xmax + 1]
        d_2d = d.reshape(len(ys), len(xs)).astype(np.float32)
        np.minimum(sub, np.where(mask, d_2d, np.inf), out=sub)
        cov_sub = coverage[ymin:ymax + 1, xmin:xmax + 1]
        cov_sub |= mask
    depth_buf[~coverage] = 0.0  # background: no surface (won't match texels)
    return depth_buf, coverage


# --- screen mask extraction -------------------------------------------------

def _luminance(rgba: np.ndarray) -> np.ndarray:
    return 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]


def extract_screen_mask(clean_png: str, worn_png: str, res: int,
                        return_worn_rgb: bool = False) -> Tuple:
    """Diff clean vs worn → 0..1 wear mask + a confidence scalar.

    The mask IS a subtraction (|worn - clean|), but a raw diff is dominated by
    the global tone offset AI models add between the clean and worn pass (a small
    ~0.03 mean shift saturates to ~0.6 because of the p95/0.05 scale floor).
    So we first luminance-match worn to clean (per-channel mean scale) — this
    removes the global exposure shift and leaves only the local wear edges as
    the diff signal. frac>0.02 drops from ~0.81 to ~0.06 on real captures.

    If return_worn_rgb, also returns an encoded LINEAR color residual:
      encoded_delta = 0.5 + 0.5 * (exposure-matched worn - clean)
    The shader decodes it and adds the residual to the authored Base Color. This
    avoids treating view-lit AI render RGB as albedo and lighting it a second
    time in Blender.
    """
    clean = utils.load_image_rgba(clean_png)
    worn = utils.load_image_rgba(worn_png)
    if clean.shape[:2] != (res, res):
        clean = utils.resize_bilinear(clean, res, res)
    if worn.shape[:2] != (res, res):
        worn = utils.resize_bilinear(worn, res, res)
    c3 = clean[..., :3].astype(np.float32)
    w3 = worn[..., :3].astype(np.float32)
    # luminance-match: scale worn so its per-channel mean equals clean's.
    # c_mean/w_mean is ~1.0 when the passes already match in exposure; it
    # corrects the global tone offset without touching local structure.
    c_mean = c3.reshape(-1, 3).mean(axis=0)
    w_mean = w3.reshape(-1, 3).mean(axis=0)
    scale = np.where(w_mean > 1e-4, c_mean / w_mean, 1.0).astype(np.float32)
    w_match = np.clip(w3 * scale, 0.0, 1.0)
    # luminance diff on the matched image + per-channel max for hue/structure
    cg = _luminance(c3)
    wg = _luminance(w_match)
    diff = np.abs(wg - cg).astype(np.float32)
    chan = np.abs(w_match - c3).max(axis=-1).astype(np.float32)
    raw = np.maximum(diff, chan)
    p95 = float(np.percentile(raw[raw > 0.02], 95)) if (raw > 0.02).any() else 0.1
    scale_norm = max(p95, 0.05)
    mask = np.clip(raw / scale_norm, 0.0, 1.0).astype(np.float32)
    confidence = float(np.clip(p95 / 0.15, 0.2, 1.0))
    if return_worn_rgb:
        # Encode signed linear residual into an unsigned RGB texture. 0.5 is
        # neutral, 0 is -1 and 1 is +1. The shader reverses this encoding before
        # adding it to the original Base Color.
        encoded_delta = np.clip(0.5 + 0.5 * (w_match - c3), 0.0, 1.0)
        return mask, confidence, encoded_delta.astype(np.float32)
    return mask, confidence


def reject_depth_edge_payload(screen_mask: np.ndarray,
                              encoded_rgb: np.ndarray,
                              depth_buf: np.ndarray,
                              coverage: np.ndarray,
                              radius: float,
                              guard_pixels: Optional[int] = None,
                              jump_fraction: float = 0.005) -> Tuple:
    """Neutralize unreliable image-edit differences at occlusion boundaries.

    Image-edit models can preserve the overall silhouette while moving a part
    boundary by a few pixels.  At such pixels the clean-geometry Z buffer says
    "rear surface", while the edited image contains the foreground part.  A
    mathematically correct reprojection then stamps that foreground rim onto the
    rear surface.  This is especially visible as long white arcs on dark parts.

    We identify background silhouettes and sharp depth discontinuities in the
    clean geometry, dilate them by a small screen-space guard, and make the
    scalar mask zero / signed RGB residual neutral there.  Smooth depth changes
    on one surface are retained, as are geometric-edge priors computed later in
    object space.
    """
    mask = np.asarray(screen_mask, dtype=np.float32).copy()
    rgb = np.asarray(encoded_rgb, dtype=np.float32).copy()
    depth = np.asarray(depth_buf, dtype=np.float32)
    covered = np.asarray(coverage, dtype=bool)
    if mask.shape != covered.shape or depth.shape != covered.shape:
        return mask, rgb, covered.copy()

    jump = max(abs(float(radius)) * float(jump_fraction), 1e-5)
    depth_pad = np.pad(depth, 1, mode="edge")
    cover_pad = np.pad(covered, 1, mode="constant", constant_values=False)
    edge = np.zeros_like(covered, dtype=bool)
    for dy, dx in ((0, 1), (1, 0), (1, 2), (2, 1)):
        neighbour_depth = depth_pad[dy:dy + depth.shape[0],
                                    dx:dx + depth.shape[1]]
        neighbour_cover = cover_pad[dy:dy + depth.shape[0],
                                    dx:dx + depth.shape[1]]
        edge |= covered & (
            ~neighbour_cover | (np.abs(depth - neighbour_depth) > jump))

    # About 0.6% of the render width: 3 px at 512, 6 px at 1024 and 12 px at
    # 2048. Image-edit boundary drift scales with image resolution.
    guard = (max(2, int(round(min(depth.shape) * 0.006)))
             if guard_pixels is None else max(0, int(guard_pixels)))
    blocked = edge
    for _ in range(guard):
        pad = np.pad(blocked, 1, mode="constant", constant_values=False)
        blocked = np.zeros_like(blocked)
        for dy in range(3):
            for dx in range(3):
                blocked |= pad[dy:dy + depth.shape[0],
                               dx:dx + depth.shape[1]]
    safe = covered & ~blocked
    mask[~safe] = 0.0
    if rgb.ndim == 3 and rgb.shape[:2] == covered.shape:
        rgb[~safe] = 0.5
    return mask, rgb, safe


# --- accumulation -----------------------------------------------------------

def _select_facing(texel_pos: np.ndarray, texel_norm: np.ndarray,
                   valid_idx: np.ndarray, view: np.ndarray, cam_loc: np.ndarray,
                   lens: float, sensor_w: float, res_x: int, res_y: int,
                   depth_buf: np.ndarray, depth_eps: float, gamma: float):
    """Project texel positions, run the z-buffer visibility + front-facing test.

    Shared by the mask and RGB accumulators so the facing-sign fix lives in ONE
    place (no second sign bug). Returns (gidx, w, vis, px_s, py_s):
      gidx  — global flat texel indices that survived (into (res*res) accumulators)
      w     — per-texel weight = vis * facing**gamma  (front-facing + unoccluded)
      vis   — per-texel visibility (0/1) for the coverage/exposure count
      px_s, py_s — screen-space pixel coords at which to sample the view's payload
    Returns (None, None, None, None, None) if no texel is inbound.
    """
    if valid_idx.size == 0:
        return None, None, None, None, None
    P = texel_pos[valid_idx]
    N = texel_norm[valid_idx]
    px, py, depth, in_front = project_points(view, P, lens, sensor_w, res_x, res_y)
    inbound = (in_front & (px >= 0) & (px < res_x) & (py >= 0) & (py < res_y))
    if not inbound.any():
        return None, None, None, None, None
    sel = np.where(inbound)[0]
    px_s = px[sel]; py_s = py[sel]; depth_s = depth[sel]
    P_s = P[sel]; N_s = N[sel]
    zbuf = bilinear_sample_points(depth_buf, px_s, py_s)
    vis = depth_s <= (zbuf + depth_eps)
    # view_dir points FROM the surface TOWARD the camera (cam_loc - P), so the
    # dot with the outward normal N is positive for front-facing texels. The
    # inverse (P - cam_loc) inverted the test and accumulated on back-facing
    # (occluded) texels only — vis*facing collapsed to ~1.8% of texels (only
    # silhouette), so the AI mask never landed on the surface and WearThreshold had
    # to extrapolate from noise → blocky/mosaic result. (Turn-5 fix.)
    view_dir = cam_loc[None, :] - P_s
    vn = view_dir / (np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-12)
    facing = np.clip(np.sum(N_s * vn, axis=1), 0.0, 1.0) ** gamma
    w = vis.astype(np.float32) * facing
    gidx = valid_idx[sel]
    return gidx, w, vis, px_s, py_s


def accumulate_view(acc_mask: np.ndarray, acc_weight: np.ndarray, count: np.ndarray,
                    texel_pos: np.ndarray, texel_norm: np.ndarray, valid_idx: np.ndarray,
                    view: np.ndarray, cam_loc: np.ndarray, lens: float, sensor_w: float,
                    res_x: int, res_y: int, depth_buf: np.ndarray, screen_mask: np.ndarray,
                    gamma: float, depth_eps: float) -> None:
    """Fold one view's mask into the surface field (in place)."""
    gidx, w, vis, px_s, py_s = _select_facing(
        texel_pos, texel_norm, valid_idx, view, cam_loc, lens, sensor_w,
        res_x, res_y, depth_buf, depth_eps, gamma)
    if gidx is None:
        return
    m = bilinear_sample_points(screen_mask, px_s, py_s)
    # scatter-add into the flat accumulation arrays (valid_idx[sel] → global)
    np.add.at(acc_mask, gidx, w * m)
    np.add.at(acc_weight, gidx, w)
    np.add.at(count, gidx, vis.astype(np.float32))


def accumulate_rgb_view(acc_rgb: np.ndarray, acc_rgb_w: np.ndarray,
                        texel_pos: np.ndarray, texel_norm: np.ndarray, valid_idx: np.ndarray,
                        view: np.ndarray, cam_loc: np.ndarray, lens: float, sensor_w: float,
                        res_x: int, res_y: int, depth_buf: np.ndarray,
                        screen_rgb: np.ndarray, gamma: float, depth_eps: float) -> None:
    """Fold one view's worn-image RGB into the UV-space texture (in place).

    Uses the SAME visibility/facing weighting as accumulate_view (via
    _select_facing) so the worn texture lands on exactly the texels the mask
    does — the overlay's texture data and its mask stay registered. acc_rgb is
    (res*res, 3) float32; acc_rgb_w is (res*res,) float32; screen_rgb is the
    encoded signed clean→worn color residual in LINEAR space, shape
    (res,res,3), where 0.5 is neutral.
    """
    gidx, w, _vis, px_s, py_s = _select_facing(
        texel_pos, texel_norm, valid_idx, view, cam_loc, lens, sensor_w,
        res_x, res_y, depth_buf, depth_eps, gamma)
    if gidx is None:
        return
    rgb = bilinear_sample_points(screen_rgb, px_s, py_s)  # (k, 3)
    np.add.at(acc_rgb, gidx, w[:, None] * rgb)
    np.add.at(acc_rgb_w, gidx, w)


def vertex_visibility(verts_world: np.ndarray, view: np.ndarray, cam_loc: np.ndarray,
                      lens: float, sensor_w: float, res_x: int, res_y: int,
                      depth_buf: np.ndarray, depth_eps: float) -> np.ndarray:
    """Per-vertex visibility (0/1) against a view's z-buffer (for exposure proxy)."""
    px, py, depth, in_front = project_points(view, verts_world, lens, sensor_w, res_x, res_y)
    inbound = (in_front & (px >= 0) & (px < res_x) & (py >= 0) & (py < res_y))
    vis = np.zeros(verts_world.shape[0], dtype=np.float32)
    sel = np.where(inbound)[0]
    if sel.size == 0:
        return vis
    zbuf = bilinear_sample_points(depth_buf, px[sel], py[sel])
    vis[sel] = (depth[sel] <= (zbuf + depth_eps)).astype(np.float32)
    return vis
