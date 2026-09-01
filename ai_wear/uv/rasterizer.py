"""UV rasterizer.

For a target UV layer at a working resolution, rasterize every UV triangle so
that each texel stores (triangle_id, barycentric). From barycentric we recover
3D world position / normal without per-texel ray_cast. The whole pass is NumPy
vectorized per-triangle.

This is the load-bearing piece of the surface field: correctness here decides
whether a screen-space mask lands on the right surface point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import utils


def _find_uv_layer(mesh, name: Optional[str]):
    if not mesh.uv_layers:
        return None
    if name:
        for l in mesh.uv_layers:
            if l.name == name:
                return l
        # A requested layer is part of the cache/projection contract. Falling
        # back to the active UV silently projects into one atlas and later asks
        # the shader to sample another (or a non-existent layer).
        return None
    return mesh.uv_layers.active


@dataclass
class UVField:
    res: int
    tri_id: np.ndarray      # (res,res) int32, -1 = empty
    bary: np.ndarray        # (res,res,3) float32
    valid: np.ndarray       # (res,res) bool
    tri_vert: np.ndarray    # (T,3) int32 vertex indices per loop-triangle
    vpos: np.ndarray        # (V,3) float32 world positions
    vnorm: np.ndarray       # (V,3) float32 world normals
    coverage: Optional[np.ndarray] = None  # (res,res) int16 (only if tracked)
    overlap_ratio: float = 0.0
    degenerate_count: int = 0
    flipped_ratio: float = 0.0

    # --- reconstruction ---------------------------------------------------

    def reconstruct_positions(self) -> np.ndarray:
        """World position per texel (res,res,3); empty where ~valid."""
        tids = np.where(self.tri_id < 0, 0, self.tri_id)
        verts = self.tri_vert[tids]  # (res,res,3)
        pos = self.vpos[verts]       # (res,res,3,3)
        out = (pos * self.bary[..., None]).sum(axis=2)  # (res,res,3)
        out[~self.valid] = 0
        return out

    def reconstruct_normals(self) -> np.ndarray:
        nrm = self.vnorm[self.tri_vert[np.where(self.tri_id < 0, 0, self.tri_id)]]
        out = (nrm * self.bary[..., None]).sum(axis=2)
        out /= (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-12)
        out[~self.valid] = 0
        return out

    def sample_field_at_uv(self, field: np.ndarray, uvs: np.ndarray) -> np.ndarray:
        """Bilinear-sample a (res,res) UV-domain field at given (N,2) uv coords."""
        x = np.clip(uvs[:, 0] * self.res, 0, self.res - 1.001)
        y = np.clip(uvs[:, 1] * self.res, 0, self.res - 1.001)
        x0 = np.floor(x).astype(np.int64); x1 = x0 + 1
        y0 = np.floor(y).astype(np.int64); y1 = y0 + 1
        fx = (x - x0).astype(np.float32); fy = (y - y0).astype(np.float32)
        a = field[y0, x0]; b = field[y0, x1]; c = field[y1, x0]; d = field[y1, x1]
        return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
                + c * (1 - fx) * fy + d * fx * fy)


def _edge(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def build_uv_field(obj, uv_layer_name: Optional[str], resolution: int,
                   depsgraph=None, track_coverage: bool = False) -> Optional[UVField]:
    """Rasterize the target UV layer. Returns None if the object has no UV."""
    import bpy
    from mathutils import Matrix

    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(dg)
    emesh = eval_obj.to_mesh()
    try:
        layer = _find_uv_layer(emesh, uv_layer_name)
        if layer is None:
            return None
        nv = len(emesh.vertices)
        if nv == 0:
            return None

        # World positions
        co = np.empty(nv * 3, dtype=np.float32)
        emesh.vertices.foreach_get("co", co)
        co = co.reshape(nv, 3)
        m44 = np.array(obj.matrix_world, dtype=np.float32)
        pos = co @ m44[:3, :3].T + m44[:3, 3]

        # World normals (transpose of inverse)
        no = np.empty(nv * 3, dtype=np.float32)
        emesh.vertices.foreach_get("normal", no)
        no = no.reshape(nv, 3)
        try:
            nmat = np.array(Matrix(obj.matrix_world).inverted().transposed().to_3x3(),
                            dtype=np.float32)
        except Exception:
            nmat = np.eye(3, dtype=np.float32)
        norm = no @ nmat.T
        norm /= (np.linalg.norm(norm, axis=1, keepdims=True) + 1e-12)

        # Loop triangles
        ntri = len(emesh.loop_triangles)
        if ntri == 0:
            return None
        tri_vert = np.empty(ntri * 3, dtype=np.int32)
        emesh.loop_triangles.foreach_get("vertices", tri_vert)
        tri_vert = tri_vert.reshape(ntri, 3)
        tri_loops = np.empty(ntri * 3, dtype=np.int32)
        emesh.loop_triangles.foreach_get("loops", tri_loops)
        tri_loops = tri_loops.reshape(ntri, 3)

        # UV per triangle
        nloops = len(emesh.loops)
        uv = np.empty(nloops * 2, dtype=np.float32)
        layer.data.foreach_get("uv", uv)
        uv = uv.reshape(nloops, 2)
        tri_uv = uv[tri_loops]  # (ntri,3,2)

        field = _rasterize(tri_uv, resolution, track_coverage)
        return UVField(
            res=resolution,
            tri_id=field["tri_id"],
            bary=field["bary"],
            valid=field["valid"],
            tri_vert=tri_vert,
            vpos=pos,
            vnorm=norm,
            coverage=field.get("coverage"),
            overlap_ratio=field["overlap_ratio"],
            degenerate_count=field["degenerate"],
            flipped_ratio=field["flipped_ratio"],
        )
    finally:
        eval_obj.to_mesh_clear()


def _rasterize(tri_uv: np.ndarray, res: int, track_coverage: bool) -> dict:
    ntri = tri_uv.shape[0]
    tri_id = np.full((res, res), -1, dtype=np.int32)
    bary = np.zeros((res, res, 3), dtype=np.float32)
    coverage = np.zeros((res, res), dtype=np.int16) if track_coverage else None
    degenerate = 0
    flipped = 0
    eps = 1e-6
    pix_eps = 1e-6

    for t in range(ntri):
        u0, v0 = tri_uv[t, 0]
        u1, v1 = tri_uv[t, 1]
        u2, v2 = tri_uv[t, 2]
        area = _edge(u0, v0, u1, v1, u2, v2)
        if abs(area) < 1e-12:
            degenerate += 1
            continue
        if area < 0:
            flipped += 1
        umin = min(u0, u1, u2); umax = max(u0, u1, u2)
        vmin = min(v0, v1, v2); vmax = max(v0, v1, v2)
        # skip tris entirely outside the 0..1 tile (clamped raster)
        if umax < 0 or umin > 1 or vmax < 0 or vmin > 1:
            continue
        x0 = max(0, int(np.floor(umin * res) - 1))
        x1 = min(res - 1, int(np.ceil(umax * res) + 1))
        y0 = max(0, int(np.floor(vmin * res) - 1))
        y1 = min(res - 1, int(np.ceil(vmax * res) + 1))
        if x1 < x0 or y1 < y0:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.int64)
        ys = np.arange(y0, y1 + 1, dtype=np.int64)
        gx, gy = np.meshgrid((xs + 0.5) / res, (ys + 0.5) / res)
        px = gx.ravel(); py = gy.ravel()
        w0 = _edge(u1, v1, u2, v2, px, py) / area
        w1 = _edge(u2, v2, u0, v0, px, py) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -pix_eps) & (w1 >= -pix_eps) & (w2 >= -pix_eps)
        if not inside.any():
            continue
        mask = inside.reshape(len(ys), len(xs))
        sub_id_region = tri_id[y0:y1 + 1, x0:x1 + 1]
        # last-writer-wins on overlap (Mode A shared wear, documented limitation)
        np.copyto(sub_id_region, np.where(mask, t, sub_id_region))
        sub_bary = bary[y0:y1 + 1, x0:x1 + 1]
        w0_2d = w0.reshape(len(ys), len(xs))
        w1_2d = w1.reshape(len(ys), len(xs))
        w2_2d = w2.reshape(len(ys), len(xs))
        sub_bary[..., 0] = np.where(mask, w0_2d.astype(np.float32), sub_bary[..., 0])
        sub_bary[..., 1] = np.where(mask, w1_2d.astype(np.float32), sub_bary[..., 1])
        sub_bary[..., 2] = np.where(mask, w2_2d.astype(np.float32), sub_bary[..., 2])
        if track_coverage:
            coverage[y0:y1 + 1, x0:x1 + 1] += mask.astype(np.int16)

    valid = tri_id >= 0
    cov = coverage if track_coverage else None
    overlap = 0.0
    if cov is not None and valid.any():
        overlap = float((cov[valid] > 1).sum()) / float(max(valid.sum(), 1))
    return {
        "tri_id": tri_id,
        "bary": bary,
        "valid": valid,
        "coverage": cov,
        "overlap_ratio": overlap,
        "degenerate": degenerate,
        "flipped_ratio": (flipped / ntri) if ntri else 0.0,
    }
