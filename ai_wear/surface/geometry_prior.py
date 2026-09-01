"""Geometry priors: signed convexity + exposure proxy + propensity combination.

Convexity uses a dihedral measure robust to winding: an edge is convex (ridge)
when its midpoint lies on the outward side of both adjacent face planes. This is
computed in object space (fine for the sign); per-vertex convexity is the mean
of incident edge values, in roughly [-1, 1].
"""

from __future__ import annotations

import numpy as np


def signed_convexity(obj) -> np.ndarray:
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    nv = len(bm.verts)

    face_n = {f.index: np.array(f.normal, dtype=np.float32) for f in bm.faces}
    face_c = {f.index: np.array(f.calc_center_median(), dtype=np.float32) for f in bm.faces}
    edge_val = {}
    for e in bm.edges:
        lf = e.link_faces
        if len(lf) != 2:
            continue
        n0 = face_n[lf[0].index]; n1 = face_n[lf[1].index]
        p0 = face_c[lf[0].index]; p1 = face_c[lf[1].index]
        mid = 0.5 * (np.array(e.verts[0].co, dtype=np.float32)
                     + np.array(e.verts[1].co, dtype=np.float32))
        d0 = float(np.dot(n0, mid - p0))
        d1 = float(np.dot(n1, mid - p1))
        mag = 1.0 - float(np.clip(np.dot(n0, n1), -1.0, 1.0))  # 0 flat → 2 opposite
        s = 1.0 if (d0 + d1) >= 0 else -1.0
        edge_val[e.index] = float(np.clip(s * mag * 0.5, -1.0, 1.0))

    vconv = np.zeros(nv, dtype=np.float32)
    vc = np.zeros(nv, dtype=np.float32)
    for e in bm.edges:
        v = edge_val.get(e.index)
        if v is None:
            continue
        for vert in e.verts:
            vconv[vert.index] += v
            vc[vert.index] += 1.0
    vconv /= np.maximum(vc, 1.0)
    bm.free()
    return vconv


def normalize_exposure(count: np.ndarray, num_views: int) -> np.ndarray:
    if num_views <= 0:
        return np.zeros_like(count, dtype=np.float32)
    return np.clip(count / float(num_views), 0.0, 1.0).astype(np.float32)


def compute_propensity(ai_vertex: np.ndarray, convexity: np.ndarray,
                       exposure: np.ndarray, weights: dict) -> np.ndarray:
    w_ai = weights.get("w_ai", 0.6)
    w_convex = weights.get("w_convex", 0.3)
    w_expose = weights.get("w_expose", 0.2)
    w_cavity = weights.get("w_cavity", 0.2)
    convex_pos = np.clip(convexity, 0.0, 1.0)
    cavity = np.clip(-convexity, 0.0, 1.0)
    p = (w_ai * ai_vertex + w_convex * convex_pos
         + w_expose * exposure - w_cavity * cavity)
    return np.clip(p, 0.0, 1.0).astype(np.float32)
