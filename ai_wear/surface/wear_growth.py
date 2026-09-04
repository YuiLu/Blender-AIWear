"""WearThreshold topology growth.

Deterministic, no AI re-run. The AI field is transferred to vertices, combined
with convexity + exposure into a propensity; seeds are picked at high-propensity
local maxima; a multi-source Dijkstra over the vertex graph produces an arrival
distance that becomes the base WearThreshold; object-space 3D noise breaks it up
(continuous across UV seams because it samples 3D position, not UV). The vertex
field is then baked back to the target UV.

Wear Amount only thresholds T; changing 0..100 never re-requests the AI.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

import numpy as np


# --- UV ↔ vertex transfer ---------------------------------------------------

def transfer_uv_to_vertex(uvfield, field_uv: np.ndarray) -> np.ndarray:
    """Accumulate a UV-domain field onto vertices via barycentric weights (vectorized)."""
    nv = uvfield.vpos.shape[0]
    out = np.zeros(nv, dtype=np.float32)
    wsum = np.zeros(nv, dtype=np.float32)
    res = uvfield.res
    tri_id = uvfield.tri_id.ravel()
    valid = uvfield.valid.ravel()
    bary = uvfield.bary.reshape(res * res, 3)
    val = field_uv.ravel()
    tids = np.where(tri_id < 0, 0, tri_id)
    verts = uvfield.tri_vert[tids]  # (N,3)
    b0 = bary[:, 0]; b1 = bary[:, 1]; b2 = bary[:, 2]
    m = valid
    np.add.at(out, verts[m, 0], b0[m] * val[m]); np.add.at(wsum, verts[m, 0], b0[m])
    np.add.at(out, verts[m, 1], b1[m] * val[m]); np.add.at(wsum, verts[m, 1], b1[m])
    np.add.at(out, verts[m, 2], b2[m] * val[m]); np.add.at(wsum, verts[m, 2], b2[m])
    out /= np.maximum(wsum, 1e-9)
    return out


def bake_vertex_to_uv(uvfield, vertex_field: np.ndarray) -> np.ndarray:
    """Interpolate a per-vertex field back to the UV domain."""
    res = uvfield.res
    out = np.zeros((res, res), dtype=np.float32)
    valid = uvfield.valid
    tri_id = uvfield.tri_id
    tri_vert = uvfield.tri_vert
    bary = uvfield.bary
    tids = np.where(tri_id < 0, 0, tri_id)
    verts = tri_vert[tids]  # (res,res,3)
    vals = vertex_field[verts]  # (res,res,3)
    out = (vals * bary).sum(axis=-1)
    out[~valid] = 0.0
    return out.astype(np.float32)


# --- topology graph ----------------------------------------------------------

def build_topology_graph(obj) -> Tuple[dict, np.ndarray]:
    """Adjacency {v: [(neighbor, edge_length, material_boundary)]} + world positions."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    M = np.array(obj.matrix_world, dtype=np.float32)
    nv = len(bm.verts)
    pos = np.empty(nv * 3, dtype=np.float32)
    # foreach_get is a Blender RNA API (mesh.vertices), NOT bmesh — BMVertSeq has
    # no foreach_get. bmesh is used here only for topology (edges/adjacency); read
    # positions from the RNA mesh. Indices match (from_mesh preserves them).
    obj.data.vertices.foreach_get("co", pos)
    pos = pos.reshape(nv, 3)
    world = pos @ M[:3, :3].T + M[:3, 3]

    adj: Dict[int, List[Tuple[int, float, bool]]] = {i: [] for i in range(nv)}
    for e in bm.edges:
        a, b = e.verts[0].index, e.verts[1].index
        length = float(e.calc_length())
        boundary = False
        if len(e.link_faces) == 2:
            m0 = e.link_faces[0].material_index
            m1 = e.link_faces[1].material_index
            boundary = (m0 != m1)
        adj[a].append((b, length, boundary))
        adj[b].append((a, length, boundary))
    bm.free()
    return adj, world


# --- seeds ------------------------------------------------------------------

def select_seeds(P: np.ndarray, world: np.ndarray, adj: dict,
                 threshold: float = 0.2, min_dist_frac: float = 0.10,
                 radius: float = 1.0) -> List[int]:
    """Local maxima of P (≥ neighbors) above threshold, suppressed by min distance."""
    cand = []
    for v in range(len(P)):
        if P[v] < threshold:
            continue
        nb = adj[v]
        if not nb:
            continue
        if all(P[v] >= P[n[0]] for n in nb):
            cand.append(v)
    if not cand:
        # fallback: top-k by P
        cand = list(np.argsort(P)[-max(1, len(P) // 500):])
    cand.sort(key=lambda v: -P[v])
    min_dist = max(radius * min_dist_frac, 1e-4)
    kept: List[int] = []
    kept_pos: List[np.ndarray] = []
    for v in cand:
        p = world[v]
        if all(np.linalg.norm(p - kp) > min_dist for kp in kept_pos):
            kept.append(v)
            kept_pos.append(p)
    return kept if kept else [cand[0]]


# --- multi-source Dijkstra ---------------------------------------------------

def edge_cost(length: float, boundary: bool, Pi: float, Pj: float,
              gamma: float, mat_penalty: float, use_barrier: bool) -> float:
    mean_p = max((Pi + Pj) * 0.5, 1e-3)
    base = length / (1e-4 + mean_p ** gamma)
    if use_barrier and boundary:
        return base * mat_penalty
    return base


def multi_source_dijkstra(adj: dict, P: np.ndarray, seeds: List[int],
                          gamma: float, mat_penalty: float,
                          use_barrier: bool) -> np.ndarray:
    nv = len(P)
    dist = np.full(nv, np.inf, dtype=np.float64)
    pq = []
    for s in seeds:
        dist[s] = 0.0
        heapq.heappush(pq, (0.0, s))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for (v, length, boundary) in adj[u]:
            c = edge_cost(length, boundary, P[u], P[v], gamma, mat_penalty, use_barrier)
            nd = d + c
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    # unreachable verts: set to max finite distance
    finite = dist[np.isfinite(dist)]
    fill = finite.max() if finite.size else 1.0
    dist[~np.isfinite(dist)] = fill
    return dist.astype(np.float32)


# --- 3D noise ----------------------------------------------------------------

def _hash3(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    h = (ix * 374761393 + iy * 668265263 + iz * 1013904223 + seed * 913) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) & 0xFFFFFFFF
    h = (h * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFFFF).astype(np.float64) / 4294967295.0


def value_noise_3d(pos: np.ndarray, scale: float, seed: int) -> np.ndarray:
    p = pos.astype(np.float64) * scale
    i = np.floor(p).astype(np.int64)
    f = p - i
    # smoothstep
    u = f * f * (3.0 - 2.0 * f)
    ix = i[:, 0]; iy = i[:, 1]; iz = i[:, 2]
    fx = u[:, 0]; fy = u[:, 1]; fz = u[:, 2]

    c000 = _hash3(ix, iy, iz, seed)
    c100 = _hash3(ix + 1, iy, iz, seed)
    c010 = _hash3(ix, iy + 1, iz, seed)
    c110 = _hash3(ix + 1, iy + 1, iz, seed)
    c001 = _hash3(ix, iy, iz + 1, seed)
    c101 = _hash3(ix + 1, iy, iz + 1, seed)
    c011 = _hash3(ix, iy + 1, iz + 1, seed)
    c111 = _hash3(ix + 1, iy + 1, iz + 1, seed)

    x00 = c000 * (1 - fx) + c100 * fx
    x10 = c010 * (1 - fx) + c110 * fx
    x01 = c001 * (1 - fx) + c101 * fx
    x11 = c011 * (1 - fx) + c111 * fx
    y0 = x00 * (1 - fy) + x10 * fy
    y1 = x01 * (1 - fy) + x11 * fy
    return (y0 * (1 - fz) + y1 * fz).astype(np.float32)


# --- assembly ---------------------------------------------------------------

def build_wearthreshold_from_graph(uvfield, ai_field_uv: np.ndarray, convexity: np.ndarray,
                              exposure: np.ndarray, adj: dict, world: np.ndarray,
                              weights: dict, gamma: float, alpha: float,
                              noise_amp: float, noise_scale: float, use_barrier: bool,
                              mat_penalty: float, seed: int, radius: float) -> dict:
    """Numpy-only WearThreshold core (graph + convexity supplied). Runs on worker thread."""
    from .geometry_prior import compute_propensity

    ai_vertex = transfer_uv_to_vertex(uvfield, ai_field_uv)
    P = compute_propensity(ai_vertex, convexity, exposure, weights)

    seeds = select_seeds(P, world, adj, threshold=0.18,
                         min_dist_frac=0.10, radius=radius)
    if not seeds:
        seeds = [int(np.argmax(P))]

    dist = multi_source_dijkstra(adj, P, seeds, gamma, mat_penalty, use_barrier)
    dmax = float(dist.max())
    T_base = (dist / (dmax + 1e-9)).astype(np.float32) if dmax > 0 else np.zeros_like(dist)

    noise = value_noise_3d(world, noise_scale, int(seed))
    noise_centered = (noise * 2.0 - 1.0)

    T = alpha * T_base + (1.0 - alpha) * (1.0 - P) + noise_amp * noise_centered
    T = np.clip(T, 0.0, 1.0).astype(np.float32)
    T = _smooth_field(T, adj, iterations=2)

    wearthreshold_uv = bake_vertex_to_uv(uvfield, T)
    return {
        "wearthreshold_vertex": T,
        "wearthreshold_uv": wearthreshold_uv,
        "propensity": P,
        "ai_vertex": ai_vertex,
        "seeds": seeds,
        "arrival_distance": dist,
    }


def build_direct_wearthreshold(uvfield, ai_field_uv: np.ndarray, convexity: np.ndarray,
                          exposure: np.ndarray, weights: dict,
                          noise_amp: float, noise_scale: float,
                          seed: int, world: np.ndarray) -> dict:
    """Ablation path without Dijkstra/topology propagation.

    Propensity is evaluated at vertices and converted directly to arrival time.
    The return schema matches ``build_wearthreshold_from_graph`` so every downstream
    bake/shader/export stage remains identical.
    """
    from .geometry_prior import compute_propensity

    ai_vertex = transfer_uv_to_vertex(uvfield, ai_field_uv)
    propensity = compute_propensity(ai_vertex, convexity, exposure, weights)
    noise = value_noise_3d(world, noise_scale, int(seed)) * 2.0 - 1.0
    vertex_time = np.clip(1.0 - propensity + noise_amp * noise, 0.0, 1.0).astype(np.float32)
    return {
        "wearthreshold_vertex": vertex_time,
        "wearthreshold_uv": bake_vertex_to_uv(uvfield, vertex_time),
        "propensity": propensity,
        "ai_vertex": ai_vertex,
        "seeds": [],
        "arrival_distance": np.zeros_like(vertex_time),
    }


def build_wearthreshold(obj, uvfield, ai_field_uv: np.ndarray, convexity: np.ndarray,
                   exposure: np.ndarray, weights: dict, gamma: float, alpha: float,
                   noise_amp: float, noise_scale: float, use_barrier: bool,
                   mat_penalty: float, seed: int, radius: float) -> dict:
    """Full WearThreshold build (builds topology graph internally). Returns vertex + UV WearThreshold."""
    adj, world = build_topology_graph(obj)
    return build_wearthreshold_from_graph(
        uvfield, ai_field_uv, convexity, exposure, adj, world,
        weights, gamma, alpha, noise_amp, noise_scale, use_barrier,
        mat_penalty, seed, radius)


def _smooth_field(field: np.ndarray, adj: dict, iterations: int = 2) -> np.ndarray:
    out = field.copy()
    for _ in range(iterations):
        new = out.copy()
        for v in range(len(out)):
            nb = adj[v]
            if nb:
                s = sum(out[n[0]] for n in nb)
                new[v] = 0.5 * out[v] + 0.5 * (s / len(nb))
        out = new
    return out
