"""Print UV-seam statistics for every mesh in the currently opened .blend.

Run with Blender in background mode.  The script is read-only: it does not save
the .blend or modify mesh data.  Its final stdout line starts with
``AIWEAR_CANDIDATE_JSON=`` so batch callers can parse it without depending on
Blender's startup log.

Example (the asset may remain outside this repository):
  blender -b "<path-to-model.blend>" --python tools/analyze_seam_candidates.py
"""

from __future__ import annotations

import json

import bpy
import bmesh


def _uv_stats(obj, layer_name: str) -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.get(layer_name)
    if uv_layer is None:
        bm.free()
        return {}

    eps = 1e-5
    manifold = 0
    seams = 0
    boundary = 0
    face_adj = {face.index: [] for face in bm.faces}

    for edge in bm.edges:
        faces = edge.link_faces
        if len(faces) == 1:
            boundary += 1
            continue
        if len(faces) != 2:
            continue
        manifold += 1
        v0, v1 = edge.verts
        loops_a = {loop.vert: loop for loop in faces[0].loops}
        loops_b = {loop.vert: loop for loop in faces[1].loops}
        if not all(v in loops_a and v in loops_b for v in (v0, v1)):
            continue
        a0 = loops_a[v0][uv_layer].uv
        a1 = loops_a[v1][uv_layer].uv
        b0 = loops_b[v0][uv_layer].uv
        b1 = loops_b[v1][uv_layer].uv
        same = ((a0 - b0).length <= eps and (a1 - b1).length <= eps)
        reverse = ((a0 - b1).length <= eps and (a1 - b0).length <= eps)
        if same or reverse:
            face_adj[faces[0].index].append(faces[1].index)
            face_adj[faces[1].index].append(faces[0].index)
        else:
            seams += 1

    islands = 0
    unseen = set(face_adj)
    while unseen:
        islands += 1
        stack = [unseen.pop()]
        while stack:
            face_index = stack.pop()
            for neighbor in face_adj[face_index]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)

    result = {
        "layer": layer_name,
        "seam_edges": seams,
        "manifold_edges": manifold,
        "boundary_edges": boundary,
        "seam_ratio": seams / max(1, manifold),
        "uv_islands": islands,
    }
    bm.free()
    return result


report = {
    "blend": bpy.data.filepath,
    "objects": [],
}

for obj in bpy.context.scene.objects:
    if obj.type != "MESH" or len(obj.data.polygons) == 0:
        continue
    layers = [_uv_stats(obj, layer.name) for layer in obj.data.uv_layers]
    layers = [layer for layer in layers if layer]
    report["objects"].append({
        "name": obj.name,
        "vertices": len(obj.data.vertices),
        "edges": len(obj.data.edges),
        "polygons": len(obj.data.polygons),
        "materials": len(obj.material_slots),
        "uv_layers": layers,
        "has_ai_wear_cache_key": bool(obj.get("ai_wear_uuid")),
    })

print("AIWEAR_CANDIDATE_JSON=" + json.dumps(report, ensure_ascii=False))
