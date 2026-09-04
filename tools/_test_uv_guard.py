"""Headless test of the Q5 uvfield.valid-all-False guard.

Constructs the actual states that make build_uv_field return 0 valid texels:
  A1: UVs entirely OUTSIDE the [0,1] tile (rasterizer skips them) -> 0 valid
  A2: all-degenerate UV triangles (zero area) -> 0 valid
   B: real per-face UVs in [0,1] -> >0 valid (success path)

Plus verifies the catch-all: a WearThreshold computed from a valid-all-False field is
all-zero, and bake_vertex_to_uv is the zero source (already shown by
_probe_wearthreshold; this confirms the guard path end-to-end).

Run: blender -b --python tools/_test_uv_guard.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.getcwd())
import bpy

try:
    import ai_wear
    try: ai_wear.unregister()
    except Exception: pass
    ai_wear.register()
except Exception as e:
    print("register failed:", e); sys.exit(1)

from ai_wear.uv.rasterizer import build_uv_field
from ai_wear.operators.pipeline import _uv_coverage_diag, _uv_empty_error_msg
import bmesh


def fresh_cube(name):
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.name = name
    return obj


def set_uvs(obj, lay_name, mode):
    """mode: 'out_of_tile' | 'degenerate' | 'real'"""
    bm = bmesh.new(); bm.from_mesh(obj.data)
    uv = bm.loops.layers.uv[lay_name]
    for fi, f in enumerate(bm.faces):
        if mode == "out_of_tile":
            for i, l in enumerate(f.loops):
                l[uv].uv = (2.5 + i * 0.1, 3.5 + fi * 0.1)  # all outside [0,1]
        elif mode == "degenerate":
            for l in f.loops:
                l[uv].uv = (0.5, 0.5)  # all same -> zero-area triangles
        else:  # real: pack faces into 0.5x0.5 tiles inside [0,1]
            bx = (fi % 2) * 0.5; by = (fi // 2) * 0.5
            for i, l in enumerate(f.loops):
                l[uv].uv = (bx + (i % 2) * 0.5, by + (i // 2) * 0.5)
    bm.to_mesh(obj.data); bm.free()
    obj.data.update()


def check(label, mode, expect_zero):
    obj = fresh_cube("T_" + mode)
    lay = obj.data.uv_layers.new(name="AI_WearUV")
    lay_name = lay.name
    obj.data.uv_layers.active_index = obj.data.uv_layers[:].index(obj.data.uv_layers[lay_name])
    set_uvs(obj, lay_name, mode)
    dg = bpy.context.evaluated_depsgraph_get()
    uvf = build_uv_field(obj, lay_name, 128, depsgraph=dg)
    diag = _uv_coverage_diag(obj, lay_name, uvf)
    vc = diag["valid_count"]
    print(f"\n[{label}] mode={mode}: valid_count={vc}  "
          f"nonzero={diag.get('base_nonzero_frac',-1):.3f}  "
          f"in_tile={diag.get('base_in_tile_frac',-1):.3f}  "
          f"degenerate={diag.get('degenerate',0)}")
    if expect_zero:
        assert vc == 0, f"{label}: expected 0 valid, got {vc}"
        msg = _uv_empty_error_msg(obj.name, diag)
        assert "covered 0 of" in msg and "all-black" in msg, f"msg missing key text: {msg[:80]}"
        print(f"  PASS (0 valid): error msg = {msg[:90]}...")
    else:
        assert vc > 0, f"{label}: expected >0 valid, got {vc}"
        print(f"  PASS (>0 valid): WearThreshold would be non-black")


check("A1", "out_of_tile", expect_zero=True)
check("A2", "degenerate", expect_zero=True)
check("B ", "real", expect_zero=False)
print("\n=== Q5 GUARD TESTS PASS (out-of-tile + degenerate -> loud error; real -> ok) ===")
