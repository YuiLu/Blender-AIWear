"""Confirm the facing-sign bug in accumulate_view: with view_dir = P - cam_loc
(current), front-facing texels get facing=0 and only ~1.8% of texels receive
mask signal. With view_dir = cam_loc - P (fixed), front-facing texels get
positive facing and coverage should jump to ~the visibility coverage.

Run: blender -b "blender/gaming_console_2k.blend/gaming_console_2k.blend" \
        --python tools/_probe_facing.py
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.getcwd())
import bpy
from ai_wear.uv.rasterizer import build_uv_field
from ai_wear.surface import projection
from ai_wear import utils

CACHE = os.path.join(os.getcwd(), "blender", "gaming_console_2k.blend",
                     ".ai_wear_cache", "gaming_console")
VIEWS = os.path.join(CACHE, "views")
with open(os.path.join(VIEWS, "views.json"), encoding="utf-8") as f:
    manifest = json.load(f)
RES = int(manifest["work_resolution"])
obj = bpy.data.objects["gaming_console"]
dg = bpy.context.evaluated_depsgraph_get()
uvf = build_uv_field(obj, "AI_WearUV", RES, depsgraph=dg)
texel_pos = uvf.reconstruct_positions().reshape(-1, 3)
texel_norm = uvf.reconstruct_normals().reshape(-1, 3)
valid_flat = uvf.valid.reshape(-1)
valid_idx = np.nonzero(valid_flat)[0]

rec = manifest["views"][0]  # V0
mask, _ = projection.extract_screen_mask(
    os.path.join(VIEWS, rec["clean"]), os.path.join(VIEWS, rec["worn"]), RES)
view = np.array(rec["view"], dtype=np.float32)
cam_loc = np.array(rec["cam_loc"], dtype=np.float32)
lens = float(rec["lens"]); sensor_w = float(rec["sensor_w"]); radius = float(rec["radius"])
depth_buf, cov = projection.rasterize_screen_depth(
    uvf.vpos, uvf.tri_vert, view, lens, sensor_w, RES, RES)
print(f"V0 depth coverage: {float(cov.mean()):.3f}")

# replicate accumulate_view's facing logic, both signs
P = texel_pos[valid_idx]
N = texel_norm[valid_idx]
px, py, depth, in_front = projection.project_points(view, P, lens, sensor_w, RES, RES)
inbound = (in_front & (px >= 0) & (px < RES) & (py >= 0) & (py < RES))
sel = np.where(inbound)[0]
P_s = P[sel]; N_s = N[sel]; depth_s = depth[sel]
zbuf = projection.bilinear_sample_points(depth_buf, px[sel], py[sel])
vis = depth_s <= (zbuf + max(radius * 0.02, 1e-3))
print(f"  inbound={inbound.sum()}  visible(vis>0)={int(vis.sum())} "
      f"({100.0*vis.sum()/max(1,inbound.sum()):.1f}% of inbound)")

for sign, label in ((+1, "BUGGY  view_dir=P-cam"), (-1, "FIXED  view_dir=cam-P")):
    view_dir = sign * (P_s - cam_loc[None, :])
    vn = view_dir / (np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-12)
    facing = np.clip(np.sum(N_s * vn, axis=1), 0.0, 1.0) ** 2.0
    w = vis.astype(np.float32) * facing
    nz_w = int((w > 1e-9).sum())
    print(f"  [{label}] facing>0 on {int((facing>1e-9).sum())}/{len(sel)} texels; "
          f"acc_weight nz={nz_w}/{len(valid_idx)} "
          f"({100.0*nz_w/len(valid_idx):.1f}%)")
