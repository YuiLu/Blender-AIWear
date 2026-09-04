"""Headless diagnostic for the gaming_console WearThreshold-all-black case.

Reproduces the EXACT downstream (mask -> projection -> fusion -> WearThreshold -> bake)
on the real gaming_console object from its saved .blend, using the CACHED per-view
clean/worn images + the SAVED camera matrices in views.json. No render, no AI.

Dumps stats at every stage so we can see WHERE the field goes to zero, and saves:
  mask_V{i}.png     - the diff'd screen mask per view (the user's request)
  ai_field.png      - the fused multi-view field (what gets projected to vertices)
  wearthreshold_recomputed.png - WearThreshold as recomputed NOW (PNG16, in-memory values)
  ondisk_stats      - min/max/mean/nz of the existing WearThreshold.png on disk
  save_roundtrip    - proves save_image preserves non-zero (rules out encode bug)

Run:
  blender -b "blender/gaming_console_2k.blend/gaming_console_2k.blend" \
    --python tools/_diag_gaming_console.py
"""
import sys, os, json
import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
import bpy

# --- register so the package + its bpy-dependent helpers are initialized ---
try:
    import ai_wear
    try:
        ai_wear.unregister()
    except Exception:
        pass
    ai_wear.register()
except Exception as e:
    print("REGISTER NOTE:", e)

from ai_wear.uv.rasterizer import build_uv_field
from ai_wear.operators.pipeline import _uv_coverage_diag
from ai_wear.surface import projection, fusion, geometry_prior, wear_growth
from ai_wear import utils

CACHE = os.path.join(ROOT, "blender", "gaming_console_2k.blend",
                     ".ai_wear_cache", "gaming_console")
VIEWS_DIR = os.path.join(CACHE, "views")
DEBUG_DIR = os.path.join(CACHE, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)


def stats(a, name):
    a = np.asarray(a, dtype=np.float64).ravel()
    nz = int((np.abs(a) > 1e-9).sum())
    print(f"  {name:24s} min={a.min():.6g}  max={a.max():.6g}  "
          f"mean={a.mean():.6g}  nz={nz}/{a.size} ({100.0*nz/max(1,a.size):.2f}%)")
    return dict(min=float(a.min()), max=float(a.max()),
                mean=float(a.mean()), nz=nz, n=a.size)


def save_dbg(arr, name):
    p = os.path.join(DEBUG_DIR, name)
    try:
        utils.save_image(p, arr, "PNG16")
        print(f"  saved {p}")
    except Exception as e:
        print(f"  SAVE FAIL {name}: {e}")


print("\n========== SCENE OBJECTS ==========")
mesh_objs = [o.name for o in bpy.data.objects if o.type == "MESH"]
print("mesh objects:", mesh_objs)
cam_objs = [o.name for o in bpy.data.objects if o.type == "CAMERA"]
print("camera objects:", cam_objs)

OBJ_NAME = "gaming_console"
obj = bpy.data.objects.get(OBJ_NAME)
if obj is None:
    # fall back to any mesh that has the AI_WearUV layer
    for n in mesh_objs:
        o = bpy.data.objects[n]
        if o.data and o.data.uv_layers.get("AI_WearUV"):
            OBJ_NAME, obj = n, o
            break
if obj is None and mesh_objs:
    OBJ_NAME, obj = mesh_objs[0], bpy.data.objects[mesh_objs[0]]
assert obj is not None, "no mesh object found"
print(f"\nTARGET OBJECT: {OBJ_NAME}  verts={len(obj.data.vertices)}  "
      f"polys={len(obj.data.polygons)}  edges={len(obj.data.edges)}")
print("uv layers:", [l.name for l in obj.data.uv_layers])

# load the saved manifest
with open(os.path.join(VIEWS_DIR, "views.json"), encoding="utf-8") as f:
    manifest = json.load(f)
LAYER = manifest.get("layer", "AI_WearUV")
RES = int(manifest.get("work_resolution", 1024))
view_records = manifest["views"]
print(f"views.json: layer={LAYER} res={RES} n_views={len(view_records)}")

# --- 1. UV field + coverage diagnostic (does my guard's condition apply?) ---
print("\n========== 1. UV FIELD ==========")
dg = bpy.context.evaluated_depsgraph_get()
uvfield = build_uv_field(obj, LAYER, RES, depsgraph=dg)
if uvfield is None:
    print("!! build_uv_field returned None — layer not found")
    sys.exit(1)
diag = _uv_coverage_diag(obj, LAYER, uvfield)
print(f"  valid_count = {diag['valid_count']} / {diag['total']}  "
      f"coverage = {diag['coverage_pct']:.2f}%")
print(f"  in_tile_frac = {diag.get('base_in_tile_frac', -1):.4f}  "
      f"nonzero_frac = {diag.get('base_nonzero_frac', -1):.4f}  "
      f"degenerate = {diag.get('degenerate', 0)}  "
      f"flipped = {diag.get('flipped_ratio', -1):.4f}")
vcov = float(uvfield.valid.mean())
print(f"  uvfield.valid.mean() = {vcov:.6f}  "
      f"(<- if 0.0, bake_vertex_to_uv zeroes EVERYTHING => WearThreshold all-black)")

texel_pos = uvfield.reconstruct_positions().reshape(-1, 3)
texel_norm = uvfield.reconstruct_normals().reshape(-1, 3)
valid_flat = uvfield.valid.reshape(-1)
valid_idx = np.nonzero(valid_flat)[0]
acc_mask = np.zeros(RES * RES, dtype=np.float32)
acc_w = np.zeros(RES * RES, dtype=np.float32)
count = np.zeros(RES * RES, dtype=np.float32)
acc_rgb = np.zeros((RES * RES, 3), dtype=np.float32)
acc_rgb_w = np.zeros(RES * RES, dtype=np.float32)
exposure_count = np.zeros(len(uvfield.vpos), dtype=np.float32)

# --- 2. per-view: mask extraction (+SAVE for user) + depth + accumulate ---
print("\n========== 2. PER-VIEW MASK + PROJECTION ==========")
for i, rec in enumerate(view_records):
    cp = os.path.join(VIEWS_DIR, rec["clean"])
    wp = os.path.join(VIEWS_DIR, rec["worn"])
    mask, conf, worn_rgb = projection.extract_screen_mask(cp, wp, RES, return_worn_rgb=True)
    mstats = stats(mask, f"mask_V{i}")
    save_dbg(np.clip(mask, 0, 1), f"mask_V{i}.png")
    view = np.array(rec["view"], dtype=np.float32)
    cam_loc = np.array(rec["cam_loc"], dtype=np.float32)
    lens = float(rec["lens"]); sensor_w = float(rec["sensor_w"])
    radius = float(rec["radius"])
    depth_buf, cov = projection.rasterize_screen_depth(
        uvfield.vpos, uvfield.tri_vert, view, lens, sensor_w, RES, RES)
    print(f"  V{i}: depth_cov={float(cov.mean()):.3f}  "
          f"depth[min/med/max]={float(np.median(depth_buf[cov])):.4g}/"
          f"{float(depth_buf[cov].min()):.4g}/{float(depth_buf[cov].max()):.4g}  "
          f"mask_conf={conf:.3f}")
    projection.accumulate_view(
        acc_mask, acc_w, count, texel_pos, texel_norm, valid_idx,
        view, cam_loc, lens, sensor_w, RES, RES, depth_buf, mask,
        2.0, depth_eps=max(radius * 0.02, 1e-3))
    projection.accumulate_rgb_view(
        acc_rgb, acc_rgb_w, texel_pos, texel_norm, valid_idx,
        view, cam_loc, lens, sensor_w, RES, RES, depth_buf, worn_rgb,
        2.0, depth_eps=max(radius * 0.02, 1e-3))
    exposure_count += projection.vertex_visibility(
        uvfield.vpos, view, cam_loc, lens, sensor_w, RES, RES,
        depth_buf, max(radius * 0.02, 1e-3))

stats(acc_mask, "acc_mask(raw)")
stats(acc_w, "acc_weight")
stats(count, "count(views-hit)")

# --- 3. fusion ---
print("\n========== 3. FUSION ==========")
fused = fusion.finalize_ai_field(acc_mask, acc_w, count)
ai_field = fused["ai_field"]
stats(ai_field, "ai_field(fused)")
print(f"  coverage_ratio = {fused['coverage_ratio']:.4f}  "
      f"texel_cov = {fused['texel_coverage_ratio']:.4f}")
save_dbg(np.clip(ai_field, 0, 1), "ai_field.png")

# --- 3b. worn-texture fusion (RGB) ---
fused_rgb = fusion.finalize_rgb_field(acc_rgb, acc_rgb_w)
worn_uv = fused_rgb["rgb"]
rgb_valid = fused_rgb["valid"]
stats(worn_uv[..., 0], "worn_uv R")
stats(worn_uv[..., 1], "worn_uv G")
stats(worn_uv[..., 2], "worn_uv B")
print(f"  worn coverage_ratio = {fused_rgb['coverage_ratio']:.4f}  "
      f"(vs ai_field texel_cov {fused['texel_coverage_ratio']:.4f})")
save_dbg(worn_uv, "worn_uv.png")

# --- 4. geometry priors + topology ---
print("\n========== 4. GEOMETRY + TOPOLOGY ==========")
convexity = geometry_prior.signed_convexity(obj)
stats(convexity, "convexity")
exposure = geometry_prior.normalize_exposure(exposure_count, len(view_records))
stats(exposure, "exposure")
adj, world = wear_growth.build_topology_graph(obj)
print(f"  topology: {len(adj)} verts, "
      f"{sum(len(v) for v in adj.values())//2} edges")
n_components_msg = ""
# connectivity check: how many verts reachable from vert 0?
seen = set([0])
stack = [0]
while stack:
    u = stack.pop()
    for (v, _l, _b) in adj[u]:
        if v not in seen:
            seen.add(v); stack.append(v)
print(f"  graph connectivity: {len(seen)}/{len(adj)} verts reachable from v0 "
      f"({'CONNECTED' if len(seen)==len(adj) else 'DISCONNECTED — see below'})")

# --- 5. WearThreshold growth (mirrors build_wearthreshold_from_graph, staged) ---
print("\n========== 5. WEARTHRESHOLD GROWTH (staged) ==========")
weights = dict(w_ai=0.6, w_convex=0.3, w_expose=0.2, w_cavity=0.2)
gamma = 2.0; alpha = 0.7
noise_amp = 0.12; noise_scale = 8.0
use_barrier = True; mat_penalty = 4.0
seed = 0  # default: seed=0, lock_seed=True -> base_seed=0
radius_wt = float(np.linalg.norm(world.max(0) - world.min(0)))
print(f"  params: alpha={alpha} noise_amp={noise_amp} noise_scale={noise_scale} "
      f"gamma={gamma} mat_penalty={mat_penalty} use_barrier={use_barrier} "
      f"seed={seed} radius_wt={radius_wt:.4f}")

ai_vertex = wear_growth.transfer_uv_to_vertex(uvfield, ai_field)
stats(ai_vertex, "ai_vertex")
P = geometry_prior.compute_propensity(ai_vertex, convexity, exposure, weights)
stats(P, "propensity P")
seeds = wear_growth.select_seeds(P, world, adj, threshold=0.18,
                                 min_dist_frac=0.10, radius=radius_wt)
print(f"  seeds: {len(seeds)} verts  (first few: {seeds[:8]})")
print(f"  P at seeds: {[float(P[s]) for s in seeds[:8]]}")
dist = wear_growth.multi_source_dijkstra(adj, P, seeds, gamma, mat_penalty, use_barrier)
stats(dist, "dijkstra dist")
dmax = float(dist.max())
T_base = (dist / (dmax + 1e-9)).astype(np.float32) if dmax > 0 else np.zeros_like(dist)
stats(T_base, "T_base=dist/dmax")
noise = wear_growth.value_noise_3d(world, noise_scale, int(seed))
noise_centered = (noise * 2.0 - 1.0)
stats(noise_centered, "noise_centered")
T = alpha * T_base + (1.0 - alpha) * (1.0 - P) + noise_amp * noise_centered
T = np.clip(T, 0.0, 1.0).astype(np.float32)
stats(T, "T(vertex) pre-smooth")
T = wear_growth._smooth_field(T, adj, iterations=2)
stats(T, "T(vertex) post-smooth")

wearthreshold_uv = wear_growth.bake_vertex_to_uv(uvfield, T)
stats(wearthreshold_uv, "wearthreshold_uv(baked)")
save_dbg(np.clip(wearthreshold_uv, 0, 1), "wearthreshold_recomputed.png")

# the catch-all condition
print(f"\n  >>> catch-all `not wearthreshold_uv.any()` = {not bool(wearthreshold_uv.any())} "
      f"(if True, the all-zero guard WOULD have fired)")

# --- 6. compare to on-disk WearThreshold.png ---
print("\n========== 6. ON-DISK WearThreshold.png ==========")
on_disk = os.path.join(CACHE, "WearThreshold.png")
if os.path.isfile(on_disk):
    arr = utils.load_image_rgba(on_disk)
    g = arr[..., :3].mean(-1)
    stats(g, "WearThreshold.png(disk)")
    print(f"  file size = {os.path.getsize(on_disk)} bytes")
    print(f"  visually-black? {g.max() < 0.01}  truly-zero? {g.max() < 1e-6}")
else:
    print("  no WearThreshold.png on disk")

# --- 7. save_image roundtrip: prove encode preserves non-zero ---
print("\n========== 7. save_image ROUNDTRIP (encode sanity) ==========")
probe = np.full((RES, RES, 4), 0.0, dtype=np.float32)
probe[..., 0] = 0.0  # left half black
probe[..., 1] = 0.0
probe[..., 2] = 0.0
probe[::4, ::4, 0] = 0.5  # sparse non-zero dots
probe[::4, ::4, 3] = 1.0
probe_p = os.path.join(DEBUG_DIR, "save_probe.png")
utils.save_image(probe_p, probe, "PNG16")
# Reload as Non-Color (identity) so we test the ENCODER, not the colorspace
# decode. load_image_rgba() defaults to sRGB, which decodes 0.5 -> 0.214 at
# load (Blender decodes sRGB->linear) — that is expected, NOT a save bug.
pimg = bpy.data.images.load(probe_p, check_existing=False)
pimg.colorspace_settings.name = "Non-Color"
pbuf = np.empty(len(pimg.pixels), dtype=np.float32)
pimg.pixels.foreach_get(pbuf)
pa = pbuf.reshape(pimg.size[1], pimg.size[0], 4)[::-1]
print(f"  PNG16 preserves non-zero 0.5 -> {float(pa[::4,::4,0].max()):.4f} "
      f"(encode is {'OK' if pa[::4,::4,0].max() > 0.4 else 'BROKEN'})")
bpy.data.images.remove(pimg)

# --- 8. write the regenerated WearThreshold to the CANONICAL path + verify ---
print("\n========== 8. WRITE CANONICAL WearThreshold.png + VERIFY ==========")
wearthreshold_path = os.path.join(CACHE, "WearThreshold.png")
utils.save_image(wearthreshold_path, np.clip(wearthreshold_uv, 0, 1), "PNG16")
print(f"  wrote {wearthreshold_path} ({os.path.getsize(wearthreshold_path)} bytes)")
# reload as Non-Color (the way the shader reads it) and check channels
vimg = bpy.data.images.load(wearthreshold_path, check_existing=False)
vimg.colorspace_settings.name = "Non-Color"
vbuf = np.empty(len(vimg.pixels), dtype=np.float32)
vimg.pixels.foreach_get(vbuf)
va = vbuf.reshape(vimg.size[1], vimg.size[0], 4)[::-1]
print(f"  canonical WearThreshold reload (Non-Color): "
      f"R[{va[...,0].min():.4f},{va[...,0].max():.4f}] "
      f"mean={va[...,0].mean():.4f} nz={int((va[...,0]>1e-6).sum())}/{va[...,0].size}")
print(f"  in-memory wearthreshold_uv: min={wearthreshold_uv.min():.4f} max={wearthreshold_uv.max():.4f} "
      f"mean={wearthreshold_uv.mean():.4f}")
print(f"  >>> WearThreshold is {'NON-BLACK (FIXED)' if va[...,0].max() > 0.05 else 'STILL BLACK'}")
bpy.data.images.remove(vimg)

# verify the diff masks the user wanted are non-black now
print("\n  --- diff masks (user preview) channel check ---")
for i in range(6):
    mp = os.path.join(DEBUG_DIR, f"mask_V{i}.png")
    mi = bpy.data.images.load(mp, check_existing=False)
    mi.colorspace_settings.name = "Non-Color"
    mbuf = np.empty(len(mi.pixels), dtype=np.float32)
    mi.pixels.foreach_get(mbuf)
    ma = mbuf.reshape(mi.size[1], mi.size[0], 4)[::-1]
    print(f"  mask_V{i}: R[{ma[...,0].min():.4f},{ma[...,0].max():.4f}] "
          f"mean={ma[...,0].mean():.4f} -> {'OK' if ma[...,0].max() > 0.05 else 'BLACK'}")
    bpy.data.images.remove(mi)

# --- 9. write the NEW canonical textures (reprojected mask + worn texture) ---
print("\n========== 9. CANONICAL M_Wear.png + AIWear_WornTex.png ==========")
m_wear_path = os.path.join(CACHE, "M_Wear.png")
mask_rgba = np.zeros((RES, RES, 4), dtype=np.float32)
mask_rgba[..., 0] = mask_rgba[..., 1] = mask_rgba[..., 2] = np.clip(ai_field, 0, 1)
mask_rgba[..., 3] = 1.0
utils.save_image(m_wear_path, mask_rgba, "PNG16")
print(f"  wrote {m_wear_path} ({os.path.getsize(m_wear_path)} bytes)")

worn_tex_path = os.path.join(CACHE, "AIWear_WornTex.png")
wtex = np.zeros((RES, RES, 4), dtype=np.float32)
wtex[..., :3] = worn_uv
wtex[..., 3] = rgb_valid.astype(np.float32)
utils.save_image(worn_tex_path, wtex, "PNG16")
print(f"  wrote {worn_tex_path} ({os.path.getsize(worn_tex_path)} bytes)")

# round-trip: reload each the way the shader reads them, compare to in-memory
mimg = bpy.data.images.load(m_wear_path, check_existing=False)
mimg.colorspace_settings.name = "Non-Color"
mbuf = np.empty(len(mimg.pixels), dtype=np.float32)
mimg.pixels.foreach_get(mbuf)
ma2 = mbuf.reshape(mimg.size[1], mimg.size[0], 4)[::-1]
rmse_mask = float(np.sqrt(np.mean((ma2[..., 0] - np.clip(ai_field, 0, 1).astype(np.float32)) ** 2)))
print(f"  M_Wear.png round-trip (Non-Color): R mean in-mem={ai_field.mean():.4f} "
      f"reloaded={ma2[...,0].mean():.4f} rmse={rmse_mask:.5f} -> "
      f"{'OK' if rmse_mask < 0.01 else 'MISMATCH'}")
bpy.data.images.remove(mimg)

wimg = bpy.data.images.load(worn_tex_path, check_existing=False)
wimg.colorspace_settings.name = "Non-Color"
wbuf = np.empty(len(wimg.pixels), dtype=np.float32)
wimg.pixels.foreach_get(wbuf)
wa = wbuf.reshape(wimg.size[1], wimg.size[0], 4)[::-1]
rmse_rgb = float(np.sqrt(np.mean((wa[..., :3] - worn_uv.astype(np.float32)) ** 2)))
alpha_diff = float(np.abs(wa[..., 3] - rgb_valid.astype(np.float32)).mean())
print(f"  AIWear_WornTex.png round-trip (Non-Color): RGB rmse={rmse_rgb:.5f} "
      f"alpha mean-diff={alpha_diff:.5f} -> "
      f"{'OK' if rmse_rgb < 0.02 and alpha_diff < 0.01 else 'CHECK COLORSPACE'}")
bpy.data.images.remove(wimg)

print("\n========== DONE ==========")
print(f"debug PNGs in: {DEBUG_DIR}")
