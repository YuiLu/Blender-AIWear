"""Headless verification of the four fixes. No AI, no network.

Runs under `blender -b --python` and prints PASS/FAIL lines:
  1. ai_wear registers; `texture_size` property is gone.
  2. fuse_seam symmetrically reconciles a narrow paired strip.
  3. passes.render_clean renders three lighting modes to distinct images.
"""
import os
import sys
from pathlib import Path

import bpy
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ai_wear
try:
    ai_wear.unregister()
except Exception:
    pass
ai_wear.register()

from ai_wear.render import passes, oblique
from ai_wear.uv.seam_registry import SeamPair, fuse_seam, seam_qa

fails = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + str(detail)) if detail else ""))
    if not ok:
        fails.append(name)


# --- 1. registration + property removal -------------------------------------
s = bpy.context.scene.ai_wear
check("texture_size property removed", not hasattr(s, "texture_size"))
check("work_resolution still present", hasattr(s, "work_resolution"))
check("safe padding default", s.padding_texels == 2, s.padding_texels)
check("render_clean accepts lighting kwarg",
      "lighting" in passes.render_clean.__code__.co_varnames)
check("fuse_seam accepts tol kwarg", "tol" in fuse_seam.__code__.co_varnames)

# --- 2. synthetic seam fusion -----------------------------------------------
res = 64
# Side A UV x∈[0.20,0.30] → cols 12.8..19.2; side B x∈[0.60,0.70] → cols 38.4..44.8.
# Fill the FULL span so bilinear sampling is uniform along the whole segment.
field = np.zeros((res, res), dtype=np.float32)
field[30:35, 12:20] = 0.8   # side A (high wear)
field[30:35, 38:46] = 0.2   # side B (low wear)
sp = SeamPair(0, 0, 1,
              np.array([0.20, 0.5]), np.array([0.30, 0.5]),
              np.array([0.60, 0.5]), np.array([0.70, 0.5]),
              np.array([0.0, 1.0]), np.array([0.0, 1.0]))
reg = [sp]

before = seam_qa(field, reg, res)
fused = fuse_seam(field, reg, res, diffuse_texels=8)
after = seam_qa(fused, reg, res)

fused_a = float(fused[32, 16])
fused_b = float(fused[32, 41])
check("fusion reduces seam diff", after["p95"] < before["p95"],
      f"p95 {before['p95']:.4f} -> {after['p95']:.4f}")
check("fusion is symmetric (no max-biased bright line)",
      abs(fused_a - fused_b) < 0.15 and fused_a < 0.75 and fused_b > 0.25,
      f"sideA {fused_a:.3f}, sideB {fused_b:.3f}")

# gate: a second field where sides already agree must NOT be smeared
field2 = np.zeros((res, res), dtype=np.float32)
# Include a one-texel guard around both segments so endpoint bilinear samples
# see the same constant signal on both islands.
field2[30:35, 11:21] = 0.8
field2[30:35, 37:47] = 0.8   # already continuous
sp2 = SeamPair(1, 0, 1,
               np.array([0.20, 0.5]), np.array([0.30, 0.5]),
               np.array([0.60, 0.5]), np.array([0.70, 0.5]),
               np.array([0.0, 1.0]), np.array([0.0, 1.0]))
fused2 = fuse_seam(field2, [sp2], res, diffuse_texels=8)
check("gate leaves continuous seams alone", np.array_equal(fused2, field2))

# --- 3. lighting modes render to distinct images ----------------------------
blend = REPO_ROOT / "blender" / "gaming_console_2k.blend" / "gaming_console_2k.blend"
if blend.is_file():
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    objs = [o for o in bpy.data.objects if o.type == "MESH" and o.data and len(o.data.vertices)]
    obj = max(objs, key=lambda o: len(o.data.vertices))
    world = bpy.context.scene.world
    check("scene has a world (HDRI source)", world is not None,
          world.name if world else "NO WORLD")
    cam, _ = oblique.make_oblique_camera(bpy.context.scene, obj)
    out_dir = Path(bpy.path.abspath("//")) / ".ai_wear_cache" / "_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for mode in ("unlit", "neutral", "scene"):
        p = str(out_dir / f"{mode}.png")
        passes.render_clean(bpy.context.scene, cam, p, 256, lighting=mode,
                            unlit_object=obj if mode == "unlit" else None)
        img = bpy.data.images.load(p)
        w, h = img.size
        px = np.asarray(img.pixels[:]).reshape(h, w, 4)[..., :3]
        stats[mode] = (float(px.mean()), float(px.std()))
        bpy.data.images.remove(img)
        check(f"rendered {mode}", Path(p).is_file(), p)
    u, n, sc = stats["unlit"], stats["neutral"], stats["scene"]
    check("unlit/neutral/scene differ",
          any(abs(a[0] - b[0]) > 1e-3 for a, b in ((u, n), (n, sc), (u, sc))),
          f"means unlit={u[0]:.4f} neutral={n[0]:.4f} scene={sc[0]:.4f}")
    # A scene-lit comparison must still be legible when a future asset has no
    # World. render_clean restores the original None after the temporary
    # fallback World/Sun pass.
    bpy.context.scene.world = None
    fallback_path = str(out_dir / "scene_no_world_fallback.png")
    passes.render_clean(bpy.context.scene, cam, fallback_path, 256, lighting="scene")
    check("scene fallback renders without World", Path(fallback_path).is_file(), fallback_path)
    check("scene fallback restores missing World", bpy.context.scene.world is None)
    bpy.context.scene.world = world
else:
    print("SKIP  lighting render (blend not found)")

print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
