"""Render the active wear-overlay model from a FIXED oblique hero camera.

Guarantees a consistent oblique view + preset-aligned filename so per-preset
experiment renders are directly comparable as qualitative results. The camera
math lives in ``ai_wear/render/oblique.py`` (shared with the pipeline's
end-of-run snapshot) so both callers stay identical.

By default it renders whatever the scene currently shows (i.e. after a pipeline
run the overlay is already attached). Pass ``--attach-cache`` to instead attach
the wear overlay from the canonical cache (WearThreshold.png + AIWear_WornTex.png)
first — this lets you re-render a cached result headless without re-running the
pipeline. No AI calls in any mode.

CLI:
  blender -b "<blend>.blend" --python tools/render_oblique.py -- <label> [object_name] [--attach-cache] [amount0-100] [feather0-100]
e.g.
  blender -b "blender/gaming_console_2k.blend/gaming_console_2k.blend" \
    --python tools/render_oblique.py -- cams_08 gaming_console --attach-cache 100 50
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ai_wear  # noqa: E402

try:
    ai_wear.unregister()
except Exception:
    pass
ai_wear.register()

from ai_wear import shader  # noqa: E402
from ai_wear.cache import job_cache  # noqa: E402
from ai_wear.render import oblique  # noqa: E402


CAMERA_NAME = oblique.CAMERA_NAME
DIRECTION = oblique.DIRECTION
LENS = oblique.LENS
DISTANCE_FACTOR = oblique.DISTANCE_FACTOR
RENDER_RESOLUTION = oblique.RENDER_RESOLUTION

# Back-compat: keep the function names the earlier helper exposed.
make_oblique_camera = oblique.make_oblique_camera


def pick_object(name=None):
    """Resolve the target mesh: explicit name > active object > largest mesh."""
    if name:
        obj = bpy.data.objects.get(name)
        if obj and obj.type == "MESH":
            return obj
    obj = bpy.context.active_object
    if obj and obj.type == "MESH":
        return obj
    meshes = [o for o in bpy.data.objects
              if o.type == "MESH" and o.data and len(o.data.vertices) > 0]
    if meshes:
        return max(meshes, key=lambda o: len(o.data.vertices))
    return None


def attach_wear_from_cache(obj, layer_name, amount01=1.0, feather01=0.5):
    """Attach the wear overlay from the object's canonical cache files.

    Loads ``WearThreshold.png`` + ``AIWear_WornTex.png`` from the cache dir and
    injects the overlay into the object's materials (in-memory only — this
    script never saves the .blend). Returns the list of preview materials.
    """
    cache_dir = Path(job_cache.object_cache_dir(obj.name))
    wt_path = cache_dir / "WearThreshold.png"
    worn_path = cache_dir / "AIWear_WornTex.png"
    if not wt_path.is_file() or not worn_path.is_file():
        raise RuntimeError(
            f"Cache textures missing for '{obj.name}': {wt_path.name}, {worn_path.name}")

    wt_img = bpy.data.images.load(str(wt_path))
    wt_img.name = "AIWear_WearThreshold"
    wt_img.colorspace_settings.name = "Non-Color"
    worn_img = bpy.data.images.load(str(worn_path))
    worn_img.name = "AIWear_WornTex"
    worn_img.colorspace_settings.name = "Non-Color"

    mats = shader.attach_wear_overlay(obj, wt_img, worn_img, layer_name)
    if not mats:
        raise RuntimeError(f"Could not attach a wear preview material to '{obj.name}'.")
    for mat in mats:
        shader.set_amount(mat, amount01)
        shader.set_feather(mat, feather01)
    return mats


def render_oblique(label="render", object_name=None, out_dir=None,
                   resolution=RENDER_RESOLUTION, attach_cache=False,
                   layer_name="AI_WearUV", amount01=1.0, feather01=0.5):
    """Render the model from the fixed oblique camera; return the output path.

    ``out_dir`` defaults to ``<cache>/experiments/oblique_renders`` so the file
    sits alongside the pipeline's experiment snapshots, named ``<label>_oblique.png``.
    """
    scene = bpy.context.scene
    obj = pick_object(object_name)
    if obj is None or obj.type != "MESH":
        raise RuntimeError("render_oblique: no mesh object found to render")

    if attach_cache:
        attach_wear_from_cache(obj, layer_name, amount01, feather01)

    if out_dir is None:
        cache_dir = Path(job_cache.object_cache_dir(obj.name))
        out_dir = cache_dir / "experiments" / "oblique_renders"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{label}_oblique.png"
    oblique.render_oblique(scene, obj, str(out_png), resolution)
    return out_png


def _cli():
    argv = sys.argv
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    label = extra[0] if len(extra) > 0 else "render"
    object_name = extra[1] if len(extra) > 1 else None
    attach_cache = "--attach-cache" in extra
    amount01 = 1.0
    feather01 = 0.5
    # optional trailing numeric args after --attach-cache: <amount0-100> <feather0-100>
    nums = [a for a in extra if a.lstrip("-").replace(".", "").isdigit()]
    if nums:
        amount01 = float(nums[0]) / 100.0
    if len(nums) > 1:
        feather01 = float(nums[1]) / 100.0
    out_png = render_oblique(label=label, object_name=object_name,
                             attach_cache=attach_cache,
                             amount01=amount01, feather01=feather01)
    print("AIWEAR_OBLIQUE_OK")
    print("AIWEAR_OBLIQUE_PATH=" + str(out_png))
    print("AIWEAR_OBLIQUE_ATTACH_CACHE=" + str(attach_cache))


if __name__ == "__main__":
    _cli()
