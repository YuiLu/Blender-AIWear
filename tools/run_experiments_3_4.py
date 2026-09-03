"""Run qualitative experiments 3 and 4 from cached six-view AI images.

No image-generation provider is invoked. Each case runs Replay Downstream,
saves an experiment bundle, and renders one fixed hero view for comparison.

Run from the repository root:
  blender -b blender/gaming_console_2k.blend/gaming_console_2k.blend \
    --python tools/run_experiments_3_4.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.getcwd()
sys.path.insert(0, ROOT)

import bpy
from mathutils import Vector

import ai_wear

try:
    ai_wear.unregister()
except Exception:
    pass
ai_wear.register()

from ai_wear.cache import job_cache
from ai_wear.operators import pipeline
from ai_wear.render import passes


class ImmediateBridge:
    @staticmethod
    def run(fn, *args, **kwargs):
        return fn(*args, **kwargs)


obj = bpy.data.objects.get("gaming_console")
assert obj is not None and obj.type == "MESH"
for selected in bpy.context.selected_objects:
    selected.select_set(False)
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

scene = bpy.context.scene
settings = scene.ai_wear
settings.target_uv_layer = "AI_WearUV"
settings.work_resolution = 1024
settings.export_format = "PNG16"
settings.wear_amount = 60.0
settings.feather = 4.0
settings.use_ai_mask = True
settings.seam_fuse = True
settings.use_padding = True
settings.save_experiment_snapshot = True

corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
center = sum(corners, Vector()) / len(corners)
radius = max((corner - center).length for corner in corners)
cam_data = bpy.data.cameras.new("AIWear_ExperimentHero_data")
cam_data.lens = 52.0
hero = bpy.data.objects.new("AIWear_ExperimentHero", cam_data)
scene.collection.objects.link(hero)
hero.location = center + Vector((1.15, -1.35, 0.95)).normalized() * radius * 3.5
hero.rotation_euler = (center - hero.location).to_track_quat("-Z", "Y").to_euler()

cases = (
    ("geometry_off", False, True),
    ("geometry_on", True, True),
    ("topology_off", True, False),
    ("topology_on", True, True),
)

results = []
for label, geometry_on, topology_on in cases:
    settings.experiment_label = label
    settings.use_geometry_prior = geometry_on
    settings.use_topology_growth = topology_on
    snap = pipeline.snapshot_context(bpy.context)
    job = job_cache.create_job()
    pipeline._run_replay(job, snap, ImmediateBridge())
    assert job.state == job_cache.JobState.DONE, (
        label, job.state, job.error, job.message)
    exp_dir = job.meta.get("experiment_dir")
    assert exp_dir and os.path.isdir(exp_dir), (label, exp_dir)
    render_path = os.path.join(exp_dir, "render_model.png")
    passes.render_clean(scene, hero, render_path, 1024)
    result = {
        "label": label,
        "geometry_prior": geometry_on,
        "topology_growth": topology_on,
        "elapsed_seconds": round(float(__import__("time").time() - job.started), 3),
        "experiment_dir": exp_dir,
        "render": render_path,
        "coverage_ratio": job.meta.get("coverage_ratio"),
    }
    results.append(result)
    print("AIWEAR_EXPERIMENT_CASE", json.dumps(result, ensure_ascii=False))

summary_path = os.path.join(
    job_cache.object_cache_dir(obj.name), "experiments", "experiments_3_4_summary.json")
with open(summary_path, "w", encoding="utf-8") as stream:
    json.dump({"image_generation_calls": 0, "cases": results}, stream,
              ensure_ascii=False, indent=2)

print("AIWEAR_EXPERIMENTS_3_4_OK")
print("summary:", summary_path)
