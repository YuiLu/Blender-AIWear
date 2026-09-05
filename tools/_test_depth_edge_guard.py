"""Regression: edited-image payload must not cross clean geometric boundaries."""

from pathlib import Path
import importlib.util
import sys
import types

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("ai_wear")
pkg.__path__ = [str(repo_root / "ai_wear")]
surface_pkg = types.ModuleType("ai_wear.surface")
surface_pkg.__path__ = [str(repo_root / "ai_wear" / "surface")]
sys.modules.setdefault("ai_wear", pkg)
sys.modules.setdefault("ai_wear.surface", surface_pkg)

utils_spec = importlib.util.spec_from_file_location(
    "ai_wear.utils", repo_root / "ai_wear" / "utils.py")
utils_mod = importlib.util.module_from_spec(utils_spec)
sys.modules["ai_wear.utils"] = utils_mod
utils_spec.loader.exec_module(utils_mod)

projection_spec = importlib.util.spec_from_file_location(
    "ai_wear.surface.projection",
    repo_root / "ai_wear" / "surface" / "projection.py")
projection_mod = importlib.util.module_from_spec(projection_spec)
sys.modules["ai_wear.surface.projection"] = projection_mod
projection_spec.loader.exec_module(projection_mod)
reject_depth_edge_payload = projection_mod.reject_depth_edge_payload


size = 21
coverage = np.ones((size, size), dtype=bool)
depth = np.ones((size, size), dtype=np.float32)
depth[:, 11:] = 2.0
mask = np.ones((size, size), dtype=np.float32)
rgb = np.ones((size, size, 3), dtype=np.float32)

guarded_mask, guarded_rgb, safe = reject_depth_edge_payload(
    mask, rgb, depth, coverage, radius=10.0, guard_pixels=2)

# Both sides of the discontinuity are protected; stable interiors remain.
assert not safe[:, 9:14].any()
assert np.allclose(guarded_mask[:, 9:14], 0.0)
assert np.allclose(guarded_rgb[:, 9:14], 0.5)
assert guarded_mask[10, 5] == 1.0
assert np.allclose(guarded_rgb[10, 5], 1.0)

# A background silhouette receives the same protection on the covered side.
coverage2 = np.zeros((size, size), dtype=bool)
coverage2[4:17, 4:17] = True
depth2 = np.where(coverage2, 1.0, 0.0).astype(np.float32)
_, _, safe2 = reject_depth_edge_payload(
    mask, rgb, depth2, coverage2, radius=10.0, guard_pixels=1)
assert safe2[10, 10]
assert not safe2[4:6, 4:17].any()

normals = np.zeros((size, size, 3), dtype=np.float32)
normals[:, :11, 0] = 1.0
normals[:, 11:, 1] = 1.0
mask3, rgb3, safe3 = reject_depth_edge_payload(
    mask, rgb, depth, coverage, radius=10.0, normal_buf=normals,
    guard_pixels=0, normal_guard_pixels=2)
assert not safe3[:, 9:14].any()
assert np.allclose(mask3[:, 9:14], 0.0)
assert np.allclose(rgb3[:, 9:14], 0.5)
assert mask3[10, 4] == 1.0

print("Depth-edge payload guard: PASS")
