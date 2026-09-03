"""Headless check: preset seeding + registration + feather clamp.

Run:
  blender -b "blender/gaming_console_2k.blend/gaming_console_2k.blend" \
    --python tools/_test_presets2.py
"""
import sys, os
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
import bpy

import ai_wear
try:
    ai_wear.unregister()
except Exception:
    pass
ai_wear.register()
print("REGISTER OK")
assert bpy.app.timers.is_registered(ai_wear._seed_retry_timer), "seed timer not registered"
print("seed timer registered OK")

s = bpy.context.scene.ai_wear
print("seeded preset count:", len(s.presets))
print("names:", [p.name for p in s.presets])

from ai_wear import presets
d = {p.name: p for p in s.presets}
assert d["cams_08"].camera_preset == "AUTO_COUNT" and d["cams_08"].camera_count == 8
assert d["geometry_off"].use_geometry_prior is False
assert d["topology_off"].use_topology_growth is False
assert d["amount_30"].wear_amount == 30.0
assert "micro-scratches" in d["extra_on"].prompt_extra
assert not ({"geometry_on", "topology_on", "seam_on", "extra_off"} & set(d))
print("spot-checks OK")

# idempotent + restore
n = len(s.presets)
presets.seed_experiment_presets(s)
assert len(s.presets) == n, "seed should be idempotent"
presets.restore_experiment_presets(s)
assert len(s.presets) == n, "restore should reseed"
print("idempotency/restore OK")

# Migration: scenes saved by 0.3.2/0.3.3 may still contain the four redundant
# built-ins.  A normal seed pass must remove them without clearing other rows.
for old_name in presets.DEPRECATED_EXPERIMENT_PRESETS:
    old = s.presets.add()
    old.name = old_name
assert len(s.presets) == 18
presets.seed_experiment_presets(s)
assert len(s.presets) == 14
assert not (presets.DEPRECATED_EXPERIMENT_PRESETS &
            {p.name for p in s.presets})
print("deprecated-preset migration OK")

# startup-restriction safety net: the retry timer seeds once bpy.data is writable
s.presets.clear()
assert len(s.presets) == 0
assert ai_wear._seed_retry_timer() is None, "timer should stop after seeding"
assert len(s.presets) == 14, "retry timer did not seed"
print("retry timer seed OK")

# feather clamp (exercise the real code path with a minimal material mock)
from ai_wear.shader import wear_nodegroup as wn

class _Node:
    def __init__(self):
        self.inputs = {"Feather": type("S", (), {"default_value": 0.0})()}
class _Nodes:
    def __init__(self):
        self._m = {wn.MASKGROUP_NODE: _Node()}
    def get(self, name):
        return self._m.get(name)
class _Tree:
    def __init__(self):
        self.nodes = _Nodes()
class _Mat:
    use_nodes = True
    def __init__(self):
        self.node_tree = _Tree()

mat = _Mat()
wn.set_feather(mat, 0.0)
got = mat.node_tree.nodes.get(wn.MASKGROUP_NODE).inputs["Feather"].default_value
assert got == wn.FEATHER_EPS, f"feather not clamped: {got}"
wn.set_feather(mat, 0.5)
got = mat.node_tree.nodes.get(wn.MASKGROUP_NODE).inputs["Feather"].default_value
assert got == 0.5, f"normal feather clobbered: {got}"
print("feather clamp OK (0 -> %r, 0.5 -> 0.5)" % wn.FEATHER_EPS)
print("ALL OK")
