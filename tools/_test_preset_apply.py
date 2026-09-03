"""Headless check: preset auto-apply on selection + apply_preset round-trip.

Run:
  blender -b --python tools/_test_preset_apply.py
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

s = bpy.context.scene.ai_wear
from ai_wear import presets
assert len(s.presets) == 18, len(s.presets)

# auto-apply: selecting 'amount_30' should move the scene Wear Amount to 30
idx = [p.name for p in s.presets].index("amount_30")
s.active_preset_index = idx
assert abs(s.wear_amount - 30.0) < 1e-6, s.wear_amount
print("auto-apply on selection OK (amount_30 -> wear_amount=30)")

# auto-apply: selecting 'geometry_off' should flip the scene switch
idx = [p.name for p in s.presets].index("geometry_off")
s.active_preset_index = idx
assert s.use_geometry_prior is False, s.use_geometry_prior
print("auto-apply switch OK (geometry_off -> use_geometry_prior=False)")

# direct apply_preset round-trip
p = s.presets[0]
p.wear_amount = 42.0
presets.apply_preset(s, p)
assert abs(s.wear_amount - 42.0) < 1e-6, s.wear_amount
print("apply_preset round-trip OK")

print("ALL PRESET-APPLY OK")
