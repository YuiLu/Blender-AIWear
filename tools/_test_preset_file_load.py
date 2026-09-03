"""Regression: presets survive the real add-on-enable -> .blend-load order.

Run with factory startup so AI Wear is first registered against the temporary
startup scene, then a project file replaces that scene.  Non-persistent load
handlers/timers used to disappear at this boundary, leaving the panel empty.
"""

from __future__ import annotations

import os
import sys

import bpy

ROOT = os.getcwd()
sys.path.insert(0, ROOT)

import ai_wear

try:
    ai_wear.unregister()
except Exception:
    pass
ai_wear.register()

assert len(bpy.context.scene.ai_wear.presets) == 14

blend_path = os.path.join(
    os.path.dirname(ROOT), "..", "blender", "gaming_console_2k.blend",
    "gaming_console_2k.blend")
blend_path = os.path.abspath(blend_path)
assert os.path.isfile(blend_path), blend_path

bpy.ops.wm.open_mainfile(filepath=blend_path)

s = bpy.context.scene.ai_wear
assert len(s.presets) == 14, (
    "Presets were lost when the .blend replaced the startup scene: "
    f"got {len(s.presets)}")
assert ai_wear._seed_presets_on_load in bpy.app.handlers.load_post

print("PRESET_FILE_LOAD_OK")
print("preset_count:", len(s.presets))
