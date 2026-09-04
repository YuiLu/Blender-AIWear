"""Headless check: the prompt-strategy presets seed + apply correctly.

Run: blender -b --python tools/_check_presets.py
No .blend loaded; does not touch any cache. Safe alongside other runs.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ai_wear  # noqa: E402

try:
    ai_wear.unregister()
except Exception:
    pass
ai_wear.register()

import bpy  # noqa: E402
from ai_wear import presets  # noqa: E402

s = bpy.context.scene.ai_wear
presets.seed_experiment_presets(s)
names = [p.name for p in s.presets]
print("AIWEAR_PRESET_CHECK")
print("count:", len(names))
print("names:", names)

for target in ("prompt_light", "prompt_severe", "prompt_wood", "prompt_plastic"):
    p = next((x for x in s.presets if x.name == target), None)
    assert p is not None, f"missing preset {target}"
print("new_presets_present: yes")

# apply_preset must map max_state -> max_wear_state, material -> prompt_material,
# wear_type -> prompt_wear_type.
p_light = next(x for x in s.presets if x.name == "prompt_light")
presets.apply_preset(s, p_light)
print("applied prompt_light -> max_wear_state:", repr(s.max_wear_state))

p_wood = next(x for x in s.presets if x.name == "prompt_wood")
presets.apply_preset(s, p_wood)
print("applied prompt_wood -> prompt_material:", repr(s.prompt_material),
      "prompt_wear_type:", repr(s.prompt_wear_type))

# _make_wear_prompt is a module-level function (properties.py:61); the scene
# delegates via make_wear_prompt() at line 254. Show the prompt scales with
# max_wear_state and adapts to material -- concrete evidence for the doc.
from ai_wear.properties import _make_wear_prompt  # noqa: E402

s.max_wear_state = "severe"
s.prompt_material = "ABS plastic"
s.prompt_wear_type = "scuffs, abrasion and stress marks"
s.prompt_extra = ""
print("prompt(severe,plastic):", _make_wear_prompt(s, None))
s.max_wear_state = "light"
print("prompt(light,plastic):", _make_wear_prompt(s, None))
print("AIWEAR_PRESET_CHECK_OK")
