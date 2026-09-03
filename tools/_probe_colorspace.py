"""Decisively determine whether image.pixels.foreach_get returns sRGB-encoded
or linear values for the AI's sRGB PNG, under each colorspace tag. This settles
whether AIWear_WornTex.png must be tagged sRGB (shader decodes once) or
Non-Color (already linear). Run:
  blender -b "blender/gaming_console_2k.blend/gaming_console_2k.blend" --python tools/_probe_colorspace.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.getcwd())
import bpy
from ai_wear import utils

CACHE = os.path.join(os.getcwd(), "blender", "gaming_console_2k.blend",
                     ".ai_wear_cache", "gaming_console")
worn = os.path.join(CACHE, "views", "worn_V0.png")

v = utils.load_image_rgba(worn)
print(f"load_image_rgba(worn_V0) R: mean={v[...,0].mean():.4f} max={v[...,0].max():.4f}")

img = bpy.data.images.load(worn, check_existing=False)
print("default colorspace on load:", img.colorspace_settings.name)

def grab():
    buf = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(img.size[1], img.size[0], 4)[::-1]

d = grab()
print(f"foreach_get [default={img.colorspace_settings.name}] R: mean={d[...,0].mean():.4f} max={d[...,0].max():.4f}")

img.colorspace_settings.name = "sRGB"
s = grab()
print(f"foreach_get [sRGB] R: mean={s[...,0].mean():.4f} max={s[...,0].max():.4f}")

img.colorspace_settings.name = "Non-Color"
n = grab()
print(f"foreach_get [Non-Color] R: mean={n[...,0].mean():.4f} max={n[...,0].max():.4f}")

# same-mean comparison
print("  -> default==load_image_rgba?", float(np.abs(d[...,0].mean() - v[...,0].mean())) < 1e-4)
print("  -> sRGB==Non-Color (identity accessor)?",
      float(np.abs(s[...,0].mean() - n[...,0].mean())) < 1e-4)
print("  -> sRGB mean < Non-Color mean (sRGB decoded to linear on access)?",
      float(s[...,0].mean()) < float(n[...,0].mean()) - 0.01)
bpy.data.images.remove(img)
