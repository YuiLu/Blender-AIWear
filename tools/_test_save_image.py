"""Regression test for the Blender-5.2 save_image bug + fix.

BUG: Blender 5.2's image-save path (save_render / Image.save) does NOT persist
the in-memory pixel buffer set via foreach_set to disk — it writes RGB=0, A=1
(a blank image) even when the buffer holds correct non-zero data. foreach_set
itself works (set+get round-trips fine in memory); only the disk encode is
blank. This made WearTime.png come out all-black despite a healthy in-memory
field (mean 0.33) -> shader read T=0 everywhere -> smoothstep -> whole surface
"fully worn" (Q5/Q7).

FIX: utils.save_image now writes PNG via a pure-Python (zlib+struct) encoder
that bypasses Blender's save entirely. EXR still falls back to Blender (rare).

This test asserts the fix: save non-zero RGB, reload as Non-Color, the values
must survive.

Run: blender -b "blender/gaming_console_2k.blend/gaming_console_2k.blend" \
        --python tools/_test_save_image.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.getcwd())
import bpy
from ai_wear import utils

OUT = os.path.join(os.getcwd(), "blender", "gaming_console_2k.blend",
                   ".ai_wear_cache", "gaming_console", "debug")
os.makedirs(OUT, exist_ok=True)


def reload_noncolor(path):
    img = bpy.data.images.load(path, check_existing=False)
    img.colorspace_settings.name = "Non-Color"  # the way the shader reads it
    try:
        buf = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(buf)
        return buf.reshape(img.size[1], img.size[0], 4)[::-1]
    finally:
        bpy.data.images.remove(img)


def check(tag, path, expect_r, expect_g):
    a = reload_noncolor(path)
    r, g = float(a[..., 0].max()), float(a[..., 1].max())
    ok = abs(r - expect_r) < 0.02 and abs(g - expect_g) < 0.02
    print(f"  [{tag}] R={r:.4f} (exp {expect_r}) G={g:.4f} (exp {expect_g}) "
          f"size={os.path.getsize(path)} -> {'OK' if ok else 'FAIL'}")
    assert ok, f"{tag}: RGB did not survive save (r={r}, g={g})"


# uniform RGB=0.5/0.25, alpha=1, 1024^2 (the WearTime shape)
arr = np.zeros((1024, 1024, 4), dtype=np.float32)
arr[..., 0] = 0.5; arr[..., 1] = 0.25; arr[..., 3] = 1.0
p16 = os.path.join(OUT, "regress_png16.png")
utils.save_image(p16, arr, "PNG16")
check("PNG16 uniform", p16, 0.5, 0.25)

p8 = os.path.join(OUT, "regress_png8.png")
utils.save_image(p8, arr, "PNG8")
check("PNG8 uniform", p8, 0.5, 0.25)

# a WearTime-like gradient (R=G=B=val) — must preserve max 1.0
val = np.linspace(0, 1, 1024 * 1024).reshape(1024, 1024)
wt = np.zeros((1024, 1024, 4), dtype=np.float32)
wt[..., 0] = wt[..., 1] = wt[..., 2] = val
wt[..., 3] = 1.0
pwt = os.path.join(OUT, "regress_gradient.png")
utils.save_image(pwt, wt, "PNG16")
a = reload_noncolor(pwt)
print(f"  [gradient PNG16] R range [{a[...,0].min():.4f},{a[...,0].max():.4f}] "
      f"(exp 0..1) -> {'OK' if a[...,0].max() > 0.98 else 'FAIL'}")
assert a[..., 0].max() > 0.98

print("\n=== save_image regression PASS (non-zero RGB survives PNG16/PNG8 roundtrip) ===")
