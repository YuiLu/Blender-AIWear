"""Shared utilities: image IO (via Blender), numpy helpers, small misc.

Image load/save MUST run on the Blender main thread (bpy.data.images access).
Worker threads only do HTTP + raw file IO and never call these image helpers.
"""

from __future__ import annotations

import base64
import os
import struct
import zlib
from typing import Optional, Tuple

import numpy as np

# Blender ships with numpy; if a custom build lacks it, surface math degrades.
HAS_NUMPY = True


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:48] or "object"


# --- Image IO ----------------------------------------------------------------

def load_image_rgba(path: str) -> np.ndarray:
    """Load an image file into a float32 (H, W, 4) array in 0..1, top-row-first.

    Main thread only. The temp image block is removed after copying pixels.
    """
    import bpy
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size[0], img.size[1]
        n = w * h * 4
        buf = np.empty(n, dtype=np.float32)
        img.pixels.foreach_get(buf)
        buf = buf.reshape(h, w, 4)
        return np.ascontiguousarray(buf[::-1])  # flip Y to top-first
    finally:
        bpy.data.images.remove(img)


def _write_png_rgba(path: str, rgba: np.ndarray, bit_depth: int) -> str:
    """Write an (H,W,4) float32 0..1 array (top-row-first) as a raw-linear PNG.

    Pure-Python (zlib+struct) encoder. This bypasses Blender's image-save path,
    which in Blender 5.2 does NOT persist the in-memory pixel buffer set via
    foreach_set to disk — save_render / Image.save() write RGB=0, A=1 (a blank
    image) even when the buffer holds correct non-zero data (verified: foreach_set
    + foreach_get round-trips fine in memory; only the disk encode is blank).
    The shader reads WearTime back as Non-Color (identity), so raw-linear values
    are exactly what we want written.
    """
    h, w = rgba.shape[:2]
    rgba = np.clip(np.asarray(rgba, dtype=np.float64), 0.0, 1.0)
    if bit_depth == 16:
        u = (rgba * 65535.0 + 0.5).astype(np.uint16).astype(">u2")  # big-endian
        bpp = 8  # 4 channels * 2 bytes
    else:
        u = (rgba * 255.0 + 0.5).astype(np.uint8)
        bpp = 4  # 4 channels * 1 byte
    flat = u.tobytes()  # row-major (H, W, 4) -> contiguous bytes
    stride = w * bpp
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # PNG per-scanline filter: None
        raw.extend(flat[y * stride:(y + 1) * stride])
    comp = zlib.compress(bytes(raw), 9)

    def _chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bit_depth, 6, 0, 0, 0)))
        f.write(_chunk(b"IDAT", comp))
        f.write(_chunk(b"IEND", b""))
    return path


def _save_image_blender(path: str, rgba: np.ndarray, fmt: str) -> str:
    """Fallback Blender image save (used only for EXR / float formats).

    PNG goes through _write_png_rgba; this remains for OPEN_EXR which the
    pure-Python encoder does not produce. NOTE: on Blender 5.2 this path
    writes a BLANK image for in-memory-created data (see _write_png_rgba),
    so it should not be relied on for the WearTime PNG output.
    """
    import bpy
    h, w = rgba.shape[:2]
    flat = np.ascontiguousarray(rgba[::-1]).ravel()  # bottom-row-first
    img = bpy.data.images.new("ai_wear_tmp", w, h, alpha=True, float_buffer=True)
    try:
        img.pixels.foreach_set(flat)
        img.colorspace_settings.name = "Non-Color"
        img.filepath_raw = path
        fmt_map = {
            "PNG16": ("PNG", "16", "RGBA"),
            "PNG8": ("PNG", "8", "RGBA"),
            "EXR": ("OPEN_EXR", "32", "RGBA"),
        }
        f, depth, mode = fmt_map.get(fmt, fmt_map["PNG16"])
        scene = bpy.context.scene
        old_fmt = scene.render.image_settings.file_format
        old_depth = scene.render.image_settings.color_depth
        old_mode = scene.render.image_settings.color_mode
        try:
            scene.render.image_settings.file_format = f
            scene.render.image_settings.color_depth = depth
            scene.render.image_settings.color_mode = mode
            img.save_render(path, scene=scene)
        finally:
            scene.render.image_settings.file_format = old_fmt
            scene.render.image_settings.color_depth = old_depth
            scene.render.image_settings.color_mode = old_mode
    finally:
        bpy.data.images.remove(img)
    return path


def save_image(path: str, arr: np.ndarray, fmt: str = "PNG16") -> str:
    """Save a float32 (H, W, C) array (0..1, top-row-first) to PNG16/PNG8/EXR.

    PNG16/PNG8 use a pure-Python encoder (zlib+struct) — see _write_png_rgba
    for why Blender's own image save can't be used here on 5.2. EXR falls back
    to the Blender path (rare; if EXR comes up blank, switch to PNG16).
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[:, :, :min(c, 4)] = arr[:, :, :min(c, 4)]
    if c < 4:
        rgba[:, :, 3] = 1.0
    if fmt in ("PNG16", "PNG8"):
        return _write_png_rgba(path, rgba, bit_depth=16 if fmt == "PNG16" else 8)
    return _save_image_blender(path, rgba, fmt)


def resize_bilinear(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Bilinear resize. arr: (H, W, C) or (H, W). No external deps."""
    arr = np.asarray(arr, dtype=np.float32)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[:, :, None]
        squeeze = True
    h, w = arr.shape[:2]
    if (h, w) == (new_h, new_w):
        return arr[:, :, 0] if squeeze else arr.copy()
    ys = np.linspace(0, h - 1, new_h) if new_h > 1 else np.array([0.0])
    xs = np.linspace(0, w - 1, new_w) if new_w > 1 else np.array([0.0])
    y0 = np.floor(ys).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    fy = (ys - y0).astype(np.float32)
    x0 = np.floor(xs).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    fx = (xs - x0).astype(np.float32)
    cols_a = arr[:, x0, :]  # (H, new_w, C)
    cols_b = arr[:, x1, :]
    row_interp = cols_a + (cols_b - cols_a) * fx[None, :, None]
    out = row_interp[y0] + (row_interp[y1] - row_interp[y0]) * fy[:, None, None]
    return out[:, :, 0] if squeeze else out.astype(np.float32)


def file_to_base64(path: str) -> Tuple[str, str]:
    """Return (base64_data, mime)."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp", "exr": "image/x-exr"}.get(ext, "image/png")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii"), mime


def save_array_png8(arr: np.ndarray, path: str) -> str:
    """Save a single-channel 0..1 array as an 8-bit grayscale mask PNG."""
    a = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
    if a.ndim == 3:
        a = a[:, :, 0]
    rgba = np.zeros((a.shape[0], a.shape[1], 4), dtype=np.float32)
    rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = a / 255.0
    rgba[:, :, 3] = 1.0
    return save_image(path, rgba, "PNG8")


def clamp01(x):
    return np.clip(x, 0.0, 1.0)
