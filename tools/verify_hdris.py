"""Verify that no image data-block is unresolved after the HDRI relink.

Run per file:
  blender -b "<file>.blend" --python tools/verify_hdris.py
"""
import bpy, os


def _abs(im):
    try:
        return bpy.path.abspath(im.filepath)
    except Exception:
        return im.filepath


missing = []
for im in bpy.data.images:
    if im.packed_file:
        continue
    if not im.filepath:  # e.g. Render Result / Viewer — not a file, ignore
        continue
    if not os.path.exists(_abs(im)):
        missing.append((im.name, im.filepath))

env_status = "none"
for w in bpy.data.worlds:
    if not (w.use_nodes and w.node_tree):
        continue
    for n in w.node_tree.nodes:
        if n.bl_idname == "ShaderNodeTexEnvironment":
            im = n.image
            if im is None:
                env_status = f"WORLD '{w.name}': env texture has NO image"
            else:
                ok = os.path.exists(_abs(im))
                env_status = f"WORLD '{w.name}': env='{im.name}' exists={ok}"
        elif n.bl_idname == "ShaderNodeGroup":
            # gaming_console uses a group; peek inside for an env image
            gnt = n.node_tree
            if gnt:
                for gn in gnt.nodes:
                    if gn.bl_idname == "ShaderNodeTexEnvironment" and gn.image:
                        ok = os.path.exists(_abs(gn.image))
                        env_status = f"WORLD '{w.name}': group env='{gn.image.name}' exists={ok}"

print("MISSING IMAGES:", missing if missing else "NONE  ✓")
print("WORLD HDR:", env_status)
print("RESULT:", "PASS  ✓" if (not missing and "exists=True" in env_status) else "CHECK")
