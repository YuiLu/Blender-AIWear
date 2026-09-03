"""Relink missing image data-blocks to files found in this .blend's textures/ dir.

Run headlessly per file:
  blender -b "<file>.blend" --python tools/relink_missing.py

Uses Blender's built-in File > External Data > Find Missing Files (operator
file.find_missing_files) so the PBR maps already in //textures/ stay as-is and
only the missing World HDRIs get repointed to the HDRs you placed next to them.
Then saves the .blend. A .blend1 backup exists beside each file.
"""
import bpy

directory = bpy.path.abspath("//textures")
print("search directory:", directory)
bpy.ops.file.find_missing_files(directory=directory, find_all=True)
bpy.ops.wm.save_mainfile()
print("RELINK + SAVE done:", bpy.data.filepath)
