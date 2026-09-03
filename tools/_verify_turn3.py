"""Verify Turn-3 fixes import + register cleanly. Run: blender -b --python tools/_verify_turn3.py"""
import sys, os
sys.path.insert(0, os.getcwd())
import bpy

errors = []
diag = []
try:
    import ai_wear
    diag.append(f"ai_wear file: {getattr(ai_wear, '__file__', '?')}")
    from ai_wear.operators import runner
    diag.append(f"runner file: {getattr(runner, '__file__', '?')}")
    diag.append(f"runner.CLASSES: {[c.__name__ for c in runner.CLASSES]}")
    # Force a clean re-register so any error in the new Turn-3 code surfaces
    # (the addon auto-loaded from the install junction at startup).
    try:
        ai_wear.unregister()
    except Exception as e:
        diag.append(f"unregister note: {e}")
    ai_wear.register()
    import bpy
    # Operators register into bpy.ops (getattr(bpy.types, name) is unreliable for
    # operator classes). The reliable check is the bpy.ops namespace.
    def _op_present(op_path):
        ns, name = op_path.split(".", 1)
        try:
            return hasattr(getattr(bpy.ops, ns), name)
        except AttributeError:
            return False
    reg_ops = [c.bl_idname for c in runner.CLASSES if _op_present(c.bl_idname)]
    diag.append(f"registered bpy.ops ops: {reg_ops}")
    diag.append(f"replay_downstream present: {_op_present('ai_wear.replay_downstream')}")
except Exception as e:
    import traceback
    errors.append(f"register failed: {e}\n{traceback.format_exc()}")

# Q3: replay operator + launchers exist
if not hasattr(bpy.ops.ai_wear, "replay_downstream"):
    errors.append("ai_wear.replay_downstream operator not registered")
from ai_wear.operators import pipeline
for fn in ("start_replay", "_run_replay", "_uv_coverage_diag", "_uv_empty_error_msg"):
    if not hasattr(pipeline, fn):
        errors.append(f"pipeline.{fn} missing")

# Q4/Q1/Q6 module-level sanity (import the modules; the real behavior was
# verified by _inspect_cache / _probe_weartime separately)
from ai_wear.surface import projection, wear_growth
from ai_wear.render import view_sampler
from ai_wear.shader import wear_nodegroup
for m, name in ((projection, "extract_screen_mask"),
                (projection, "accumulate_rgb_view"),
                (wear_growth, "build_weartime_from_graph"),
                (view_sampler, "compute_framing"),
                (wear_nodegroup, "ensure_node_group"),
                (wear_nodegroup, "attach_wear_overlay")):
    if not hasattr(m, name):
        errors.append(f"{m.__name__}.{name} missing")

# exercise ensure_node_group + attach_wear_overlay end-to-end (Q6 layout)
try:
    ng = wear_nodegroup.ensure_node_group()
    locs = [tuple(n.location) for n in ng.nodes]
    if len(locs) != len(set(locs)) and all(l == (0.0, 0.0) for l in locs):
        errors.append("node group nodes still all at (0,0)")
    wt_img = bpy.data.images.new("verify_wt", 16, 16)
    worn_img = bpy.data.images.new("verify_worn", 16, 16)
    # temp mesh + material with a Principled BSDF so the inject path runs
    mesh = bpy.data.meshes.new("verify_mesh")
    obj = bpy.data.objects.new("verify_obj", mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat = bpy.data.materials.new("verify_mat")
    mat.use_nodes = True
    obj.data.materials.append(mat)  # slot 0 (new material has a Principled BSDF)
    out_mats = wear_nodegroup.attach_wear_overlay(obj, wt_img, worn_img)
    if not out_mats:
        errors.append("attach_wear_overlay returned no materials")
    for out_mat in out_mats:
        mlocs = [tuple(n.location) for n in out_mat.node_tree.nodes]
        if all(l == (0.0, 0.0) for l in mlocs):
            errors.append(f"overlay material nodes all at (0,0): {out_mat.name}")
        names = {n.name for n in out_mat.node_tree.nodes}
        if wear_nodegroup.OVERLAY_MIX not in names:
            errors.append(f"overlay mix node missing: {out_mat.name}")
        if wear_nodegroup.MASKGROUP_NODE not in names:
            errors.append(f"mask group node missing: {out_mat.name}")
    # idempotency: re-attach must not nest a second overlay mix
    wear_nodegroup.attach_wear_overlay(obj, wt_img, worn_img)
    for out_mat in out_mats:
        mix_count = sum(
            1 for n in out_mat.node_tree.nodes
            if n.name == wear_nodegroup.OVERLAY_MIX)
        if mix_count != 1:
            errors.append(
                f"re-attach nested overlays: {out_mat.name} "
                f"(mix_count={mix_count})")
    # cleanup
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)
    bpy.data.materials.remove(mat)
    bpy.data.images.remove(wt_img)
    bpy.data.images.remove(worn_img)
except Exception as e:
    import traceback
    errors.append(f"node build failed: {e}\n{traceback.format_exc()}")

if errors:
    print("VERIFY FAIL:")
    for e in errors:
        print(" -", e)
    for d in diag:
        print("  diag:", d)
    sys.exit(1)
print("VERIFY OK: register + replay op + pipeline helpers + modules + node layout all good.")
for d in diag:
    print("  diag:", d)
