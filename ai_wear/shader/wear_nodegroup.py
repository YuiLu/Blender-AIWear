"""Wear overlay shader.

A reusable node group 'AIWearMask' computes the wear gate:
    gate = smoothstep(T - feather, T + feather, wear_amount)
from a WearTime texture (Non-Color, R channel = T). The Wear Amount slider
only re-thresholds; no AI / surface field re-run.

attach_wear_overlay() composites the AI clean→worn color residual over
the object's EXISTING material: it injects
    bsdf.BaseColor = mix(existing_base_color,
                         existing_base_color + decode(worn_tex.rgb),
                         gate * worn_tex.alpha)
into the existing material's Principled BSDF, preserving the rest of the
authored material (roughness, metallic, normals). worn_tex.alpha is the
contrast-shaped, reprojected AI wear mask (already intersected with valid
camera coverage), so even Wear Amount=100 stays inside observed scratches and
edge chips. Falls back to a standalone overlay material if the existing
material has no Principled BSDF.

The 'AIWearMask' group is the portable part — production materials can route
its Mask output into their own wear layers (Substance / mix shaders).
"""

from __future__ import annotations

import bpy
from bpy.types import NodeTree, ShaderNodeTree

GROUP_NAME = "AIWearMask"
PREVIEW_MAT = "AIWear_Preview"
WEARTIME_NODE = "AIWear_WearTime"
WORNTEX_NODE = "AIWear_WornTex"
MASKGROUP_NODE = "AIWear_MaskGroup"
UVMAP_NODE = "AIWear_UVMap"
OVERLAY_MIX = "AIWear_OverlayMix"   # MixRGB: mix(base, worn_tex, gate*alpha)
ALPHA_MUL = "AIWear_AlphaMul"       # Math MULTIPLY: gate * worn_tex.alpha
BASE_SRC = "AIWear_BaseSrc"         # ShaderNodeRGB materializing an unlinked BSDF default
DELTA_SUB = "AIWear_DeltaSubtract"
DELTA_SCALE = "AIWear_DeltaScale"
BASE_ADD = "AIWear_BasePlusDelta"
OWNER_PROP = "ai_wear_preview_owner"
SOURCE_PROP = "ai_wear_source_material"
# 2.0 exactly decodes the 0.5+0.5*delta encoding.  The bake stage already
# rejects global relighting and bounds the residual.  Extra gain here amplified
# thin-island reprojection texels into the blocky/mosaic pattern seen on the
# gaming-console sides, so the shader deliberately performs an exact decode.
DELTA_DECODE_GAIN = 2.0
# The smoothstep gate collapses to a division-by-zero (hard step / NaN) in the
# MapRange node when feather is exactly 0.  Clamp to a tiny positive width so the
# gate stays well-defined at the slider's 0 end; 1e-4 is imperceptibly narrow and
# matches the mask-export guard in operators/runner.py.
FEATHER_EPS = 1e-4


def ensure_node_group() -> ShaderNodeTree:
    ng = bpy.data.node_groups.get(GROUP_NAME)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(GROUP_NAME, "ShaderNodeTree")
    nt = ng.nodes
    links = ng.links

    # Interface sockets — Blender 4.0+ API: ng.interface.new_socket(...). The
    # old ng.inputs / ng.outputs collections were removed in 4.0; the default
    # value now lives on the returned interface socket item (NodeTreeInterfaceSocket).
    iface = ng.interface
    iface.new_socket(name="WearTime", in_out="INPUT", socket_type="NodeSocketColor")
    s_amt = iface.new_socket(name="Wear Amount", in_out="INPUT", socket_type="NodeSocketFloat")
    s_amt.default_value = 0.6
    s_feather = iface.new_socket(name="Feather", in_out="INPUT", socket_type="NodeSocketFloat")
    s_feather.default_value = 0.04
    iface.new_socket(name="Mask", in_out="OUTPUT", socket_type="NodeSocketFloat")

    sep = nt.new("ShaderNodeSeparateXYZ")
    sub = nt.new("ShaderNodeMath"); sub.operation = "SUBTRACT"
    add = nt.new("ShaderNodeMath"); add.operation = "ADD"
    mr = nt.new("ShaderNodeMapRange")
    # Blender 4.0+ renamed MapRange.interpolation -> interpolation_type, and the
    # "SMOOTH" enum value -> "SMOOTHSTEP" (enum: LINEAR/STEPPED/SMOOTHSTEP/SMOOTHERSTEP)
    mr.interpolation_type = "SMOOTHSTEP"
    mr.clamp = True
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 1.0

    out_node = nt.new("NodeGroupOutput")
    in_node = nt.new("NodeGroupInput")

    # Lay nodes out left-to-right so they don't pile on (0,0). Without explicit
    # .location every node lands at the origin and overlaps, unreadable in the
    # shader editor (Q6). Spacing: 300px columns, 200px rows.
    in_node.location = (-600, 0)
    sep.location = (-300, 0)
    sub.location = (0, 120)       # T - feather
    add.location = (0, -120)      # T + feather
    mr.location = (300, 0)
    out_node.location = (600, 0)

    links.new(in_node.outputs["WearTime"], sep.inputs["Vector"])
    links.new(sep.outputs["X"], sub.inputs[0])
    links.new(in_node.outputs["Feather"], sub.inputs[1])
    links.new(sep.outputs["X"], add.inputs[0])
    links.new(in_node.outputs["Feather"], add.inputs[1])
    links.new(in_node.outputs["Wear Amount"], mr.inputs["Value"])
    links.new(sub.outputs[0], mr.inputs["From Min"])
    links.new(add.outputs[0], mr.inputs["From Max"])
    links.new(mr.outputs["Result"], out_node.inputs["Mask"])
    return ng


def _find_node(nt, name, bl_idname):
    n = nt.nodes.get(name)
    if n is None:
        n = nt.nodes.new(bl_idname)
        n.name = name
    return n


def _find_surface_principled(mat):
    """Find a Principled node on the active Material Output's surface branch.

    Falling back to the first top-level Principled keeps simple legacy materials
    working, while the branch walk avoids modifying an unused node when a
    material contains several shader experiments.
    """
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    nt = mat.node_tree
    outputs = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"]
    output = next((n for n in outputs if getattr(n, "is_active_output", False)),
                  outputs[0] if outputs else None)
    stack = []
    if output is not None:
        surface = output.inputs.get("Surface")
        if surface is not None:
            stack.extend(link.from_node for link in surface.links)
    seen = set()
    while stack:
        node = stack.pop()
        ptr = node.as_pointer()
        if ptr in seen:
            continue
        seen.add(ptr)
        if node.type == "BSDF_PRINCIPLED":
            return node
        for socket in node.inputs:
            stack.extend(link.from_node for link in socket.links)
    return next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)


def _preview_material_for_slot(obj, slot, copies):
    """Return an object-local material for a slot, copying authored data once.

    The gaming-console body and lid share one source material.  Mutating that
    material in place makes the unprocessed lid sample the body's AI_WearUV.
    A per-object preview copy keeps the source material and other objects intact
    and is reused on subsequent Replay runs.
    """
    source = slot.material
    if source is None:
        mat = bpy.data.materials.new(f"AIWear_Preview_{obj.name}")
        mat.use_nodes = True
        mat[OWNER_PROP] = obj.name
        slot.material = mat
        return mat
    if source.get(OWNER_PROP) == obj.name:
        return source
    key = source.as_pointer()
    if key in copies:
        slot.material = copies[key]
        return copies[key]
    mat = source.copy()
    mat.name = f"{source.name}_AIWearPreview_{obj.name}"
    mat[OWNER_PROP] = obj.name
    mat[SOURCE_PROP] = source.get(SOURCE_PROP, source.name)
    copies[key] = mat
    slot.material = mat
    return mat


def attach_wear_overlay(obj, weartime_img, worn_img,
                        uv_layer_name: str = "AI_WearUV") -> list[bpy.types.Material]:
    """Composite the AI worn texture over every material slot on ``obj``.

    Injects  bsdf.BaseColor = mix(existing_base_color,
                                  existing_base_color + decode(worn_delta),
                                  gate * worn_tex.alpha)  into the existing
    material's Principled BSDF, where gate = the AIWearMask group's smoothstep
    threshold of WearTime and worn_tex.alpha is reprojected AI wear mask
    (so camera coverage alone can never activate the appearance). The
    rest of the authored material (roughness, metallic, normals) is preserved.
    Authored materials are copied per object before injection, so shared source
    materials are never mutated.  The WearTime and worn textures are explicitly
    sampled with ``uv_layer_name`` instead of relying on Blender's active UV.
    Returns all object-local materials carrying the overlay.
    """
    ensure_node_group()
    if obj is None:
        return []
    if not obj.material_slots:
        obj.data.materials.append(bpy.data.materials.new(f"AIWear_Preview_{obj.name}"))

    copies = {}
    attached = []
    seen = set()
    for slot in obj.material_slots:
        mat = _preview_material_for_slot(obj, slot, copies)
        mat.use_nodes = True
        bsdf = _find_surface_principled(mat)
        if bsdf is not None:
            _inject_overlay(mat, bsdf, weartime_img, worn_img, uv_layer_name)
        else:
            _build_standalone_overlay(mat.node_tree, weartime_img, worn_img,
                                      uv_layer_name)
        ptr = mat.as_pointer()
        if ptr not in seen:
            seen.add(ptr)
            attached.append(mat)
    return attached


def _set_img(node, image, colorspace):
    node.image = image
    # colorspace is set on the image data-block itself (cross-version)
    try:
        image.colorspace_settings.name = colorspace
    except Exception:
        pass


def _inject_overlay(mat, bsdf, weartime_img, worn_img,
                    uv_layer_name: str) -> None:
    """Insert the overlay nodes between the existing base-color source and the
    BSDF's Base Color input (in place, in the EXISTING material's node tree)."""
    nt = mat.node_tree
    links = nt.links

    wt_node = _find_node(nt, WEARTIME_NODE, "ShaderNodeTexImage")
    _set_img(wt_node, weartime_img, "Non-Color")
    worn_node = _find_node(nt, WORNTEX_NODE, "ShaderNodeTexImage")
    _set_img(worn_node, worn_img, "Non-Color")
    uv_node = _find_node(nt, UVMAP_NODE, "ShaderNodeUVMap")
    uv_node.uv_map = uv_layer_name
    grp = _find_node(nt, MASKGROUP_NODE, "ShaderNodeGroup")
    grp.node_tree = bpy.data.node_groups[GROUP_NAME]

    # capture the ORIGINAL base-color source, peeling back any prior overlay
    # mix wired to the BSDF so re-runs don't nest overlays on top of each other.
    bc_input = bsdf.inputs["Base Color"]
    base_src_socket = None
    cur = bc_input.links[0].from_socket if bc_input.links else None
    seen = set()
    while cur is not None and cur.node.name == OVERLAY_MIX and id(cur) not in seen:
        seen.add(id(cur))
        c1 = cur.node.inputs["Color1"]
        cur = c1.links[0].from_socket if c1.links else None
    if cur is not None:
        base_src_socket = cur
    else:
        # BSDF Base Color was unlinked (a flat default) — materialize it so the
        # mix has something to composite over.
        rgb = _find_node(nt, BASE_SRC, "ShaderNodeRGB")
        dv = bc_input.default_value
        rgb.outputs["Color"].default_value = (dv[0], dv[1], dv[2], 1.0)
        base_src_socket = rgb.outputs["Color"]

    # gate * AI mask(alpha): show the residual only where WearTime says worn
    # AND the clean/worn comparison found a real scratch/chip.
    alpha_mul = _find_node(nt, ALPHA_MUL, "ShaderNodeMath")
    alpha_mul.operation = "MULTIPLY"
    delta_sub = _find_node(nt, DELTA_SUB, "ShaderNodeMixRGB")
    delta_sub.blend_type = "SUBTRACT"; delta_sub.inputs[0].default_value = 1.0
    delta_sub.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    delta_scale = _find_node(nt, DELTA_SCALE, "ShaderNodeMixRGB")
    delta_scale.blend_type = "MULTIPLY"; delta_scale.inputs[0].default_value = 1.0
    delta_scale.inputs[2].default_value = (DELTA_DECODE_GAIN,) * 3 + (1.0,)
    base_add = _find_node(nt, BASE_ADD, "ShaderNodeMixRGB")
    base_add.blend_type = "ADD"; base_add.inputs[0].default_value = 1.0
    mix = _find_node(nt, OVERLAY_MIX, "ShaderNodeMixRGB")

    wt_node.extension = "CLIP"
    worn_node.extension = "CLIP"
    links.new(uv_node.outputs["UV"], wt_node.inputs["Vector"])
    links.new(uv_node.outputs["UV"], worn_node.inputs["Vector"])
    links.new(wt_node.outputs["Color"], grp.inputs["WearTime"])
    links.new(grp.outputs["Mask"], alpha_mul.inputs[0])
    links.new(worn_node.outputs["Alpha"], alpha_mul.inputs[1])
    links.new(worn_node.outputs["Color"], delta_sub.inputs[1])
    links.new(delta_sub.outputs["Color"], delta_scale.inputs[1])
    links.new(base_src_socket, base_add.inputs[1])
    links.new(delta_scale.outputs["Color"], base_add.inputs[2])
    links.new(alpha_mul.outputs[0], mix.inputs["Fac"])
    links.new(base_src_socket, mix.inputs["Color1"])
    links.new(base_add.outputs["Color"], mix.inputs["Color2"])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

    # grid layout (Q6 — don't pile at origin); leave existing nodes in place
    uv_node.location = (-1750, 200)
    wt_node.location = (-1400, 400)
    worn_node.location = (-1400, 0)
    grp.location = (-1050, 400)
    delta_sub.location = (-1050, -50)
    delta_scale.location = (-750, -50)
    base_add.location = (-450, -50)
    alpha_mul.location = (-700, 300)
    mix.location = (-100, 200)
    if base_src_socket.node is not None and base_src_socket.node.name == BASE_SRC:
        base_src_socket.node.location = (-1400, -400)


def _build_standalone_overlay(nt, weartime_img, worn_img,
                              uv_layer_name: str) -> None:
    """Fallback: build a self-contained mid-gray material plus wear residual."""
    nt.nodes.clear()
    links = nt.links
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    wt_node = nt.nodes.new("ShaderNodeTexImage"); wt_node.name = WEARTIME_NODE
    _set_img(wt_node, weartime_img, "Non-Color")
    worn_node = nt.nodes.new("ShaderNodeTexImage"); worn_node.name = WORNTEX_NODE
    _set_img(worn_node, worn_img, "Non-Color")
    uv_node = nt.nodes.new("ShaderNodeUVMap"); uv_node.name = UVMAP_NODE
    uv_node.uv_map = uv_layer_name
    grp = nt.nodes.new("ShaderNodeGroup"); grp.node_tree = bpy.data.node_groups[GROUP_NAME]
    grp.name = MASKGROUP_NODE
    base = nt.nodes.new("ShaderNodeRGB")
    base.outputs["Color"].default_value = (0.6, 0.6, 0.6, 1)
    alpha_mul = nt.nodes.new("ShaderNodeMath"); alpha_mul.operation = "MULTIPLY"
    alpha_mul.name = ALPHA_MUL
    delta_sub = nt.nodes.new("ShaderNodeMixRGB"); delta_sub.name = DELTA_SUB
    delta_sub.blend_type = "SUBTRACT"; delta_sub.inputs[0].default_value = 1.0
    delta_sub.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    delta_scale = nt.nodes.new("ShaderNodeMixRGB"); delta_scale.name = DELTA_SCALE
    delta_scale.blend_type = "MULTIPLY"; delta_scale.inputs[0].default_value = 1.0
    delta_scale.inputs[2].default_value = (DELTA_DECODE_GAIN,) * 3 + (1.0,)
    base_add = nt.nodes.new("ShaderNodeMixRGB"); base_add.name = BASE_ADD
    base_add.blend_type = "ADD"; base_add.inputs[0].default_value = 1.0
    mix = nt.nodes.new("ShaderNodeMixRGB"); mix.name = OVERLAY_MIX

    wt_node.extension = "CLIP"; worn_node.extension = "CLIP"
    links.new(uv_node.outputs["UV"], wt_node.inputs["Vector"])
    links.new(uv_node.outputs["UV"], worn_node.inputs["Vector"])
    links.new(wt_node.outputs["Color"], grp.inputs["WearTime"])
    links.new(grp.outputs["Mask"], alpha_mul.inputs[0])
    links.new(worn_node.outputs["Alpha"], alpha_mul.inputs[1])
    links.new(worn_node.outputs["Color"], delta_sub.inputs[1])
    links.new(delta_sub.outputs["Color"], delta_scale.inputs[1])
    links.new(base.outputs["Color"], base_add.inputs[1])
    links.new(delta_scale.outputs["Color"], base_add.inputs[2])
    links.new(alpha_mul.outputs[0], mix.inputs["Fac"])
    links.new(base.outputs["Color"], mix.inputs["Color1"])
    links.new(base_add.outputs["Color"], mix.inputs["Color2"])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    uv_node.location = (-1350, 150)
    wt_node.location = (-1050, 300); worn_node.location = (-1050, 0)
    grp.location = (-700, 350); alpha_mul.location = (-350, 350)
    delta_sub.location = (-700, 0); delta_scale.location = (-400, 0)
    base.location = (-700, -250); base_add.location = (-100, -50)
    mix.location = (200, 150); bsdf.location = (500, 150); out.location = (800, 150)


def _group_node(mat):
    if mat is None or not mat.use_nodes:
        return None
    return mat.node_tree.nodes.get(MASKGROUP_NODE)


def set_amount(mat, amount01: float):
    g = _group_node(mat)
    if g:
        g.inputs["Wear Amount"].default_value = amount01


def set_feather(mat, feather01: float):
    g = _group_node(mat)
    if g:
        g.inputs["Feather"].default_value = max(float(feather01), FEATHER_EPS)


def update_active(context, amount: float, feather: float):
    """Property-update hook: refresh the active object's overlay material(s).

    The overlay is injected into the object's EXISTING material (unknown name),
    not a fixed 'AIWear_Preview', so scan the active object's material slots for
    the one carrying the AIWearMask group node and update it.
    """
    obj = getattr(context, "active_object", None)
    if obj is None:
        return
    for slot in obj.material_slots:
        mat = slot.material
        if mat is not None and _group_node(mat) is not None:
            set_amount(mat, amount)
            set_feather(mat, feather)


# Property callbacks (referenced from properties.py)
def on_wear_amount(self, context):
    update_active(context, self.wear_amount / 100.0, self.feather / 100.0)


def on_feather(self, context):
    update_active(context, self.wear_amount / 100.0, self.feather / 100.0)
