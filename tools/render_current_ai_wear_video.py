"""Render a 10-second AI Wear showcase from the currently open Blender scene.

Shot 1 (6 s): orbit once around world X from the Auto-8 oblique direction.
Shot 2 (4 s): hold the camera and animate the actual AIWearMask/Wear Amount
input from 0.6 to 0 (the UI's 60 to 0), never a material-wide alpha.

The scene HDRI remains connected for lighting and reflections, while camera
rays see a neutral middle-gray World background.

Run inside the open Blender process with no arguments. For composition QA from
the command line, pass ``-- --preview`` to render frame 1 as a PNG only.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Vector


PREFIX = "AIWearVideo_"
FPS = 24
ORBIT_SECONDS = 6
RAMP_SECONDS = 4
RESOLUTION_X = 1920
RESOLUTION_Y = 1080


def _visible_meshes(scene):
    return [obj for obj in scene.objects
            if obj.type == "MESH" and not obj.hide_render and obj.visible_get()]


def _world_bounds(objects):
    points = [obj.matrix_world @ Vector(corner)
              for obj in objects for corner in obj.bound_box]
    low = Vector((min(p.x for p in points), min(p.y for p in points),
                  min(p.z for p in points)))
    high = Vector((max(p.x for p in points), max(p.y for p in points),
                   max(p.z for p in points)))
    return low, high


def _object(name, data=None):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
    return obj


def _find_wear_inputs(meshes):
    sockets = []
    seen = set()
    for obj in meshes:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or not material.use_nodes or material.node_tree is None:
                continue
            node = material.node_tree.nodes.get("AIWear_MaskGroup")
            if node is None or node.type != "GROUP":
                continue
            socket = node.inputs.get("Wear Amount")
            if socket is not None and socket.as_pointer() not in seen:
                seen.add(socket.as_pointer())
                sockets.append(socket)
    return sockets


def _set_camera_gray_world(scene):
    """Keep the World shader for lighting, but show gray to camera rays."""
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new(PREFIX + "World")
        scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    output = next((node for node in tree.nodes
                   if node.type == "OUTPUT_WORLD" and node.is_active_output), None)
    if output is None:
        output = tree.nodes.new("ShaderNodeOutputWorld")
    surface = output.inputs.get("Surface")
    if surface is None:
        raise RuntimeError("The World output has no Surface input")

    # Make reruns idempotent: unwrap our previous camera-ray mix first.
    old_mix = tree.nodes.get(PREFIX + "WorldCameraMix")
    if old_mix is not None and surface.is_linked:
        current = surface.links[0].from_node
        if current == old_mix and old_mix.inputs[1].is_linked:
            source = old_mix.inputs[1].links[0].from_socket
            tree.links.new(source, surface)
    for name in (PREFIX + "WorldCameraMix", PREFIX + "WorldCameraGray",
                 PREFIX + "WorldLightPath"):
        node = tree.nodes.get(name)
        if node is not None:
            tree.nodes.remove(node)

    source = surface.links[0].from_socket if surface.is_linked else None
    if source is None:
        fallback = tree.nodes.new("ShaderNodeBackground")
        fallback.name = PREFIX + "WorldHDRIFallback"
        fallback.inputs[0].default_value = (*world.color, 1.0)
        source = fallback.outputs[0]

    light_path = tree.nodes.new("ShaderNodeLightPath")
    light_path.name = PREFIX + "WorldLightPath"
    gray = tree.nodes.new("ShaderNodeBackground")
    gray.name = PREFIX + "WorldCameraGray"
    gray.inputs[0].default_value = (0.18, 0.18, 0.18, 1.0)
    gray.inputs[1].default_value = 1.0
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.name = PREFIX + "WorldCameraMix"
    tree.links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
    tree.links.new(source, mix.inputs[1])
    tree.links.new(gray.outputs[0], mix.inputs[2])
    tree.links.new(mix.outputs[0], surface)


def _linearize(owner):
    animation = getattr(owner, "animation_data", None)
    action = animation.action if animation else None
    if action is None:
        return
    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


scene = bpy.context.scene
if not bpy.data.filepath:
    raise RuntimeError("Save the current .blend before rendering the video")

meshes = _visible_meshes(scene)
if not meshes:
    raise RuntimeError("The current scene contains no visible renderable mesh")
wear_inputs = _find_wear_inputs(meshes)
if not wear_inputs:
    raise RuntimeError("No AIWear_MaskGroup/Wear Amount input exists in the current scene")

low, high = _world_bounds(meshes)
centre = 0.5 * (low + high)
radius = max(0.5 * (high - low).length, 1e-4)

# Camera starts at the same (-1,-1,+1) oblique direction used by Auto 8. Its
# parent rotates around world X, retaining the X offset and therefore the
# oblique side view throughout the orbit.
target = _object(PREFIX + "Target")
target.location = centre
orbit = _object(PREFIX + "OrbitX")
orbit.location = centre
orbit.rotation_mode = "XYZ"

camera = bpy.data.objects.get(PREFIX + "Camera")
if camera is None or camera.type != "CAMERA":
    camera_data = bpy.data.cameras.new(PREFIX + "CameraData")
    camera = _object(PREFIX + "Camera", camera_data)
camera.data.lens = 50.0
camera.data.sensor_width = 36.0
camera.data.clip_start = max(radius * 0.002, 0.001)
camera.data.clip_end = max(radius * 30.0, 100.0)

fov_x = 2.0 * math.atan(camera.data.sensor_width / (2.0 * camera.data.lens))
fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) /
                        (RESOLUTION_X / RESOLUTION_Y))
distance = radius / max(math.sin(min(fov_x, fov_y) * 0.5), 1e-4) * 1.08
direction = Vector((-1.0, -1.0, 1.0)).normalized()
camera.parent = orbit
camera.matrix_parent_inverse.identity()
camera.location = direction * distance
camera.rotation_euler = (0.0, 0.0, 0.0)

for constraint in list(camera.constraints):
    if constraint.name.startswith(PREFIX):
        camera.constraints.remove(constraint)
track = camera.constraints.new("TRACK_TO")
track.name = PREFIX + "TrackTarget"
track.target = target
track.track_axis = "TRACK_NEGATIVE_Z"
track.up_axis = "UP_Y"
scene.camera = camera

orbit.animation_data_clear()
orbit.rotation_euler = (0.0, 0.0, 0.0)
orbit.keyframe_insert("rotation_euler", index=0, frame=1)
orbit.rotation_euler.x = math.tau
orbit.keyframe_insert("rotation_euler", index=0, frame=FPS * ORBIT_SECONDS)
_linearize(orbit)

orbit_end = FPS * ORBIT_SECONDS
ramp_start = orbit_end + 1
final_frame = FPS * (ORBIT_SECONDS + RAMP_SECONDS)
for socket in wear_inputs:
    # The first shot displays the current final result. The fixed-camera shot
    # then reduces the real threshold input linearly from 0.6 to zero.
    for frame, value in ((1, 0.6), (orbit_end, 0.6),
                         (ramp_start, 0.6), (final_frame, 0.0)):
        socket.default_value = value
        socket.keyframe_insert("default_value", frame=frame)
    _linearize(socket.id_data)

settings = getattr(scene, "ai_wear", None)
if settings is not None:
    for frame, value in ((1, 60.0), (orbit_end, 60.0),
                         (ramp_start, 60.0), (final_frame, 0.0)):
        settings.wear_amount = value
        settings.keyframe_insert("wear_amount", frame=frame)
    _linearize(scene)

for marker in list(scene.timeline_markers):
    if marker.name.startswith(PREFIX):
        scene.timeline_markers.remove(marker)
scene.timeline_markers.new(PREFIX + "Orbit_6s", frame=1)
scene.timeline_markers.new(PREFIX + "WearAmount_60_to_0_4s", frame=ramp_start)

# Keep the HDRI shader for non-camera rays, so illumination and reflections
# match the Rendered viewport while the visible background stays neutral.
_set_camera_gray_world(scene)

scene.frame_start = 1
scene.frame_end = final_frame
scene.render.fps = FPS
scene.render.fps_base = 1.0
scene.render.resolution_x = RESOLUTION_X
scene.render.resolution_y = RESOLUTION_Y
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"
scene.render.image_settings.compression = 15
scene.render.use_file_extension = True
if scene.render.engine not in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
    available = {item.identifier for item in
                 bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    scene.render.engine = ("BLENDER_EEVEE" if "BLENDER_EEVEE" in available
                           else "BLENDER_EEVEE_NEXT")

output_dir = Path(bpy.data.filepath).parent / "ai_wear_video"
output_dir.mkdir(parents=True, exist_ok=True)
stem = Path(bpy.data.filepath).stem
video_path = output_dir / f"{stem}_ai_wear_demo.mp4"
preview_path = output_dir / f"{stem}_ai_wear_preview.png"
frame_dir = output_dir / "_frames"
status_path = output_dir / "video_render_status.json"

preview_only = "--preview" in sys.argv
status = {
    "blend": bpy.data.filepath,
    "axis": "world X",
    "fps": FPS,
    "orbit_frames": [1, orbit_end],
    "wear_amount_frames": [ramp_start, final_frame],
    "wear_amount_values": [60, 0],
    "animated_node_inputs": len(wear_inputs),
    "resolution": [RESOLUTION_X, RESOLUTION_Y],
    "output": str(preview_path if preview_only else video_path),
    "state": "rendering",
}
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

if preview_only:
    scene.frame_set(1)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(preview_path)
    bpy.ops.render.render(write_still=True)
else:
    frame_dir.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(frame_dir / f"{stem}_")
    bpy.ops.render.render(animation=True)
    ffmpeg = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
    if not Path(ffmpeg).is_file():
        raise RuntimeError("FFmpeg is unavailable; PNG frames remain in " + str(frame_dir))
    subprocess.run([
        ffmpeg, "-y", "-framerate", str(FPS), "-start_number", "1",
        "-i", str(frame_dir / f"{stem}_%04d.png"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path),
    ], check=True)
    if frame_dir.resolve().parent == output_dir.resolve():
        shutil.rmtree(frame_dir)

status["state"] = "complete"
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
scene.frame_set(final_frame)
print("AIWEAR_VIDEO_OUTPUT=" + status["output"])
