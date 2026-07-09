"""Reusable bpy script fragments for the Blender bridge.

These are Python source strings run inside Blender via `runner.run_script`. Keeping
them here (not inline) makes the common building blocks reusable and testable.
"""
from __future__ import annotations

# Prelude: clean scene + helpers available to every script we run.
PRELUDE = r'''
import bpy, sys, math, mathutils

def _argv():
    return sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def set_engine(preferred=("BLENDER_EEVEE_NEXT","BLENDER_EEVEE","CYCLES")):
    sc = bpy.context.scene
    for e in preferred:
        try:
            sc.render.engine = e
            return e
        except Exception:
            continue
    return sc.render.engine

def add_camera(location, look_at=(0,0,0), lens=50):
    cam_data = bpy.data.cameras.new("Camera"); cam_data.lens = lens
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    d = mathutils.Vector(look_at) - mathutils.Vector(location)
    cam.rotation_euler = d.to_track_quat('-Z','Y').to_euler()
    bpy.context.scene.camera = cam
    return cam

def add_sun(energy=4.0, angle=(0.6,0.2,0.0)):
    l = bpy.data.lights.new("Sun", type='SUN'); l.energy = energy
    o = bpy.data.objects.new("Sun", l); bpy.context.collection.objects.link(o)
    o.rotation_euler = angle
    return o

def render_to(path, x=512, y=512, samples=None):
    sc = bpy.context.scene
    sc.render.resolution_x = x; sc.render.resolution_y = y
    sc.render.image_settings.file_format = 'PNG'
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("RENDERED", sc.render.engine, path)

def enable_gpu(samples=1):
    """Force Cycles onto the GPU (OptiX/CUDA) with low samples. Headless Blender under
    --factory-startup otherwise CPU-renders (no compute device enabled) -> 600s hangs.
    Depth is noise-free geometry so samples=1 is plenty + fast. Safe no-op on failure."""
    sc = bpy.context.scene
    try:
        sc.render.engine = 'CYCLES'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        ok = 0
        for dt in ('OPTIX', 'CUDA'):
            try:
                prefs.compute_device_type = dt; prefs.get_devices()
                ok = sum(1 for d in prefs.devices if d.type == dt)
                if ok: break
            except Exception:
                continue
        for d in prefs.devices:
            d.use = (d.type in ('OPTIX', 'CUDA'))
        sc.cycles.device = 'GPU'
        sc.cycles.samples = samples
        try: sc.cycles.use_denoising = False
        except Exception: pass
        print('GPU_CFG', prefs.compute_device_type, 'gpu=', ok)
    except Exception as e:
        print('GPU_CFG_FAIL', e)
    return sc.render.engine
'''


# A reusable demo scene body (assumes PRELUDE helpers + reset already run).
_DEMO_SCENE = r'''
bpy.ops.mesh.primitive_plane_add(size=8, location=(0,0,-1))
bpy.ops.mesh.primitive_monkey_add(location=(0,0,0))
bpy.ops.object.shade_smooth()
suz = bpy.context.active_object
mat = bpy.data.materials.new("Hero"); mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.55, 0.3, 0.85, 1.0)
    if "Roughness" in bsdf.inputs: bsdf.inputs["Roughness"].default_value = 0.3
suz.data.materials.append(mat)
add_camera((4,-4,2.5), (0,0,0.3), lens=50)
add_sun(5.0)
bpy.context.scene.world = bpy.data.worlds.new("W")
bpy.context.scene.world.use_nodes = True
'''


def smoke_render() -> str:
    """A self-contained scene (Suzanne + ground + camera + sun) rendered to argv[0]."""
    return PRELUDE + "out = _argv()[0]\nreset_scene()\neng = set_engine()\n" + _DEMO_SCENE + r'''
render_to(out, 640, 640)
print("ENGINE_USED", eng)
print("OBJECTS", len(bpy.data.objects))
'''


def depth_pass() -> str:
    """Render the demo scene's BEAUTY (argv[0]) + a normalized DEPTH map (argv[1]).

    Depth is the canonical ControlNet conditioning for "AI restyles the 3D while
    geometry stays locked". Output: near=bright, far=dark (MiDaS-style), grayscale.
    Depth is routed to the Composite output and rendered with a Raw view transform
    (so the linear depth isn't tonemapped). Blender 5.x compositor = node group.
    """
    return PRELUDE + "beauty, depth_out = _argv()[0], _argv()[1]\nreset_scene()\neng = set_engine()\n" + _DEMO_SCENE + r'''
sc = bpy.context.scene
vl = bpy.context.view_layer
# 1) beauty (no compositing)
sc.render.use_compositing = False
render_to(beauty, 640, 640)
# 2) depth: enable Z pass, route Depth -> Normalize -> Invert -> Composite
vl.use_pass_z = True
try: sc.view_settings.view_transform = 'Raw'
except Exception: pass
sc.render.use_compositing = True
ng = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
sc.compositing_node_group = ng
nodes, links = ng.nodes, ng.links
for n in list(nodes): nodes.remove(n)
rl = nodes.new('CompositorNodeRLayers')
norm = nodes.new('CompositorNodeNormalize')   # auto 0..1 over the depth pass
inv = nodes.new('CompositorNodeInvert')       # near = bright (MiDaS-style)
# the scene compositor is a node group -> output goes through a Group Output
ng.interface.new_socket(name="Image", in_out='OUTPUT', socket_type='NodeSocketColor')
gout = nodes.new('NodeGroupOutput')
ds = rl.outputs.get('Depth') or rl.outputs.get('Z')
links.new(ds, norm.inputs[0])
links.new(norm.outputs[0], inv.inputs['Color'])
links.new(inv.outputs['Color'], gout.inputs[0])
sc.render.image_settings.color_mode = 'BW'
render_to(depth_out, 640, 640)
print("DEPTH", depth_out)
print("ENGINE_USED", eng)
'''


# Shared depth-sequence tail: FIXED near/far linear map (ShaderNodeMath, stable
# across frames -> temporally consistent) -> render the animation as depth PNGs.
# Each SCENE body must build geometry + camera, set sc.frame_start/end, and keyframe.
_DEPTH_TAIL = r'''
sc = bpy.context.scene
vl = bpy.context.view_layer
vl.use_pass_z = True
try: sc.view_settings.view_transform = 'Raw'
except Exception: pass
sc.render.use_compositing = True
ng = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
sc.compositing_node_group = ng
nodes, links = ng.nodes, ng.links
for n in list(nodes): nodes.remove(n)
rl = nodes.new('CompositorNodeRLayers')
sub = nodes.new('ShaderNodeMath'); sub.operation = 'SUBTRACT'; sub.inputs[1].default_value = near
div = nodes.new('ShaderNodeMath'); div.operation = 'DIVIDE'; div.inputs[1].default_value = (far - near); div.use_clamp = True
invn = nodes.new('ShaderNodeMath'); invn.operation = 'SUBTRACT'; invn.inputs[0].default_value = 1.0
ng.interface.new_socket(name="Image", in_out='OUTPUT', socket_type='NodeSocketColor')
gout = nodes.new('NodeGroupOutput')
ds = rl.outputs.get('Depth') or rl.outputs.get('Z')
links.new(ds, sub.inputs[0])
links.new(sub.outputs[0], div.inputs[0])
links.new(div.outputs[0], invn.inputs[1])      # 1.0 - normalized
links.new(invn.outputs[0], gout.inputs[0])
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_mode = 'BW'
sc.render.resolution_x = 768; sc.render.resolution_y = 768
sc.render.filepath = depth_dir.rstrip('/\\') + '/depth_'
bpy.ops.render.render(animation=True)
print("SEQ_DONE", sc.frame_end, depth_dir)
print("ENGINE_USED", eng)
'''

# --- animated scene bodies (geometry only; depth ignores materials/lights) -------
_BODY_MONKEY_ORBIT = r'''
bpy.ops.mesh.primitive_plane_add(size=8, location=(0,0,-1))
bpy.ops.mesh.primitive_monkey_add(location=(0,0,0))
bpy.ops.object.shade_smooth()
cam = add_camera((4,-4,2.6), (0,0,0.3), 50)
sc = bpy.context.scene; sc.frame_start = 1; sc.frame_end = frames
look = mathutils.Vector((0,0,0.3)); R, H = 6.5, 2.6
for f in range(1, frames+1):
    sc.frame_set(f); t = (f-1)/max(1, frames-1)
    ang = math.radians(-80 + 60*t); rad = R - 1.0*t
    cam.location = (rad*math.cos(ang), rad*math.sin(ang), H)
    d = look - cam.location; cam.rotation_euler = d.to_track_quat('-Z','Y').to_euler()
    cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_euler", frame=f)
'''

_BODY_TURNTABLE = r'''
bpy.ops.mesh.primitive_plane_add(size=8, location=(0,0,-1.5))
bpy.ops.mesh.primitive_torus_add(location=(0,0,0.0), major_radius=1.5, minor_radius=0.42)
ring = bpy.context.active_object; bpy.ops.object.shade_smooth()
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.75, location=(0,0,0.0))
orb = bpy.context.active_object; bpy.ops.object.shade_smooth(); orb.parent = ring
add_camera((0,-6.0,1.8), (0,0,0.0), 50)
sc = bpy.context.scene; sc.frame_start = 1; sc.frame_end = frames
for f in range(1, frames+1):
    sc.frame_set(f)
    ring.rotation_euler = (math.radians(18), 0, math.radians(360*(f-1)/frames))
    ring.keyframe_insert("rotation_euler", frame=f)
'''

_BODY_CITY = r'''
bpy.ops.mesh.primitive_plane_add(size=60, location=(0,0,0))
for ix in range(-4, 5):
    if ix == 0: continue          # clear central avenue for the fly-through
    for iy in range(0, 16):
        h = 0.8 + ((ix*ix*3 + iy*7) % 11) * 0.95
        bpy.ops.mesh.primitive_cube_add(size=1, location=(ix*1.7, iy*1.7, h/2))
        bpy.context.active_object.scale = (0.7, 0.7, h)
cam = add_camera((0,-3,1.4), (0,8,1.1), 32)
sc = bpy.context.scene; sc.frame_start = 1; sc.frame_end = frames
for f in range(1, frames+1):
    sc.frame_set(f); t = (f-1)/max(1, frames-1); y = -3 + t*13.0
    cam.location = (0, y, 1.4)
    look = mathutils.Vector((0, y+8, 1.0)); d = look - cam.location
    cam.rotation_euler = d.to_track_quat('-Z','Y').to_euler()
    cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_euler", frame=f)
'''

# A PRODUCT turntable: a perfume-bottle proxy on a pedestal that spins on a seamless
# studio backdrop (infinity cove). Geometry is just a silhouette for the depth pass;
# VACE restyles it into a photoreal glass product. The floor + backdrop stay static
# (only the platform spins), so the depth reads as a real product-photography set.
_BODY_PRODUCT = r'''
# static studio set: floor + seamless vertical backdrop (infinity cove)
bpy.ops.mesh.primitive_plane_add(size=40, location=(0,0,0))
bpy.ops.mesh.primitive_plane_add(size=40, location=(0,8,8), rotation=(math.radians(90),0,0))
# rotating platform (empty parent)
bpy.ops.object.empty_add(location=(0,0,0))
turn = bpy.context.active_object
# pedestal / podium
bpy.ops.mesh.primitive_cylinder_add(radius=1.4, depth=0.5, location=(0,0,-0.25))
ped = bpy.context.active_object; bpy.ops.object.shade_smooth(); ped.parent = turn
# bottle body (rounded rectangular prism)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0.48))
body = bpy.context.active_object; body.scale = (0.5, 0.28, 0.95)
bev = body.modifiers.new("bevel", "BEVEL"); bev.width = 0.07; bev.segments = 4
bpy.ops.object.shade_smooth(); body.parent = turn
# neck
bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=0.2, location=(0,0,1.055))
neck = bpy.context.active_object; bpy.ops.object.shade_smooth(); neck.parent = turn
# cap
bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,1.335))
cap = bpy.context.active_object; cap.scale = (0.17, 0.17, 0.18); cap.parent = turn
# spin the platform a full turn
sc = bpy.context.scene; sc.frame_start = 1; sc.frame_end = frames
for f in range(1, frames+1):
    sc.frame_set(f)
    turn.rotation_euler = (0, 0, math.radians(360*(f-1)/frames))
    turn.keyframe_insert("rotation_euler", frame=f)
# static camera, slightly elevated, tight product framing
add_camera((0,-5.5,1.7), (0,0,0.7), 60)
'''

SCENE_BODIES = {
    "monkey_orbit": _BODY_MONKEY_ORBIT,
    "sculpture_turntable": _BODY_TURNTABLE,
    "city_flythrough": _BODY_CITY,
    "product_turntable": _BODY_PRODUCT,
}


# Casting per scene: FIXED near/far depth normalization (stable across frames) + a
# fitting default style prompt + human labels. Shared by the CLI runner, the web UI
# (/api/blender/scenes) and the MCP tool, so all three drive Blender identically.
SCENE_PRESETS: dict[str, dict] = {
    "monkey_orbit": {
        "label": "Marble bust (camera orbit)", "motion": "camera orbits a hero object",
        "near": 3.0, "far": 11.0, "prompt": (
            "a photorealistic carved white marble bust of a monkey on a pedestal, bright museum "
            "gallery lighting, soft shadows, intricate chiselled detail, sharp focus, high detail")},
    "sculpture_turntable": {
        "label": "Sculpture (turntable spin)", "motion": "subject rotates on a turntable",
        "near": 3.2, "far": 9.5, "prompt": (
            "a polished bronze and gold abstract sculpture, a ring around an orb, brightly lit "
            "studio, reflective burnished metal, rim light, museum backdrop, ultra detailed, sharp focus")},
    "city_flythrough": {
        "label": "City street (fly-through)", "motion": "camera flies down a street",
        "near": 2.0, "far": 18.0, "prompt": (
            "a vibrant neon cyberpunk city street at night, glowing pink and cyan signs, bright "
            "neon glow, rain-soaked reflective street, blade runner, cinematic, ultra detailed, high contrast")},
    "product_turntable": {
        "label": "Product bottle (turntable)", "motion": "product spins on a studio pedestal",
        "near": 3.8, "far": 9.0, "prompt": (
            "a luxury perfume bottle on a polished round pedestal, glossy reflective glass with amber "
            "liquid, metallic gold cap, elegant studio product photography, soft seamless gradient "
            "backdrop, dramatic rim lighting and soft shadows, commercial advertisement, photorealistic, "
            "ultra detailed, sharp focus, high resolution")},
}


def scene_list() -> list[dict]:
    """Public scene catalogue for the UI / MCP tool: [{key,label,motion,prompt,near,far}]."""
    return [{"key": k, **{f: v[f] for f in ("label", "motion", "prompt", "near", "far")}}
            for k, v in SCENE_PRESETS.items()]


def depth_sequence(scene: str = "monkey_orbit") -> str:
    """Animated DEPTH SEQUENCE for a named scene -> argv[0] dir.

    argv: [depth_dir, frames, near, far]. FIXED near/far normalization keeps depth
    brightness stable across frames (temporally consistent ControlNet conditioning).
    Scenes: monkey_orbit | sculpture_turntable | city_flythrough.
    """
    body = SCENE_BODIES.get(scene, _BODY_MONKEY_ORBIT)
    return PRELUDE + r'''
import math
a = _argv()
depth_dir = a[0]
frames = int(a[1]) if len(a) > 1 else 16
near = float(a[2]) if len(a) > 2 else 3.0
far  = float(a[3]) if len(a) > 3 else 11.0
reset_scene()
eng = set_engine()
eng = enable_gpu()   # GPU depth (OptiX/CUDA, samples=1) — else CPU-hangs headless
''' + body + _DEPTH_TAIL
