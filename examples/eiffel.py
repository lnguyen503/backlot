"""[MODEL] Procedural Eiffel Tower in Blender (bpy) — recognizable silhouette + iron lattice.
  python eiffel.py preview   # build + 1 front still (check silhouette, cheap)
  python eiffel.py full      # build + save .blend + GLB/OBJ + turntable mp4 + 3 stills
Runs Blender directly on GPU (Cycles/OptiX).
"""
import sys, time, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backlot.blender import runner as bl
import imageio_ffmpeg

OUT = Path(__file__).resolve().parents[1] / "runs/eiffel"
OUT.mkdir(parents=True, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()

# ---- the bpy builder script ----
BUILD = r'''
import bpy, math, mathutils
from mathutils import Vector

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
reset()

MK = []  # collected strut objects
def beam(p1, p2, r=0.9):
    p1 = Vector(p1); p2 = Vector(p2); d = p2 - p1; L = d.length
    if L < 1e-3: return
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=L, vertices=6, location=(p1 + p2) / 2)
    o = bpy.context.active_object
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = d.to_track_quat('Z', 'Y')
    MK.append(o)

def box(cx, cy, cz, sx, sy, sz):
    bpy.ops.mesh.primitive_cube_add(location=(cx, cy, cz))
    o = bpy.context.active_object; o.scale = (sx/2, sy/2, sz/2); MK.append(o)

# --- profile: horizontal offset of a leg center from the axis, vs height z (0..H2) ---
H1, H2, HTOP, HSPIRE = 57.0, 115.0, 300.0, 330.0
RBASE, RMERGE = 55.0, 13.0          # leg-center offset at ground / at 2nd platform
def rleg(z):
    t = max(0.0, min(1.0, z / H2))
    return RMERGE + (RBASE - RMERGE) * (1.0 - t) ** 1.7   # convex-out splay (Eiffel curve)
def rtower(z):                       # single tower above H2, taper to spire
    t = (z - H2) / (HTOP - H2)
    return 11.0 + (3.0 - 11.0) * t
def legw(z):                         # each leg's own half cross-section
    return 8.0 - 5.0 * min(1.0, z / H2)

CORNERS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

# --- 4 curved lattice legs (ground -> 2nd platform) ---
SEG = 12
for (sx, sy) in CORNERS:
    zs = [H2 * i / SEG for i in range(SEG + 1)]
    for i in range(SEG):
        z0, z1 = zs[i], zs[i + 1]
        r0, r1 = rleg(z0), rleg(z1); w0, w1 = legw(z0), legw(z1)
        c0 = Vector((sx * r0, sy * r0, z0)); c1 = Vector((sx * r1, sy * r1, z1))
        # 4 rails of the leg cross-section (square around the leg center)
        rails = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        pts0 = [c0 + Vector((rx * w0, ry * w0, 0)) for rx, ry in rails]
        pts1 = [c1 + Vector((rx * w1, ry * w1, 0)) for rx, ry in rails]
        for a in range(4):
            beam(pts0[a], pts1[a], 0.9)                    # vertical rails
        # horizontal ring + X-braces on the 4 faces of the leg
        face_pairs = [(0, 1), (1, 3), (3, 2), (2, 0)]
        for a, b in face_pairs:
            beam(pts1[a], pts1[b], 0.6)                     # top ring
            beam(pts0[a], pts1[b], 0.5); beam(pts0[b], pts1[a], 0.5)   # X

# --- 4 grand ARCHES: one per face, spanning between the two adjacent legs, curving up ---
SIDE_PAIRS = [((1, 1), (1, -1)), ((1, -1), (-1, -1)), ((-1, -1), (-1, 1)), ((-1, 1), (1, 1))]
ZF, AZ = 9.0, 42.0                       # arch springs at ZF, apex AZ
for (a, b) in SIDE_PAIRS:
    pa = Vector((a[0] * rleg(ZF), a[1] * rleg(ZF), ZF))
    pb = Vector((b[0] * rleg(ZF), b[1] * rleg(ZF), ZF))
    N = 18
    # two parallel arch rails (inner/outer of the leg width) + rungs = a thick arch band
    for off in (-3.0, 3.0):
        prev = None
        for j in range(N + 1):
            u = j / N
            p = pa.lerp(pb, u)
            # bulge z up as a sine arch; pull slightly inward toward the axis at the apex
            p.z = ZF + (AZ - ZF) * math.sin(math.pi * u)
            shrink = 1.0 - 0.12 * math.sin(math.pi * u)
            p.x *= shrink; p.y *= shrink
            p += Vector((0, 0, 0)) if off == 0 else Vector((0, 0, 0))
            # offset the two rails along the face-tangent so the band has depth (in +z)
            p.z += off * 0.0
            pp = p + Vector((0, 0, off))
            if prev is not None: beam(prev, pp, 0.7)
            prev = pp
    # rungs across the arch band
    for j in range(0, N + 1, 2):
        u = j / N
        p = pa.lerp(pb, u); p.z = ZF + (AZ - ZF) * math.sin(math.pi * u)
        shrink = 1.0 - 0.12 * math.sin(math.pi * u); p.x *= shrink; p.y *= shrink
        beam(p + Vector((0, 0, -3)), p + Vector((0, 0, 3)), 0.5)

# --- platforms (deck slabs, lattice-edged) ---
def platform(z, half, th=2.2):
    box(0, 0, z, half * 2, half * 2, th)
    c = [(half, half), (half, -half), (-half, -half), (-half, half)]
    for a in range(4):
        beam((c[a][0], c[a][1], z - 3), (c[a][0], c[a][1], z + 3), 0.7)
        beam((c[a][0], c[a][1], z), (c[(a+1)%4][0], c[(a+1)%4][1], z), 0.6)
platform(H1, rleg(H1) + 8, 2.4)
platform(H2, rleg(H2) + 5, 2.0)

# --- central tapering tower (H2 -> HTOP) ---
SEG2 = 16
for i in range(SEG2):
    z0 = H2 + (HTOP - H2) * i / SEG2; z1 = H2 + (HTOP - H2) * (i + 1) / SEG2
    r0, r1 = rtower(z0), rtower(z1)
    p0 = [Vector((rx * r0, ry * r0, z0)) for rx, ry in CORNERS]
    p1 = [Vector((rx * r1, ry * r1, z1)) for rx, ry in CORNERS]
    order = [0, 1, 3, 2]
    for a in range(4):
        beam(p0[order[a]], p1[order[a]], 0.6)
    for a in range(4):
        b = (a + 1) % 4
        beam(p1[order[a]], p1[order[b]], 0.4)
        beam(p0[order[a]], p1[order[b]], 0.35); beam(p0[order[b]], p1[order[a]], 0.35)

# --- top platform + spire/antenna ---
platform(HTOP, 4.0, 1.4)
beam((0, 0, HTOP), (0, 0, HSPIRE), 1.2)
beam((0, 0, HSPIRE), (0, 0, HSPIRE + 8), 0.5)

# --- join everything into ONE mesh "Eiffel" ---
bpy.ops.object.select_all(action='DESELECT')
for o in MK: o.select_set(True)
bpy.context.view_layer.objects.active = MK[0]
bpy.ops.object.join()
tower = bpy.context.active_object; tower.name = "Eiffel"
bpy.ops.object.shade_smooth()

# --- iron material ---
mat = bpy.data.materials.new("Iron"); mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (0.16, 0.11, 0.07, 1.0)
if "Roughness" in bsdf.inputs: bsdf.inputs["Roughness"].default_value = 0.55
if "Metallic" in bsdf.inputs: bsdf.inputs["Metallic"].default_value = 0.85
tower.data.materials.append(mat)

# --- ground + sky ---
bpy.ops.mesh.primitive_plane_add(size=2000, location=(0, 0, 0))
gp = bpy.context.active_object
gm = bpy.data.materials.new("Ground"); gm.use_nodes = True
gm.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (0.22, 0.22, 0.20, 1)
gp.data.materials.append(gm)
world = bpy.data.worlds.new("W"); bpy.context.scene.world = world; world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg: bg.inputs[0].default_value = (0.5, 0.6, 0.75, 1.0); bg.inputs[1].default_value = 1.0
sun = bpy.data.lights.new("Sun", type='SUN'); sun.energy = 4.0
so = bpy.data.objects.new("Sun", sun); bpy.context.collection.objects.link(so)
so.rotation_euler = (math.radians(55), math.radians(15), math.radians(40))

print("BUILT eiffel objects_joined verts=%d" % len(tower.data.vertices))
'''

GPU = r'''
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
        except Exception: continue
    for d in prefs.devices: d.use = (d.type in ('OPTIX', 'CUDA'))
    sc.cycles.device = 'GPU'; sc.cycles.samples = 48
    print('GPU_CFG', prefs.compute_device_type, 'gpu=', ok)
except Exception as e:
    print('GPU_CFG_FAIL', e)
'''

CAM = r'''
import math, mathutils
def add_cam(loc, look, lens=52):
    cd = bpy.data.cameras.new("Cam"); cd.lens = lens
    o = bpy.data.objects.new("Cam", cd); bpy.context.collection.objects.link(o)
    o.location = loc
    d = mathutils.Vector(look) - mathutils.Vector(loc)
    o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = o; return o
sc = bpy.context.scene
sc.render.resolution_x = 720; sc.render.resolution_y = 1280   # portrait suits the tall tower
sc.render.image_settings.file_format = 'PNG'
'''


def run(script, tag, timeout=1200):
    r = bl.run_script(script, factory_startup=True, timeout=timeout)
    for ln in r.stdout.splitlines():
        if ln.startswith(("BUILT", "GPU_CFG", "GPU_CFG_FAIL", "SAVED", "EXPORT", "RENDERED", "TT")):
            print("  " + ln.strip(), flush=True)
    if not r.ok:
        print(f"  !! {tag} failed rc={r.returncode}: " + " ".join(r.stderr.splitlines()[-5:]), flush=True)
    return r


def preview():
    s = BUILD + GPU + CAM + r'''
add_cam((0, -560, 165), (0, 0, 150), 50)
sc.render.filepath = r"%s"
bpy.ops.render.render(write_still=True)
print("RENDERED front")
''' % str(OUT / "eiffel_front.png")
    run(s, "preview")


def full():
    op = OUT.as_posix()
    tt = (OUT / "tt"); tt.mkdir(exist_ok=True)
    s = BUILD + GPU + CAM + r'''
import math
# --- save .blend + export GLB + OBJ ---
bpy.ops.wm.save_as_mainfile(filepath=r"%(op)s/eiffel_tower.blend")
print("SAVED blend")
try:
    bpy.ops.export_scene.gltf(filepath=r"%(op)s/eiffel_tower.glb", export_format='GLB')
    print("EXPORT glb")
except Exception as e: print("EXPORT glb FAIL", e)
try:
    bpy.ops.wm.obj_export(filepath=r"%(op)s/eiffel_tower.obj")
    print("EXPORT obj")
except Exception as e:
    try: bpy.ops.export_scene.obj(filepath=r"%(op)s/eiffel_tower.obj"); print("EXPORT obj(legacy)")
    except Exception as e2: print("EXPORT obj FAIL", e2)

def shoot(cam_loc, path, samples=64):
    sc.cycles.samples = samples
    add_cam(cam_loc, (0, 0, 150), 50)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)

# --- 3 stills ---
shoot((0, -560, 165), r"%(op)s/eiffel_front.png")
shoot((360, -400, 210), r"%(op)s/eiffel_34.png")
shoot((150, -230, 55), r"%(op)s/eiffel_base.png")   # low, arch/base detail
print("RENDERED stills")

# --- turntable (GPU orbit) ---
sc.render.resolution_x = 720; sc.render.resolution_y = 1280
sc.cycles.samples = 28
cam = add_cam((0, -560, 165), (0, 0, 150), 50)
N = 48; R = 560
import mathutils
for f in range(N):
    a = 2 * math.pi * f / N
    cam.location = (R * math.sin(a), -R * math.cos(a), 165)
    d = mathutils.Vector((0, 0, 150)) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = r"%(tt)s/tt_%%04d" %% f
    bpy.ops.render.render(write_still=True)
    print("TT %%d" %% f)
print("TT_DONE")
''' % {"op": op, "tt": tt.as_posix()}
    run(s, "full", timeout=3000)
    # assemble turntable -> mp4
    frames = sorted(tt.glob("tt_*.png"))
    if frames:
        import imageio.v2 as imageio
        w = imageio.get_writer(str(OUT / "eiffel_turntable.mp4"), fps=24, codec="libx264",
                               quality=8, macro_block_size=1)
        for fp in frames:
            w.append_data(imageio.imread(str(fp)))
        w.close()
        print(f"  turntable -> eiffel_turntable.mp4 ({len(frames)}f)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if cmd == "preview":
        preview()
    elif cmd == "full":
        full()
