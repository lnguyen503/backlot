"""Blender-native nature scene: a butterfly flying through a valley of grass and flowers.

Realism pass v3: CYCLES GPU path-tracing, golden-hour light + volumetric haze (god rays),
translucent grass with tip-gradient color + two green variants, a monarch-patterned butterfly
(orange / black border / white spots) with backlit translucency, depth of field, and motion
blur. Blender does everything; ComfyUI stylization is an optional later pass.

    .venv\\Scripts\\python.exe tests\\make_butterfly_valley.py --still     # 1-frame look check
    .venv\\Scripts\\python.exe tests\\make_butterfly_valley.py --frames 72  # full animation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.blender import runner as bl, scenes  # noqa: E402


SCRIPT = scenes.PRELUDE + r'''
import math as M, random
a = _argv()
out_dir = a[0]
frames = int(a[1]) if len(a) > 1 else 72
wing_tex = a[2]
still = (len(a) > 3 and a[3] == "still")
still_frame = int(a[4]) if len(a) > 4 else max(1, frames // 2)
RES_X, RES_Y, SAMPLES = 1920, 1080, 128

reset_scene()
sc = bpy.context.scene
random.seed(7)

# ---------- render engine: Cycles GPU (OptiX), fall back to EEVEE ----------
def enable_cycles_gpu():
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
    except Exception as e:
        print("NO_CYCLES_ADDON", e); return None
    for ctype in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI'):
        try:
            prefs.compute_device_type = ctype
            prefs.get_devices()
            if [d for d in prefs.devices if d.type == ctype]:
                for d in prefs.devices:
                    d.use = (d.type == ctype)
                return ctype
        except Exception as e:
            print("GPU_TRY_FAIL", ctype, e)
    return None

eng = "BLENDER_EEVEE"
gpu = enable_cycles_gpu()
if gpu:
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'GPU'; sc.cycles.samples = SAMPLES
    try:
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = 'OPTIX' if gpu == 'OPTIX' else 'OPENIMAGEDENOISE'
    except Exception as e:
        print("DENOISE_SKIP", e)
    eng = "CYCLES/" + gpu
else:
    eng = set_engine()
print("RENDER_ENGINE", eng)

# ---------- helpers ----------
def pbsdf(name, color, rough=0.7, sss=0.0):
    mt = bpy.data.materials.new(name); mt.use_nodes = True
    b = mt.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    if sss:
        try:
            b.inputs["Subsurface Weight"].default_value = sss
            b.inputs["Subsurface Radius"].default_value = (color[0]+0.2, color[1]+0.2, color[2])
        except Exception: pass
    return mt

def mix_rgba(nt, fac, a, b):
    mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'
    facin = [i for i in mx.inputs if i.type == 'VALUE' and i.enabled][0]
    cols = [i for i in mx.inputs if i.type == 'RGBA']
    if isinstance(fac, float): facin.default_value = fac
    else: nt.links.new(fac, facin)
    for sock, val in ((cols[0], a), (cols[1], b)):
        if isinstance(val, tuple):
            sock.default_value = val if len(val) == 4 else (*val, 1.0)
        else:
            nt.links.new(val, sock)
    return [o for o in mx.outputs if o.type == 'RGBA'][0]

# ---------- golden-hour sky + warm sun + volumetric haze ----------
world = bpy.data.worlds.new("Sky"); sc.world = world; world.use_nodes = True
wnt = world.node_tree; bg = wnt.nodes.get("Background")
SUN_ELEV, SUN_AZ = 11.0, 50.0
try:
    sky = wnt.nodes.new("ShaderNodeTexSky"); sky.sky_type = 'MULTIPLE_SCATTERING'
    sky.sun_elevation = M.radians(SUN_ELEV); sky.sun_rotation = M.radians(SUN_AZ)
    try: sky.air_density = 1.5; sky.dust_density = 2.5
    except Exception: pass
    wnt.links.new(sky.outputs[0], bg.inputs[0]); bg.inputs[1].default_value = 0.5
except Exception as e:
    bg.inputs[0].default_value = (0.85, 0.6, 0.4, 1.0); bg.inputs[1].default_value = 0.5
    print("SKY_FALLBACK", e)
# (a world-level volume would extinguish the sun/sky -> black frame; atmospheric haze
#  via a bounded volume domain is a future polish)

sun = add_sun(2.8, (M.radians(90 - SUN_ELEV), 0.0, M.radians(SUN_AZ + 90)))
sun.data.color = (1.0, 0.82, 0.55)
try: sun.data.angle = M.radians(2.0)
except Exception: pass

# ---------- terrain ----------
bpy.ops.mesh.primitive_grid_add(x_subdivisions=200, y_subdivisions=200, size=80, location=(0, 0, 0))
terrain = bpy.context.active_object
tex = bpy.data.textures.new("terr", type='CLOUDS')
try: tex.noise_scale = 4.5; tex.noise_depth = 2
except Exception: pass
m = terrain.modifiers.new("disp", "DISPLACE"); m.texture = tex
m.texture_coords = 'GLOBAL'; m.strength = 2.6; m.mid_level = 0.45
bpy.ops.object.shade_smooth()
terrain.data.materials.append(pbsdf("ground", (0.05, 0.11, 0.025), 0.95))

# ---------- grass blade with tip->base color gradient + translucency ----------
def grass_mat(name, base, tip):
    mt = bpy.data.materials.new(name); mt.use_nodes = True
    nt = mt.node_tree; b = nt.nodes.get("Principled BSDF"); b.inputs["Roughness"].default_value = 0.5
    try:
        b.inputs["Subsurface Weight"].default_value = 0.18
        b.inputs["Subsurface Radius"].default_value = (0.10, 0.40, 0.05)
    except Exception: pass
    try:
        tcn = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ')
        rmp = nt.nodes.new('ShaderNodeValToRGB')
        rmp.color_ramp.elements[0].color = (*base, 1.0); rmp.color_ramp.elements[1].color = (*tip, 1.0)
        nt.links.new(tcn.outputs['Generated'], sep.inputs[0])
        nt.links.new(sep.outputs['Z'], rmp.inputs['Fac'])
        nt.links.new(rmp.outputs['Color'], b.inputs['Base Color'])
    except Exception as e:
        print("GRASS_GRAD_SKIP", e); b.inputs["Base Color"].default_value = (*base, 1.0)
    return mt

bpy.ops.mesh.primitive_cone_add(vertices=5, radius1=0.028, radius2=0.0, depth=0.7, location=(0, 0, 500))
grass = bpy.context.active_object
grass.data.materials.append(grass_mat("grass_warm", (0.06, 0.20, 0.02), (0.34, 0.55, 0.10)))
bpy.ops.mesh.primitive_cone_add(vertices=5, radius1=0.026, radius2=0.0, depth=0.62, location=(2, 0, 500))
grass2 = bpy.context.active_object
grass2.data.materials.append(grass_mat("grass_cool", (0.04, 0.16, 0.04), (0.18, 0.45, 0.14)))

# ---------- flowers (varied colors, with centers) ----------
def flower(color, z):
    bpy.ops.mesh.primitive_circle_add(vertices=10, radius=0.14, fill_type='NGON', location=(0, 0, z))
    f = bpy.context.active_object; f.data.materials.append(pbsdf("flower_%d" % int(z), color, 0.45))
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.045, location=(0, 0, z))
    c = bpy.context.active_object; bpy.ops.object.shade_smooth()
    c.data.materials.append(pbsdf("ctr_%d" % int(z), (0.5, 0.35, 0.05), 0.6)); c.parent = f
    return f

fl_y = flower((0.97, 0.84, 0.12), 520)
fl_p = flower((0.90, 0.18, 0.45), 525)
fl_w = flower((0.92, 0.92, 0.98), 530)
fl_b = flower((0.45, 0.35, 0.95), 535)

# ---------- Geometry-Nodes scatter (random scale + spin) ----------
def scatter(inst, density, smin, smax, name, seed):
    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = ng.nodes, ng.links
    gin = nodes.new('NodeGroupInput'); gout = nodes.new('NodeGroupOutput')
    dist = nodes.new('GeometryNodeDistributePointsOnFaces'); dist.distribute_method = 'RANDOM'
    dist.inputs['Density'].default_value = density
    try: dist.inputs['Seed'].default_value = seed
    except Exception: pass
    oi = nodes.new('GeometryNodeObjectInfo'); oi.transform_space = 'ORIGINAL'
    oi.inputs['Object'].default_value = inst
    try: oi.inputs['As Instance'].default_value = True
    except Exception: pass
    iop = nodes.new('GeometryNodeInstanceOnPoints')
    try:
        rv = nodes.new('FunctionNodeRandomValue'); rv.data_type = 'FLOAT'
        for s in rv.inputs:
            if s.enabled and s.type == 'VALUE' and s.name == 'Min': s.default_value = smin
            if s.enabled and s.type == 'VALUE' and s.name == 'Max': s.default_value = smax
            if s.enabled and s.type == 'INT' and s.name == 'Seed': s.default_value = seed + 5
        links.new(next(o for o in rv.outputs if o.enabled), iop.inputs['Scale'])
    except Exception as e:
        print("SCALE_RND_SKIP", e); iop.inputs['Scale'].default_value = (smax, smax, smax)
    join = nodes.new('GeometryNodeJoinGeometry')
    links.new(gin.outputs[0], dist.inputs[0])
    links.new(dist.outputs['Points'], iop.inputs['Points'])
    links.new(oi.outputs['Geometry'], iop.inputs['Instance'])
    try: links.new(dist.outputs['Rotation'], iop.inputs['Rotation'])
    except Exception as e: print("ROT_SKIP", e)
    last = iop.outputs['Instances']
    try:
        ri = nodes.new('GeometryNodeRotateInstances')
        rvv = nodes.new('FunctionNodeRandomValue'); rvv.data_type = 'FLOAT_VECTOR'
        for s in rvv.inputs:
            if s.enabled and s.type == 'VECTOR' and s.name == 'Max': s.default_value = (0, 0, 6.283)
            if s.enabled and s.type == 'INT' and s.name == 'Seed': s.default_value = seed + 9
        links.new(last, ri.inputs['Instances'])
        links.new(next(o for o in rvv.outputs if o.enabled), ri.inputs['Rotation'])
        last = ri.outputs['Instances']
    except Exception as e:
        print("SPIN_SKIP", e)
    links.new(gin.outputs[0], join.inputs[0])
    links.new(last, join.inputs[0])
    links.new(join.outputs[0], gout.inputs[0])
    terrain.modifiers.new(name, 'NODES').node_group = ng

scatter(grass, 22.0, 0.7, 1.4, "grass1", 1)
scatter(grass2, 12.0, 0.7, 1.3, "grass2", 7)
scatter(fl_y, 0.32, 0.8, 1.3, "fl_y", 2)
scatter(fl_p, 0.28, 0.8, 1.3, "fl_p", 3)
scatter(fl_w, 0.26, 0.8, 1.3, "fl_w", 4)
scatter(fl_b, 0.20, 0.8, 1.3, "fl_b", 5)

# ---------- butterfly: body + antennae + monarch wings ----------
bfly = bpy.data.objects.new("bfly", None); bpy.context.collection.objects.link(bfly)
bpy.ops.mesh.primitive_uv_sphere_add(segments=18, ring_count=10, radius=0.05, location=(0, 0.02, 0))
body = bpy.context.active_object; body.scale = (1.0, 3.0, 1.0); body.parent = bfly
bpy.ops.object.shade_smooth(); body.data.materials.append(pbsdf("bbody", (0.04, 0.025, 0.02), 0.45))
bpy.ops.mesh.primitive_uv_sphere_add(segments=14, ring_count=8, radius=0.045, location=(0, 0.18, 0))
head = bpy.context.active_object; head.parent = bfly
bpy.ops.object.shade_smooth(); head.data.materials.append(pbsdf("bhead", (0.03, 0.02, 0.015), 0.45))
for sx in (-1, 1):
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.006, depth=0.26, location=(sx*0.025, 0.32, 0.07))
    an = bpy.context.active_object; an.rotation_euler = (M.radians(60), 0, 0); an.parent = bfly
    an.data.materials.append(pbsdf("ant", (0.02, 0.02, 0.02), 0.5))

# real monarch wing: ComfyUI/SDXL cutout (RGBA) mapped onto alpha planes
wimg = bpy.data.images.load(wing_tex)
try: wimg.colorspace_settings.name = 'sRGB'
except Exception as e: print("WING_CS_SKIP", e)
WSPAN = 0.66                              # body centre -> right wingtip (world units)
_iw, _ih = wimg.size
WHALF_H = WSPAN * (_ih / max(1, _iw)) / 2.0   # preserve the image aspect ratio

def wing_mat():
    mt = bpy.data.materials.new("wing"); mt.use_nodes = True
    nt = mt.node_tree; n = nt.nodes; l = nt.links
    b = n.get("Principled BSDF"); b.inputs["Roughness"].default_value = 0.42
    out = n.get('Material Output')
    tex = n.new('ShaderNodeTexImage'); tex.image = wimg
    l.new(tex.outputs['Color'], b.inputs['Base Color'])
    surf = b.outputs[0]
    try:  # backlit translucency tinted by the wing colour (sun through wing)
        trans = n.new('ShaderNodeBsdfTranslucent')
        l.new(tex.outputs['Color'], trans.inputs['Color'])
        ms = n.new('ShaderNodeMixShader'); ms.inputs[0].default_value = 0.20
        l.new(surf, ms.inputs[1]); l.new(trans.outputs[0], ms.inputs[2])
        surf = ms.outputs[0]
    except Exception as e:
        print("WING_TRANS_SKIP", e)
    try:  # honour the cutout alpha (transparent outside the wing silhouette)
        tb = n.new('ShaderNodeBsdfTransparent'); ms2 = n.new('ShaderNodeMixShader')
        l.new(tex.outputs['Alpha'], ms2.inputs[0])
        l.new(tb.outputs[0], ms2.inputs[1]); l.new(surf, ms2.inputs[2])
        surf = ms2.outputs[0]
    except Exception as e:
        print("WING_ALPHA_SKIP", e)
    l.new(surf, out.inputs['Surface'])
    return mt

wmat = wing_mat()

def make_wing_plane(parent):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
    w = bpy.context.active_object
    w.scale = (WSPAN / 2.0, WHALF_H, 1.0)     # span [0..WSPAN] in x after offset, image aspect in y
    w.location = (WSPAN / 2.0, 0.0, 0.0)      # inner edge (u=0) at hinge x=0, tip (u=1) outward
    w.data.materials.append(wmat); w.parent = parent
    return w

def side(sx):
    h = bpy.data.objects.new("hinge_%d" % sx, None); bpy.context.collection.objects.link(h)
    h.parent = bfly; h.location = (sx * 0.02, 0, 0)
    if sx < 0: h.scale = (-1, 1, 1)           # mirror the right-wing texture -> symmetric left wing
    make_wing_plane(h)
    return h
    return h

hR = side(1); hL = side(-1)

# ---------- tracking camera + DOF ----------
cam = add_camera((0, -22, 4), (0, 0, 2), 55)
tc2 = cam.constraints.new('TRACK_TO'); tc2.target = bfly
tc2.track_axis = 'TRACK_NEGATIVE_Z'; tc2.up_axis = 'UP_Y'
try:
    cam.data.dof.use_dof = True; cam.data.dof.focus_object = bfly; cam.data.dof.aperture_fstop = 2.2
except Exception as e:
    print("DOF_SKIP", e)

# ---------- animate ----------
sc.frame_start = 1; sc.frame_end = frames
def flight(t):
    return (M.sin(t * M.pi * 2.0) * 4.5, -15 + t * 30.0, 2.5 + M.sin(t * M.pi * 4.0) * 0.6)
for f in range(1, frames + 1):
    sc.frame_set(f); t = (f - 1) / max(1, frames - 1)
    bx, by, bz = flight(t)
    nx, ny, _ = flight(min(1.0, t + 1.0 / frames))
    yaw = -M.atan2(nx - bx, max(1e-4, ny - by))
    bfly.location = (bx, by, bz)
    bfly.rotation_euler = (0, M.sin(t * M.pi * 2.0) * 0.18, yaw)
    bfly.keyframe_insert("location", frame=f); bfly.keyframe_insert("rotation_euler", frame=f)
    # keep wings in a raised dihedral V the whole cycle (14deg..46deg) so flat planes
    # never go edge-on to the camera and the dorsal pattern always reads
    dihedral = M.radians(30) + M.radians(16) * M.sin(t * 2 * M.pi * (frames / 16.0))
    hR.rotation_euler = (0, -dihedral, 0); hL.rotation_euler = (0, dihedral, 0)
    hR.keyframe_insert("rotation_euler", frame=f); hL.keyframe_insert("rotation_euler", frame=f)
    cam.location = (bx * 0.45, by - 5.5, bz + 1.2); cam.keyframe_insert("location", frame=f)

# ---------- render ----------
sc.render.resolution_x = RES_X; sc.render.resolution_y = RES_Y
sc.render.image_settings.file_format = 'PNG'
sc.view_settings.view_transform = 'AgX'
try: sc.view_settings.exposure = -0.4
except Exception: pass
try:
    sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.12
except Exception as e: print("MBLUR_SKIP", e)
try:
    bpy.ops.wm.save_as_mainfile(filepath=out_dir.rstrip('/\\') + '/butterfly_valley.blend')
    print("BLEND_SAVED")
except Exception as e:
    print("BLEND_SAVE_SKIP", e)
if still:
    sc.frame_set(still_frame)
    hR.rotation_euler = (0, M.radians(-22), 0); hL.rotation_euler = (0, M.radians(22), 0)
    sc.render.filepath = out_dir.rstrip('/\\') + '/still.png'
    bpy.ops.render.render(write_still=True); print("STILL_DONE")
else:
    sc.render.filepath = out_dir.rstrip('/\\') + '/fly_'
    bpy.ops.render.render(animation=True); print("ANIM_DONE", sc.frame_end)
print("ENGINE_USED", eng)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--interp-fps", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/butterfly_valley.mp4"))
    args = ap.parse_args()

    out = Path(args.out)
    fdir = out.parent / "butterfly_frames"
    fdir.mkdir(parents=True, exist_ok=True)

    wing_tex = str((out.parent / "wing_assets" / "wing_R.png").resolve())
    sargs = [str(fdir.resolve()), args.frames, wing_tex]
    if args.still:
        sargs += ["still", args.frames // 2]
    print(f"[blender] rendering {'still' if args.still else str(args.frames)+' frames'} (Cycles) ...", flush=True)
    r = bl.run_script(SCRIPT, args=sargs, timeout=10800)
    print(r.stdout[-1500:])
    if not r.ok:
        print("BLENDER FAILED:\n" + r.stderr[-2500:]); return
    if args.still:
        print(f"STILL -> {fdir/'still.png'}"); return

    pngs = sorted(fdir.glob("fly_*.png"))
    print(f"[encode] {len(pngs)} frames -> {out}", flush=True)
    raw = str(out.with_suffix(".raw.mp4")) if args.interp_fps else str(out)
    w = imageio.get_writer(raw, fps=args.fps, codec="libx264", quality=9, macro_block_size=1)
    for p in pngs:
        w.append_data(imageio.imread(p)[:, :, :3])
    w.close()
    if args.interp_fps:
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-i", raw, "-vf",
                        f"minterpolate=fps={args.interp_fps}:mi_mode=mci:mc_mode=aobmc:"
                        "me_mode=bidir:vsbmc=1", "-c:v", "libx264", "-crf", "16",
                        "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True)
        Path(raw).unlink(missing_ok=True)
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
