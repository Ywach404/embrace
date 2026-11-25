"""Procedural leather glove hands: mesh, rig, textures, poses, USDZ.

Run headless, one stage per invocation, each stage reading the previous
stage's .blend:

    Blender --background --factory-startup --python build_gloves.py -- model OUT_DIR
    Blender --background --factory-startup --python build_gloves.py -- bake OUT_DIR
    Blender --background --factory-startup --python build_gloves.py -- export OUT_DIR
    Blender --background --factory-startup --python build_gloves.py -- preview OUT_DIR

Working frame inside Blender (Z up): wrist joint at the origin, fingers
along +Y, palm facing -Z, right thumb on -X. The export stage converts to
the shipped frame (Y up, fingers along -Z, palm facing -Y).
"""

import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
STAGE = ARGS[0] if ARGS else "model"
OUT = os.path.abspath(ARGS[1]) if len(ARGS) > 1 else os.getcwd()
# Intermediate .blend files and reports stay out of the output folder.
WORK = os.environ.get("GLOVE_WORK", os.path.join("/tmp", "glove_build"))
os.makedirs(WORK, exist_ok=True)

FINGERS = ["index", "middle", "ring", "little"]
FINGER_JOINTS = ["Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]
THUMB_JOINTS = ["thumbKnuckle", "thumbIntermediateBase", "thumbIntermediateTip", "thumbTip"]


# ---------------------------------------------------------------------------
# Hand proportions (metres, right hand, Blender working frame)
# ---------------------------------------------------------------------------

# Column boundaries across the knuckle line, thumb side first.
B = [-0.0440, -0.0218, -0.0005, 0.0193, 0.0378]
C = [(B[i] + B[i + 1]) * 0.5 for i in range(4)]

FINGER = {
    #          mcp y   mcp z    seg lengths              half width  spread  rest curl (mcp, pip, dip) deg
    "index":  (0.0905, -0.0010, (0.0410, 0.0245, 0.0200), 0.0108, -0.070, (8, 12, 6)),
    "middle": (0.0935, 0.0000, (0.0445, 0.0275, 0.0210), 0.0110, -0.010, (9, 13, 7)),
    "ring":   (0.0895, -0.0020, (0.0415, 0.0260, 0.0200), 0.0104, 0.065, (11, 15, 8)),
    "little": (0.0810, -0.0060, (0.0325, 0.0195, 0.0180), 0.0093, 0.160, (14, 18, 10)),
}
METACARPAL_BASE = {"index": (-0.017, 0.012), "middle": (-0.005, 0.012),
                   "ring": (0.007, 0.012), "little": (0.017, 0.010)}

# Thumb chain: CMC (thumbKnuckle), MCP, IP, tip.
THUMB_PTS = [Vector((-0.0200, 0.0200, -0.0080)),
             Vector((-0.0455, 0.0500, -0.0160)),
             Vector((-0.0580, 0.0780, -0.0195)),
             Vector((-0.0635, 0.1005, -0.0210))]
THUMB_NAIL = Vector((-0.62, -0.05, 0.78)).normalized()
THUMB_HALF_WIDTH = 0.0124


def finger_chain(name):
    """Joint positions and frames for one finger in the rest pose."""
    y, z, segs, hw, spread, curl = FINGER[name]
    col = FINGERS.index(name)
    p = Vector((C[col], y, z))
    d = Vector((math.sin(spread), math.cos(spread), 0.0)).normalized()
    side = d.cross(Vector((0, 0, 1))).normalized()
    pts, dirs = [p.copy()], []
    for seg, ang in zip(segs, curl):
        d = Quaternion(side, -math.radians(ang)) @ d
        dirs.append(d.copy())
        p = p + d * seg
        pts.append(p.copy())
    return pts, dirs, side


def frame_from(t, up_hint):
    t = t.normalized()
    u = (up_hint - t * up_hint.dot(t)).normalized()
    s = t.cross(u).normalized()
    return t, s, u


# ---------------------------------------------------------------------------
# Stage: model
# ---------------------------------------------------------------------------

RING_ANGLES = [135, 90, 45, 0, -45, -90, -135, 180]  # TL TM TR MR BR BM BL ML


def palm_ring(y, W, T, zc, cx=0.0):
    """20 vertices: top row thumb side to little side, R mid, bottom row back, L mid."""
    xd = [B[0], C[0], B[1], C[1], B[2], C[2], B[3], C[3], B[4]]
    wd = (B[4] - B[0]) * 0.5
    cd = (B[4] + B[0]) * 0.5
    xs = [cx + (x - cd) * (W / wd) for x in xd]
    ring = []
    for x in xs:
        xn = (x - cx) / W
        ring.append(Vector((x, y, zc + T * (1.0 - 0.32 * xn * xn))))
    ring.append(Vector((cx + W * 1.07, y, zc - 0.002)))
    for x in reversed(xs):
        xn = (x - cx) / W
        ring.append(Vector((x, y, zc - T * (1.0 - 0.22 * xn * xn))))
    ring.append(Vector((cx - W * 1.07, y, zc - 0.002)))
    return ring


def distal_ring():
    """Knuckle line ring, shaped per finger column, plus the three web vertices."""
    ys, zs = [], []
    for name in FINGERS:
        ys.append(FINGER[name][0])
        zs.append(FINGER[name][1])
    # per column position k = 0..8 (boundaries even, centres odd)
    ky, kz = [], []
    for k in range(9):
        if k % 2 == 1:
            ky.append(ys[k // 2] - 0.002)
            kz.append(zs[k // 2])
        else:
            a = max(0, k // 2 - 1)
            b = min(3, k // 2)
            ky.append((ys[a] + ys[b]) * 0.5 - (0.004 if k in (0, 8) else 0.0))
            kz.append((zs[a] + zs[b]) * 0.5)
    xd = [B[0], C[0], B[1], C[1], B[2], C[2], B[3], C[3], B[4]]
    ring = []
    for k in range(9):
        bump = 0.0022 if k % 2 == 1 else 0.0
        ring.append(Vector((xd[k], ky[k], kz[k] + 0.0118 + bump)))
    ring.append(Vector((B[4] + 0.0035, ky[8] - 0.002, kz[8] - 0.001)))
    for k in reversed(range(9)):
        ring.append(Vector((xd[k], ky[k] + 0.003, kz[k] - 0.0128)))
    ring.append(Vector((B[0] - 0.0035, ky[0], kz[0] - 0.001)))
    webs = {}
    for k in (2, 4, 6):
        webs[k] = Vector((xd[k], ky[k] + 0.012, kz[k] - 0.003))
    return ring, webs


def tube_stations(pts, dirs, up_hint_fn, hw, fracs_per_seg, tip_scale):
    """Ring stations along a joint chain: (centre, s, u, rw, rh)."""
    st = []
    nseg = len(dirs)
    for si in range(nseg):
        for f in fracs_per_seg[si]:
            c = pts[si].lerp(pts[si + 1], f)
            if f >= 0.999 and si + 1 < nseg:
                t = (dirs[si] + dirs[si + 1]).normalized()
            else:
                t = dirs[si]
            t, s, u = frame_from(t, up_hint_fn(si, f))
            taper = 1.0 - 0.16 * ((si + f) / nseg)
            rw = hw * taper
            if f >= 0.999 and si + 1 < nseg:
                rw *= 1.06  # joint bulge
            st.append((c, s, u, rw, rw * 0.90))
    # rounded tip
    c_end, d_end = pts[-1], dirs[-1]
    t, s, u = frame_from(d_end, up_hint_fn(nseg - 1, 1.0))
    st[-1] = (st[-1][0], st[-1][1], st[-1][2], st[-1][3] * tip_scale[0], st[-1][4] * tip_scale[0])
    return st, (c_end + d_end * 0.0015)


def build_right_mesh():
    bm = bmesh.new()
    seam_pairs = []

    def V(p):
        return bm.verts.new(p)

    def quad(a, b, c, d):
        return bm.faces.new((a, b, c, d))

    # Ordered rings from the cuff hem inward over the hand.
    cuff_specs = [  # y, W, T, zc
        (-0.0880, 0.0392, 0.0302, 0.0005),
        (-0.0700, 0.0368, 0.0280, 0.0005),
        (-0.0520, 0.0352, 0.0265, 0.0005),
        (-0.0360, 0.0342, 0.0255, 0.0005),
        (-0.0300, 0.0336, 0.0248, 0.0005),   # cuff rim
        (-0.0272, 0.0296, 0.0206, 0.0000),   # step down to the wrist
        (-0.0160, 0.0290, 0.0196, 0.0000),
    ]
    palm_specs = [
        (-0.0030, 0.0300, 0.0192, 0.0000),
        (0.0160, 0.0345, 0.0180, -0.0010),
        (0.0400, 0.0392, 0.0168, -0.0015),
        (0.0630, 0.0414, 0.0158, -0.0015),
        (0.0720, 0.0418, 0.0152, -0.0012),
    ]
    inner_specs = [  # back inside the cuff, closing the shell
        (-0.0860, 0.0362, 0.0272, 0.0005),
        (-0.0700, 0.0340, 0.0250, 0.0005),
    ]
    rings_pos = []
    for y, W, T, zc in reversed(inner_specs):
        rings_pos.append(palm_ring(y, W, T, zc))
    hem_index = len(rings_pos)
    for y, W, T, zc in cuff_specs:
        rings_pos.append(palm_ring(y, W, T, zc))
    palm_start = len(rings_pos)
    for y, W, T, zc in palm_specs:
        rings_pos.append(palm_ring(y, W, T, zc, cx=-0.0015))
    dr, webs = distal_ring()
    rings_pos.append(dr)
    distal_index = len(rings_pos) - 1

    # thumb port on the L side of palm rings palm_start+1 .. palm_start+3
    port_lo, port_hi = palm_start + 1, palm_start + 3
    skip_vert = (palm_start + 2, 19)

    rings = []
    for ri, ring in enumerate(rings_pos):
        rv = []
        for vi, p in enumerate(ring):
            if (ri, vi) == skip_vert:
                rv.append(None)
            else:
                rv.append(V(p))
        rings.append(rv)

    for ri in range(len(rings) - 1):
        a, b = rings[ri], rings[ri + 1]
        for i in range(20):
            j = (i + 1) % 20
            if port_lo <= ri < port_hi and i in (18, 19):
                continue
            quad(a[i], a[j], b[j], b[i])
        # seams along the two sides, and the hem ring
        for i in (9, 19):
            if a[i] is not None and b[i] is not None and not (port_lo <= ri < port_hi and i == 19):
                seam_pairs.append((a[i], b[i]))
    for i in range(20):
        seam_pairs.append((rings[hem_index][i], rings[hem_index][(i + 1) % 20]))

    # inner cap closing the cuff
    def cap20(ring, inward):
        mids = {0: ring[19], 8: ring[9]}
        for k in range(1, 8):
            mids[k] = V(ring[k].co.lerp(ring[18 - k].co, 0.5) + inward)
        for k in range(8):
            quad(ring[k], ring[k + 1], mids[k + 1], mids[k])
            quad(mids[k], mids[k + 1], ring[18 - (k + 1)], ring[18 - k])
    cap20(rings[0], Vector((0, 0.004, 0)))

    # fingers
    distal = rings[distal_index]
    web_v = {k: V(p) for k, p in webs.items()}
    mid = {0: distal[19], 8: distal[9], 2: web_v[2], 4: web_v[4], 6: web_v[6]}
    finger_rings = {}

    def tube(ring0, stations, tip_point, name):
        prev = ring0
        all_rings = [ring0]
        for (c, s, u, rw, rh) in stations:
            cur = []
            for ang in RING_ANGLES:
                a = math.radians(ang)
                k = 1.0 / 0.93
                cur.append(V(c + s * (math.cos(a) * rw * k) + u * (math.sin(a) * rh * k)))
            for i in range(8):
                j = (i + 1) % 8
                quad(prev[i], prev[j], cur[j], cur[i])
            for i in (3, 7):
                seam_pairs.append((prev[i], cur[i]))
            prev = cur
            all_rings.append(cur)
        cv = V(tip_point)
        TL, TM, TR, MR, BR, BM, BL, ML = prev
        quad(TL, TM, cv, ML)
        quad(TM, TR, MR, cv)
        quad(cv, MR, BR, BM)
        quad(ML, cv, BM, BL)
        seam_pairs.append((MR, cv))
        seam_pairs.append((cv, ML))
        finger_rings[name] = all_rings

    for fi, name in enumerate(FINGERS):
        k0 = 2 * fi
        ring0 = [distal[k0], distal[k0 + 1], distal[k0 + 2], mid[k0 + 2],
                 distal[18 - (k0 + 2)], distal[18 - (k0 + 1)], distal[18 - k0], mid[k0]]
        pts, dirs, side = finger_chain(name)
        hw = FINGER[name][3]

        def up_fn(si, f, dirs=dirs, side=side):
            return side.cross(dirs[si])
        fr = [(0.30, 0.62, 0.86, 1.0), (0.22, 0.55, 0.84, 1.0), (0.20, 0.52, 0.78, 0.93)]
        st, tip = tube_stations(pts, dirs, up_fn, hw, fr, (0.80,))
        # second last a touch smaller, last much smaller for a round tip
        c, s, u, rw, rh = st[-2]
        st[-2] = (c, s, u, rw * 0.97, rh * 0.97)
        tube(ring0, st, tip, name)

    # thumb
    r1, r2, r3 = rings[port_lo], rings[port_lo + 1], rings[port_hi]
    ring0 = [r1[0], r2[0], r3[0], r3[19], r3[18], r2[18], r1[18], r1[19]]
    tdirs = [(THUMB_PTS[i + 1] - THUMB_PTS[i]).normalized() for i in range(3)]

    def thumb_up(si, f):
        return THUMB_NAIL
    fr = [(0.55, 0.80, 1.0), (0.25, 0.55, 0.85, 1.0), (0.20, 0.52, 0.78, 0.93)]
    st, tip = tube_stations(THUMB_PTS, tdirs, thumb_up, THUMB_HALF_WIDTH, fr, (0.80,))
    c, s, u, rw, rh = st[0]
    st[0] = (c, s, u, rw * 1.25, rh * 1.15)  # thenar root
    c, s, u, rw, rh = st[1]
    st[1] = (c, s, u, rw * 1.08, rh * 1.05)
    tube(ring0, st, tip, "thumb")

    bm.verts.ensure_lookup_table()
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    seam_set = set()
    for a, b in seam_pairs:
        e = bm.edges.get((a, b))
        if e:
            e.seam = True
            seam_set.add(e.index)

    me = bpy.data.meshes.new("glove_right")
    bm.to_mesh(me)
    bm.free()
    return me


def build_armature(name, mirror):
    sx = -1.0 if mirror else 1.0

    def M(v):
        return Vector((v.x * sx, v.y, v.z))

    arm = bpy.data.armatures.new(name)
    ob = bpy.data.objects.new(name, arm)
    bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones
    up = Vector((0, 0, 1))

    def bone(bname, head, tail, parent, roll_up, deform=True):
        b = eb.new(bname)
        b.head, b.tail = M(head), M(tail)
        b.align_roll(M(roll_up))
        if parent:
            b.parent = eb[parent]
        b.use_deform = deform
        b.use_connect = False
        return b

    bone("wrist", Vector((0, 0, 0)), Vector((0, 0.035, 0)), None, up)
    bone("forearmWrist", Vector((0, 0, 0)), Vector((0, -0.075, 0)), "wrist", up)
    for name in FINGERS:
        pts, dirs, side = finger_chain(name)
        bx, by = METACARPAL_BASE[name]
        base = Vector((bx, by, pts[0].z * 0.5))
        bone(f"{name}FingerMetacarpal", base, pts[0], "wrist", up)
        parent = f"{name}FingerMetacarpal"
        for i, j in enumerate(FINGER_JOINTS[1:4]):
            u = dirs[i].cross(side) * -1.0
            bone(f"{name}Finger{j}", pts[i], pts[i + 1], parent, u)
            parent = f"{name}Finger{j}"
        bone(f"{name}FingerTip", pts[3], pts[3] + dirs[2] * 0.012, parent,
             dirs[2].cross(side) * -1.0, deform=False)
    tp = THUMB_PTS
    parent = "wrist"
    for i, j in enumerate(THUMB_JOINTS[:3]):
        bone(j, tp[i], tp[i + 1], parent, THUMB_NAIL)
        parent = j
    tdir = (tp[3] - tp[2]).normalized()
    bone("thumbTip", tp[3], tp[3] + tdir * 0.012, parent, THUMB_NAIL, deform=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in ob.pose.bones:
        pb.rotation_mode = "QUATERNION"
    ob.show_in_front = True
    return ob


def select_only(*obs):
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[-1]


def stage_model():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.unit_settings.system = "METRIC"
    scn.unit_settings.scale_length = 1.0

    me = build_right_mesh()
    ob = bpy.data.objects.new("glove_right", me)
    scn.collection.objects.link(ob)
    cage_faces = len(me.polygons)

    # one level of Catmull-Clark, applied
    select_only(ob)
    mod = ob.modifiers.new("subd", "SUBSURF")
    mod.levels = 1
    mod.render_levels = 1
    mod.uv_smooth = "PRESERVE_BOUNDARIES"
    bpy.ops.object.modifier_apply(modifier=mod.name)
    seams = sum(1 for e in me.edges if e.use_seam)
    print(f"cage faces {cage_faces}, final faces {len(me.polygons)}, seams kept {seams}")

    # UVs: the seams split the glove into a back piece, a palm piece and the lining
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.02)
    bpy.ops.uv.pack_islands(rotate=True, margin=0.006)
    bpy.ops.object.mode_set(mode="OBJECT")
    for p in me.polygons:
        p.use_smooth = True

    rig = build_armature("rig_right", mirror=False)
    rig.data.display_type = "OCTAHEDRAL"

    # automatic weights; heat weighting is unstable at hand scale, so solve at x10
    for o in (ob, rig):
        o.scale = (10, 10, 10)
    bpy.context.view_layer.update()
    select_only(ob, rig)
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    for o in (ob, rig):
        o.scale = (1, 1, 1)
    ob.matrix_parent_inverse = Matrix.Identity(4)
    bpy.context.view_layer.update()

    # Keep at most four influences per vertex, as a GPU skinning path expects.
    select_only(ob)
    bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
    bpy.ops.object.vertex_group_limit_total(group_select_mode="BONE_DEFORM", limit=4)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="BONE_DEFORM", lock_active=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    unweighted = 0
    for v in me.vertices:
        if sum(g.weight for g in v.groups) < 1e-4:
            unweighted += 1
    print("vertex groups", len(ob.vertex_groups), "unweighted verts", unweighted)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(WORK, "stage_model.blend"))


# ---------------------------------------------------------------------------
# Poses: Euler XYZ degrees in each bone's local frame, relative to the rest
# pose (relaxed open). Bone Y runs along the bone, bone Z points out of the
# back of the hand (the nail side), so negative X curls toward the palm and
# positive Z swings toward the thumb side.
# ---------------------------------------------------------------------------

def fingers(**curls):
    out = {}
    for name, c in curls.items():
        m, p, d = c[:3]
        out[f"{name}FingerKnuckle"] = (-m, 0, c[3] if len(c) > 3 else 0)
        out[f"{name}FingerIntermediateBase"] = (-p, 0, 0)
        out[f"{name}FingerIntermediateTip"] = (-d, 0, 0)
    return out


def thumb(cmc, mcp, ip, cmc_z=0.0, cmc_y=0.0):
    return {"thumbKnuckle": (-cmc, cmc_y, cmc_z),
            "thumbIntermediateBase": (-mcp, 0, 0),
            "thumbIntermediateTip": (-ip, 0, 0)}


def cup(ring=0.0, little=0.0):
    return {"ringFingerMetacarpal": (-ring, 0, 0), "littleFingerMetacarpal": (-little, 0, 0)}


FIST4 = dict(index=(80, 100, 55), middle=(82, 100, 55), ring=(84, 100, 55), little=(86, 98, 55))

POSES = {
    "open": {**fingers(index=(-8, -12, -6, 5), middle=(-9, -13, -7), ring=(-11, -15, -8, -4),
                       little=(-14, -18, -10, -8)),
             **thumb(-6, -6, -4, cmc_z=8)},
    "relaxed": {**fingers(index=(8, 14, 8), middle=(10, 18, 10), ring=(12, 22, 12), little=(14, 26, 14)),
                **thumb(8, 8, 10), **cup(2, 4)},
    "trigger": {**fingers(index=(28, 52, 24), middle=(66, 88, 48), ring=(68, 90, 48), little=(70, 90, 48)),
                **thumb(28, 18, 16, cmc_z=-6), **cup(4, 8)},
    "grip": {**fingers(index=(10, 16, 8), middle=(74, 96, 52), ring=(76, 96, 52), little=(78, 94, 52)),
             **thumb(22, 14, 14, cmc_z=-4), **cup(4, 8)},
    "fist": {**fingers(**FIST4), **thumb(38, 34, 40, cmc_z=-10), **cup(6, 10)},
    "point": {**fingers(index=(-8, -12, -6), middle=FIST4["middle"], ring=FIST4["ring"],
                        little=FIST4["little"]),
              **thumb(36, 30, 32, cmc_z=-8), **cup(6, 10)},
    "pistol_grip": {**fingers(index=(26, 48, 20), middle=(70, 86, 50), ring=(72, 88, 50),
                              little=(74, 88, 50)),
                    **thumb(30, 8, 4, cmc_z=-12), **cup(4, 8)},
    "rifle_grip": {**fingers(index=(44, 62, 38), middle=(46, 66, 40), ring=(48, 68, 40),
                             little=(50, 70, 40)),
                   **thumb(30, 10, 10, cmc_z=-6), **cup(3, 6)},
    "melee_grip": {**fingers(index=(74, 94, 52), middle=(76, 96, 54), ring=(78, 96, 54),
                             little=(80, 94, 54)),
                   **thumb(42, 30, 44, cmc_z=-12), **cup(5, 9)},
}
POSE_ORDER = ["open", "relaxed", "trigger", "grip", "fist", "point",
              "pistol_grip", "rifle_grip", "melee_grip"]


def pose_quat(euler_deg, mirror):
    x, y, z = euler_deg
    from mathutils import Euler
    q = Euler((math.radians(x), math.radians(y), math.radians(z)), "XYZ").to_quaternion()
    if mirror:
        # Left bones are the mirror image of the right ones with X kept:
        # a mirrored rotation keeps its X component and negates Y and Z.
        q = Quaternion((q.w, q.x, -q.y, -q.z))
    return q


def apply_pose(rig, name, mirror=False):
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    if name:
        for bname, e in POSES[name].items():
            rig.pose.bones[bname].rotation_quaternion = pose_quat(e, mirror)
    bpy.context.view_layer.update()


def deformation_report(ob, rig, label):
    """Compare the evaluated (posed) mesh with the rest mesh."""
    me = ob.data
    rest = [v.co.copy() for v in me.vertices]
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    em = ev.to_mesh()
    pos = [v.co.copy() for v in em.vertices]
    ev.to_mesh_clear()

    def area(pts, poly):
        vs = [pts[i] for i in poly.vertices]
        a = 0.0
        for k in range(1, len(vs) - 1):
            a += (vs[k] - vs[0]).cross(vs[k + 1] - vs[0]).length * 0.5
        return a
    ratios = [area(pos, p) / max(area(rest, p), 1e-12) for p in me.polygons]
    eratio = [(pos[e.vertices[0]] - pos[e.vertices[1]]).length /
              max((rest[e.vertices[0]] - rest[e.vertices[1]]).length, 1e-9) for e in me.edges]
    # A face is flipped if its posed normal opposes its rest normal carried
    # rigidly by the bone that weighs most on its first vertex.
    skin = {}
    for pb in rig.pose.bones:
        skin[pb.name] = (pb.matrix @ pb.bone.matrix_local.inverted()).to_3x3()
    gname = {g.index: g.name for g in ob.vertex_groups}
    flipped = 0
    flipped_dorsal = 0
    for p in me.polygons:
        v0 = me.vertices[p.vertices[0]]
        best = max(v0.groups, key=lambda g: g.weight)
        R = skin[gname[best.group]]
        vs0 = [rest[i] for i in p.vertices]
        vs1 = [pos[i] for i in p.vertices]
        n0 = R @ (vs0[1] - vs0[0]).cross(vs0[2] - vs0[0])
        n1 = (vs1[1] - vs1[0]).cross(vs1[2] - vs1[0])
        if n0.length > 1e-12 and n1.length > 1e-12 and n0.normalized().dot(n1.normalized()) < 0.0:
            flipped += 1
            bz = rig.data.bones[gname[best.group]].matrix_local.to_3x3().col[2]
            if p.normal.dot(bz) > 0.0:
                flipped_dorsal += 1
    line = (f"{label:24s} face area ratio min {min(ratios):.2f} max {max(ratios):.2f}  "
            f"edge ratio min {min(eratio):.2f} max {max(eratio):.2f}  folded {flipped} "
            f"(on the back of the hand {flipped_dorsal})")
    print(line)
    return {"pose": label, "area_min": min(ratios), "area_max": max(ratios),
            "edge_min": min(eratio), "edge_max": max(eratio), "folded_in_crease": flipped - flipped_dorsal,
            "folded_on_back": flipped_dorsal}


def stage_check():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(WORK, "stage_model.blend"))
    ob, rig = bpy.data.objects["glove_right"], bpy.data.objects["rig_right"]
    results = []
    for name in FINGERS:
        apply_pose(rig, None)
        for j, ang in zip(("Knuckle", "IntermediateBase", "IntermediateTip"), (90, 105, 70)):
            rig.pose.bones[f"{name}Finger{j}"].rotation_quaternion = pose_quat((-ang, 0, 0), False)
        bpy.context.view_layer.update()
        results.append(deformation_report(ob, rig, f"{name} full curl"))
    apply_pose(rig, None)
    for j, ang in zip(THUMB_JOINTS[:3], (45, 60, 75)):
        rig.pose.bones[j].rotation_quaternion = pose_quat((-ang, 0, 0), False)
    bpy.context.view_layer.update()
    results.append(deformation_report(ob, rig, "thumb full curl"))
    for name in POSE_ORDER:
        apply_pose(rig, name)
        results.append(deformation_report(ob, rig, f"pose {name}"))
    apply_pose(rig, None)
    with open(os.path.join(WORK, "bend_check.json"), "w") as f:
        json.dump(results, f, indent=1)

# ---------------------------------------------------------------------------
# Stage: bake. Blender bakes the rest-pose surface position and normal into
# UV space; the leather, stitching and wear are then computed per texel in
# numpy so every detail lives in plain image textures.
# ---------------------------------------------------------------------------

TEX = 1024


def emit_bake(ob, name, socket_fn, colorspace="Non-Color"):
    img = bpy.data.images.new(name, TEX, TEX, alpha=True, float_buffer=True)
    img.colorspace_settings.name = colorspace
    img.generated_color = (0, 0, 0, 0)
    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    socket_fn(nt, em.inputs["Color"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.nodes.active = tex
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    scn = bpy.context.scene
    scn.cycles.samples = 1
    scn.render.bake.margin = 0
    scn.render.bake.use_clear = True
    select_only(ob)
    bpy.ops.object.bake(type="EMIT")
    import numpy as np
    arr = np.array(img.pixels[:], dtype=np.float32).reshape(TEX, TEX, 4)
    return arr


def hash3(ix, iy, iz, seed=0):
    import numpy as np
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ \
        (iz.astype(np.int64) * 83492791) ^ (seed * 2654435761)
    h = h & 0xFFFFFFFF
    out = []
    for k in range(3):
        h = (h ^ (h >> 16)) * 0x45D9F3B & 0xFFFFFFFF
        h = (h ^ (h >> 16)) * 0x45D9F3B & 0xFFFFFFFF
        h = h ^ (h >> 16)
        out.append((h & 0xFFFFFF).astype(np.float32) / float(0x1000000))
        h = (h + 0x9E3779B9 * (k + 1)) & 0xFFFFFFFF
    return out


def worley(P, cell, seed=0):
    import numpy as np
    p = P / cell
    i = np.floor(p)
    f = (p - i).astype(np.float32)
    i = i.astype(np.int64)
    F1 = np.full(len(P), 9.0, np.float32)
    F2 = np.full(len(P), 9.0, np.float32)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for oz in (-1, 0, 1):
                hx, hy, hz = hash3(i[:, 0] + ox, i[:, 1] + oy, i[:, 2] + oz, seed)
                dx = ox + hx - f[:, 0]
                dy = oy + hy - f[:, 1]
                dz = oz + hz - f[:, 2]
                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                F2 = np.where(d < F1, F1, np.minimum(F2, d))
                F1 = np.minimum(F1, d)
    return F1, F2


def value_noise(P, cell, seed=0):
    import numpy as np
    p = P / cell
    i = np.floor(p).astype(np.int64)
    f = (p - i).astype(np.float32)
    u = f * f * (3 - 2 * f)
    acc = np.zeros(len(P), np.float32)
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                h = hash3(i[:, 0] + ox, i[:, 1] + oy, i[:, 2] + oz, seed)[0]
                w = (u[:, 0] if ox else 1 - u[:, 0]) * (u[:, 1] if oy else 1 - u[:, 1]) * \
                    (u[:, 2] if oz else 1 - u[:, 2])
                acc += w * h
    return acc


def fbm(P, cell, octaves=4, seed=0):
    total, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        total = total + amp * value_noise(P, cell / (2 ** o), seed + o * 17)
        norm += amp
        amp *= 0.5
    return total / norm


def smoothstep(a, b, x):
    import numpy as np
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def seam_chains(me):
    """Arc length along every seam polyline, per (edge, vertex)."""
    adj = {}
    seam_edges = [e for e in me.edges if e.use_seam]
    for e in seam_edges:
        a, b = e.vertices
        adj.setdefault(a, []).append((b, e.index))
        adj.setdefault(b, []).append((a, e.index))
    co = [v.co for v in me.vertices]
    done = set()
    arc = {}  # edge index -> (va, sa, vb, sb)
    starts = [v for v, n in adj.items() if len(n) != 2] + list(adj.keys())
    for v0 in starts:
        for (v1, ei) in adj[v0]:
            if ei in done:
                continue
            s, prev, cur, edge = 0.0, v0, v1, ei
            while True:
                done.add(edge)
                L = (co[cur] - co[prev]).length
                arc[edge] = (prev, s, cur, s + L)
                s += L
                nxt = [(w, ej) for (w, ej) in adj[cur] if ej not in done]
                if len(adj[cur]) != 2 or not nxt:
                    break
                prev, (cur, edge) = cur, nxt[0]
    return arc


def stage_bake():
    import numpy as np
    bpy.ops.wm.open_mainfile(filepath=os.path.join(WORK, "stage_model.blend"))
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    ob = bpy.data.objects["glove_right"]
    rig = bpy.data.objects["rig_right"]
    me = ob.data

    def pos_socket(nt, sock):
        g = nt.nodes.new("ShaderNodeNewGeometry")
        nt.links.new(g.outputs["Position"], sock)

    def nrm_socket(nt, sock):
        g = nt.nodes.new("ShaderNodeNewGeometry")
        nt.links.new(g.outputs["Normal"], sock)

    def white_socket(nt, sock):
        sock.default_value = (1, 1, 1, 1)

    cov_img = emit_bake(ob, "bake_cover", white_socket)
    pos_img = emit_bake(ob, "bake_position", pos_socket)
    nrm_img = emit_bake(ob, "bake_normal_obj", nrm_socket)
    cover = cov_img[:, :, 0] > 0.5
    print("covered texels", int(cover.sum()), "of", TEX * TEX)

    P = pos_img[cover][:, :3].astype(np.float32)
    N = nrm_img[cover][:, :3].astype(np.float32)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-6)
    n = len(P)

    # texel size in metres along U and V, for slope and density
    full = pos_img[:, :, :3]
    du = np.zeros((TEX, TEX), np.float32)
    dv = np.zeros((TEX, TEX), np.float32)
    du[:, 1:-1] = np.linalg.norm(full[:, 2:] - full[:, :-2], axis=2) * 0.5
    dv[1:-1, :] = np.linalg.norm(full[2:, :] - full[:-2, :], axis=2) * 0.5
    okdu = cover.copy()
    okdu[:, 1:-1] &= cover[:, 2:] & cover[:, :-2]
    okdv = cover.copy()
    okdv[1:-1, :] &= cover[2:, :] & cover[:-2, :]
    med = float(np.median(du[okdu & (du > 0)]))
    du = np.where(okdu & (du > 0) & (du < med * 4), du, med)
    dv = np.where(okdv & (dv > 0) & (dv < med * 4), dv, med)
    print(f"texel size about {med * 1000:.3f} mm, {1.0 / (med * 1000):.2f} texels per mm")

    H = np.zeros(n, np.float32)       # height in millimetres
    MM = 1000.0
    Pm = P * MM                       # positions in millimetres

    # leather grain: pebbled cells with soft creases between them
    F1, F2 = worley(Pm, 1.15, seed=3)
    grain = smoothstep(0.0, 0.32, F2 - F1)
    H += 0.045 * grain
    fine = fbm(Pm, 0.6, 2, seed=9)
    H += 0.015 * (fine - 0.5)
    mott = fbm(Pm, 9.0, 4, seed=21)
    H += 0.05 * (fbm(Pm, 3.0, 3, seed=33) - 0.5)

    # creases across the finger joints, deeper on the palm side
    crease = np.zeros(n, np.float32)
    wear = np.zeros(n, np.float32)
    chains = {name: finger_chain(name) for name in FINGERS}
    tpts = THUMB_PTS
    tdirs = [(tpts[i + 1] - tpts[i]).normalized() for i in range(3)]
    for name in FINGERS + ["thumb"]:
        if name == "thumb":
            pts, dirs = tpts, tdirs
            joints = [(1, tdirs[0], THUMB_NAIL), (2, tdirs[1], THUMB_NAIL)]
            hw = THUMB_HALF_WIDTH
        else:
            pts, dirs, side = chains[name]
            joints = [(k, dirs[max(0, k - 1)], side.cross(dirs[max(0, k - 1)])) for k in (0, 1, 2)]
            hw = FINGER[name][3]
        for k, d, up in joints:
            J = np.array(pts[k], np.float32) * MM
            dd = np.array((d + (dirs[k] if k < len(dirs) else d)).normalized(), np.float32)
            uu = np.array(up.normalized(), np.float32)
            rel = Pm - J
            ax = rel @ dd
            radial = np.linalg.norm(rel - np.outer(ax, dd), axis=1)
            near = (radial < hw * MM * 1.6) & (np.abs(ax) < 9.0)
            env = np.exp(-(ax / 4.0) ** 2) * near
            dors = rel @ uu
            wave = np.sin(ax * (2 * np.pi / 1.7) + 0.8 * fbm(Pm, 2.0, 2, seed=k + 50) * 6.0)
            side_w = np.where(dors > 0, 0.55, 1.0)
            crease += env * side_w * (0.5 + 0.5 * wave) ** 3
            # scuffing on the back of each knuckle
            kn = J + uu * hw * MM * 0.95
            dist = np.linalg.norm(Pm - kn, axis=1)
            wear += np.exp(-(dist / (5.5 if k == 0 else 4.2)) ** 2) * (dors > 0) * near * (1.0 if k else 0.85)
        tip = np.array(pts[-1], np.float32) * MM
        wear += 0.8 * np.exp(-(np.linalg.norm(Pm - tip, axis=1) / 5.0) ** 2)
    H -= 0.09 * np.clip(crease, 0, 1.5)

    # the cuff rim and its edge take the most handling
    y = P[:, 1]
    radial_xz = np.sqrt(P[:, 0] ** 2 + P[:, 2] ** 2 * 1.6)
    outer_cuff = (y < -0.026) & (np.abs(N[:, 1]) < 0.8)
    wear += 0.7 * np.exp(-((y + 0.030) / 0.0035) ** 2) * outer_cuff
    wear += 0.6 * np.exp(-((y + 0.087) / 0.004) ** 2) * outer_cuff
    # the inside of the trigger finger rubs
    breakup = fbm(Pm, 2.2, 4, seed=77)
    wear = np.clip(wear, 0, 1.2) * smoothstep(0.25, 0.75, breakup + 0.25 * wear)
    wear = np.clip(wear, 0, 1)

    # stitching
    arc = seam_chains(me)
    vco = np.array([v.co for v in me.vertices], np.float32) * MM
    vno = np.array([v.normal for v in me.vertices], np.float32)
    best_d = np.full(n, 99.0, np.float32)
    best_s = np.zeros(n, np.float32)
    for ei, (va, sa, vb, sb) in arc.items():
        A, Bv = vco[va], vco[vb]
        AB = Bv - A
        L2 = max(float(AB @ AB), 1e-9)
        nrm = vno[va] + vno[vb]
        nrm /= max(np.linalg.norm(nrm), 1e-6)
        lo = np.minimum(A, Bv) - 4.0
        hi = np.maximum(A, Bv) + 4.0
        m = np.all((Pm > lo) & (Pm < hi), axis=1) & ((N @ nrm) > 0.15)
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        t = np.clip(((Pm[idx] - A) @ AB) / L2, 0, 1)
        d = np.linalg.norm(Pm[idx] - (A + np.outer(t, AB)), axis=1)
        better = d < best_d[idx]
        best_d[idx[better]] = d[better]
        best_s[idx[better]] = (sa + t[better] * (sb - sa)) * MM

    STITCH_OFF, STITCH_W, PERIOD = 2.3, 0.42, 3.1

    def stitch_rows(dist, s_mm, offsets):
        across = np.zeros(n, np.float32)
        for o in offsets:
            across = np.maximum(across, np.clip(1 - ((np.abs(dist) - o) / STITCH_W) ** 2, 0, 1))
        ph = np.mod(s_mm / PERIOD, 1.0)
        along = smoothstep(0.08, 0.2, ph) * (1 - smoothstep(0.62, 0.74, ph))
        return np.sqrt(across) * along

    stitch = stitch_rows(best_d, best_s, [STITCH_OFF])
    welt = np.exp(-(best_d / 1.3) ** 2)
    groove = np.exp(-(best_d / 0.32) ** 2)

    # decorative rows: a stitched band under the cuff rim and three points on the back
    ang = np.arctan2(P[:, 2], P[:, 0]) * 34.0
    for y0 in (-0.0345, -0.0385):
        stitch = np.maximum(stitch, stitch_rows((y - y0) * MM, ang, [0.0]) * outer_cuff)
    back = (N[:, 2] > 0.45) & (P[:, 2] > 0.004) & (y > 0.004) & (y < 0.07)
    for xa, xb in ((-0.0105, -0.0135), (0.0005, 0.0005), (0.0105, 0.0135)):
        A = np.array((xa, 0.008), np.float32)
        Bv = np.array((xb, 0.064), np.float32)
        AB = Bv - A
        t = np.clip(((P[:, :2] - A) @ AB) / float(AB @ AB), 0, 1)
        dist = np.linalg.norm(P[:, :2] - (A + np.outer(t, AB)), axis=1) * MM
        fade = smoothstep(0.0, 0.08, t) * (1 - smoothstep(0.9, 1.0, t))
        row = stitch_rows(dist, t * np.linalg.norm(AB) * MM, [0.0]) * back * (fade > 0.5)
        stitch = np.maximum(stitch, row)
        H -= 0.05 * np.exp(-(dist / 0.5) ** 2) * back * fade

    H += 0.10 * welt - 0.12 * groove
    H = H * (1 - 0.6 * stitch) + 0.16 * stitch

    # colour, linear
    base = np.array((0.020, 0.0145, 0.0138), np.float32)
    worn = np.array((0.085, 0.064, 0.052), np.float32)
    thread = np.array((0.085, 0.010, 0.012), np.float32)
    shade = (0.78 + 0.45 * mott) * (0.86 + 0.22 * grain)
    col = base[None, :] * shade[:, None]
    col = col * (1 - 0.25 * np.clip(crease, 0, 1))[:, None]
    col = col + (worn[None, :] * (0.7 + 0.6 * grain)[:, None] - col) * (wear[:, None] * 0.85)
    col = col * (1 - 0.35 * groove)[:, None]
    col = col + (thread[None, :] * (0.8 + 0.3 * fine)[:, None] - col) * stitch[:, None]

    rough = 0.58 + 0.12 * (1 - grain) + 0.05 * (mott - 0.5) + 0.08 * np.clip(crease, 0, 1)
    rough = rough - 0.2 * wear
    rough = rough + (0.78 - rough) * stitch
    rough = np.clip(rough, 0.22, 0.95)

    # height to tangent space normals through texel-space slopes
    Himg = np.zeros((TEX, TEX), np.float32)
    Himg[cover] = H
    Himg = dilate(Himg, cover, 3)
    gx = np.zeros_like(Himg)
    gy = np.zeros_like(Himg)
    gx[:, 1:-1] = (Himg[:, 2:] - Himg[:, :-2]) * 0.5
    gy[1:-1, :] = (Himg[2:, :] - Himg[:-2, :]) * 0.5
    sx = gx / (du * MM)
    sy = gy / (dv * MM)
    STRENGTH = 1.6
    nx, ny = -sx * STRENGTH, -sy * STRENGTH
    nz = np.ones_like(nx)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    nmap = np.stack([nx / ln, ny / ln, nz / ln], axis=2) * 0.5 + 0.5

    colimg = np.zeros((TEX, TEX, 3), np.float32)
    colimg[cover] = col
    rimg = np.zeros((TEX, TEX), np.float32)
    rimg[cover] = rough
    colimg = dilate(colimg, cover, 12)
    rimg = dilate(rimg, cover, 12)
    nmap = dilate(nmap, cover, 12)

    def to_srgb(c):
        c = np.clip(c, 0, 1)
        return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)

    tex_dir = os.path.join(OUT, "textures")
    os.makedirs(tex_dir, exist_ok=True)
    paths = {}
    for key, data, cs in (("basecolor", to_srgb(colimg), "sRGB"),
                          ("roughness", np.repeat(rimg[:, :, None], 3, axis=2), "Non-Color"),
                          ("normal", nmap, "Non-Color")):
        img = bpy.data.images.new(f"glove_{key}", TEX, TEX, alpha=False, float_buffer=False)
        rgba = np.ones((TEX, TEX, 4), np.float32)
        rgba[:, :, :3] = data
        img.pixels.foreach_set(rgba.ravel())
        path = os.path.join(tex_dir, f"glove_{key}.png")
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        img.colorspace_settings.name = cs
        paths[key] = img
    for im in ("bake_cover", "bake_position", "bake_normal_obj"):
        bpy.data.images.remove(bpy.data.images[im])
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)

    mat = build_material(paths)
    ob.data.materials.clear()
    ob.data.materials.append(mat)

    make_left(ob, rig)
    make_pose_library()
    scn.render.engine = "CYCLES"
    for img in paths.values():
        img.filepath = "//textures/" + os.path.basename(img.filepath_raw)
    bpy.context.preferences.filepaths.save_version = 0   # no gloves.blend1 beside it
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "gloves.blend"), relative_remap=False)
    print("saved", os.path.join(OUT, "gloves.blend"))


def shift(a, dy, dx):
    """Shift an image by whole texels, filling with zeros instead of wrapping."""
    import numpy as np
    out = np.zeros_like(a)
    H, W = a.shape[:2]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


def dilate(img, mask, steps):
    """Grow the texels under mask outward, so mip levels do not pull in background."""
    import numpy as np
    img = np.where(mask[:, :, None] if img.ndim == 3 else mask, img, 0).astype(np.float32)
    m = mask.copy()
    for _ in range(steps):
        acc = np.zeros_like(img)
        cnt = np.zeros(m.shape, np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            sm = shift(m, dy, dx).astype(np.float32)
            si = shift(img, dy, dx)
            acc += si * (sm[:, :, None] if img.ndim == 3 else sm)
            cnt += sm
        grow = (~m) & (cnt > 0)
        if img.ndim == 3:
            img[grow] = acc[grow] / cnt[grow][:, None]
        else:
            img[grow] = acc[grow] / cnt[grow]
        m = m | grow
    return img


def build_material(imgs):
    mat = bpy.data.materials.new("glove_leather")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(nd for nd in nt.nodes if nd.type == "BSDF_PRINCIPLED")
    uv = nt.nodes.new("ShaderNodeUVMap")
    nodes = {}
    for i, key in enumerate(("basecolor", "roughness", "normal")):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = imgs[key]
        t.location = (-600, 300 - 300 * i)
        nt.links.new(uv.outputs[0], t.inputs[0])
        nodes[key] = t
    nt.links.new(nodes["basecolor"].outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(nodes["roughness"].outputs["Color"], bsdf.inputs["Roughness"])
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.location = (-250, -350)
    nt.links.new(nodes["normal"].outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.45
    return mat


def make_left(ob_r, rig_r):
    me_l = ob_r.data.copy()
    me_l.name = "glove_left"
    me_l.transform(Matrix.Scale(-1.0, 4, (1, 0, 0)))
    me_l.flip_normals()
    me_l.update()
    ob_l = ob_r.copy()
    ob_l.name = "glove_left"
    ob_l.data = me_l
    bpy.context.scene.collection.objects.link(ob_l)
    rig_l = build_armature("rig_left", mirror=True)
    rig_l.data.display_type = "OCTAHEDRAL"
    ob_l.parent = rig_l
    ob_l.matrix_parent_inverse = Matrix.Identity(4)
    for m in ob_l.modifiers:
        if m.type == "ARMATURE":
            m.object = rig_l
    rig_l.location.x = -0.25   # side by side in the .blend; the mesh follows its rig
    return ob_l, rig_l


def make_pose_library():
    """One single-frame action per pose and hand, marked as pose assets."""
    for side, mirror in (("right", False), ("left", True)):
        rig = bpy.data.objects[f"rig_{side}"]
        rig.animation_data_create()
        for name in POSE_ORDER:
            act = bpy.data.actions.new(f"{side}_{name}")
            rig.animation_data.action = act
            apply_pose(rig, name, mirror)
            for pb in rig.pose.bones:
                pb.keyframe_insert("rotation_quaternion", frame=1, group=pb.name)
            act.use_fake_user = True
            try:
                act.asset_mark()
                act.asset_data.tags.new(side)
            except Exception as ex:
                print("asset mark skipped", ex)
        rig.animation_data.action = None
        apply_pose(rig, None)

# ---------------------------------------------------------------------------
# Stage: export. Converts each hand to the shipped frame in memory (Y up,
# fingers along -Z, palm toward -Y), writes USDZ and poses.json, and never
# saves the converted state back over gloves.blend.
# ---------------------------------------------------------------------------

TO_YUP = Matrix.Rotation(math.radians(-90.0), 4, "X")   # Blender (x, y, z) -> (x, z, -y)


def evaluated_coords(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    em = ev.to_mesh()
    co = [ob.matrix_world @ v.co for v in em.vertices]
    ev.to_mesh_clear()
    return co


def mirror_check():
    """Posed left hand against the mirrored posed right hand, vertex by vertex."""
    rr, rl = bpy.data.objects["rig_right"], bpy.data.objects["rig_left"]
    gr, gl = bpy.data.objects["glove_right"], bpy.data.objects["glove_left"]
    worst = 0.0
    for name in POSE_ORDER:
        apply_pose(rr, name, False)
        apply_pose(rl, name, True)
        a = evaluated_coords(gr)
        b = evaluated_coords(gl)
        off = rl.matrix_world.translation - rr.matrix_world.translation
        for va, vb in zip(a, b):
            worst = max(worst, (Vector((-va.x, va.y, va.z)) - (vb - off)).length)
    apply_pose(rr, None)
    apply_pose(rl, None, True)
    print(f"mirror check: worst vertex gap between left and mirrored right over all poses {worst * 1000:.4f} mm")
    return worst


def convert_hand(side):
    rig = bpy.data.objects[f"rig_{side}"]
    ob = bpy.data.objects[f"glove_{side}"]
    apply_pose(rig, None)
    ob.parent = None
    for o in (rig, ob):
        o.matrix_world = TO_YUP.copy()
    bpy.context.view_layer.update()
    for o in (rig, ob):
        select_only(o)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    ob.parent = rig
    ob.matrix_parent_inverse = Matrix.Identity(4)
    bpy.context.view_layer.update()
    return rig, ob


def pose_table(rig, mirror):
    table = {}
    for name in ["rest"] + POSE_ORDER:
        apply_pose(rig, None if name == "rest" else name, mirror)
        joints = {}
        for pb in rig.pose.bones:
            m = pb.matrix if pb.parent is None else pb.parent.matrix.inverted() @ pb.matrix
            q = m.to_quaternion().normalized()
            d = pb.rotation_quaternion.normalized()
            joints[pb.name] = {"local": q, "delta": d, "t": m.translation.copy()}
        table[name] = joints
    apply_pose(rig, None)
    return table


def q_xyzw(q):
    if q.w < 0:
        q = -q
    return [round(q.x, 6), round(q.y, 6), round(q.z, 6), round(q.w, 6)]


def stage_export():
    import subprocess
    from pxr import Gf, Usd, UsdGeom, UsdSkel, UsdUtils
    bpy.ops.wm.open_mainfile(filepath=os.path.join(OUT, "gloves.blend"))
    report = {"mirror_gap_mm": mirror_check() * 1000.0}
    bpy.data.objects["rig_left"].location.x = 0.0
    bpy.context.view_layer.update()
    hands = {}
    for side, mirror in (("right", False), ("left", True)):
        rig, ob = convert_hand(side)
        table = pose_table(rig, mirror)
        stem = f"glove_{side}"
        wdir = os.path.join(WORK, stem)
        os.makedirs(wdir, exist_ok=True)
        usdc = os.path.join(wdir, stem + ".usdc")
        select_only(ob, rig)
        bpy.ops.wm.usd_export(
            filepath=usdc, selected_objects_only=True, visible_objects_only=False,
            export_animation=False, export_armatures=True, only_deform_bones=False,
            export_shapekeys=False, export_materials=True, generate_preview_surface=True,
            export_textures=True, overwrite_textures=True, relative_paths=True,
            export_uvmaps=True, rename_uvmaps=True, export_normals=True,
            triangulate_meshes=True, convert_orientation=False, root_prim_path=f"/{stem}",
            export_custom_properties=False, author_blender_name=False,
            export_lights=False, export_cameras=False, convert_world_material=False)
        stage = Usd.Stage.Open(usdc)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetDefaultPrim(stage.GetPrimAtPath(f"/{stem}"))
        stage.GetRootLayer().Save()
        usdz = os.path.join(OUT, stem + ".usdz")
        if os.path.exists(usdz):
            os.remove(usdz)
        UsdUtils.CreateNewARKitUsdzPackage(usdc, usdz)

        # read back what actually shipped
        st = Usd.Stage.Open(usdz)
        info = {"upAxis": str(UsdGeom.GetStageUpAxis(st)),
                "metersPerUnit": UsdGeom.GetStageMetersPerUnit(st)}
        skel = next(UsdSkel.Skeleton(p) for p in st.Traverse() if p.IsA(UsdSkel.Skeleton))
        jpaths = [str(j) for j in skel.GetJointsAttr().Get()]
        names = [j.split("/")[-1] for j in jpaths]
        rest = skel.GetRestTransformsAttr().Get()
        topo = UsdSkel.Topology(skel.GetJointsAttr().Get())
        parents = [int(topo.GetParent(i)) for i in range(len(names))]
        worst_t, worst_r = 0.0, 0.0
        rest_out = []
        for i, nm in enumerate(names):
            m = Gf.Matrix4d(rest[i])
            t = m.ExtractTranslation()
            r = m.ExtractRotationQuat()
            qb = table["rest"][nm]["local"]
            tb = table["rest"][nm]["t"]
            qu = Quaternion((r.GetReal(), *r.GetImaginary()))
            worst_t = max(worst_t, (Vector(tuple(t)) - tb).length)
            worst_r = max(worst_r, 1.0 - abs(qu.dot(qb)))
            rest_out.append({"t": [round(c, 6) for c in tb], "r": q_xyzw(qb)})
        info["restTransform_vs_json_max_translation_m"] = worst_t
        info["restTransform_vs_json_max_rotation_dot_error"] = worst_r
        meshp = next(UsdGeom.Mesh(p) for p in st.Traverse() if p.IsA(UsdGeom.Mesh))
        pts = meshp.GetPointsAttr().Get()
        counts = meshp.GetFaceVertexCountsAttr().Get()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        info["triangles"] = len(counts)
        info["vertices"] = len(pts)
        info["bounds_min"] = [round(min(xs), 4), round(min(ys), 4), round(min(zs), 4)]
        info["bounds_max"] = [round(max(xs), 4), round(max(ys), 4), round(max(zs), 4)]
        model = {}
        xf = skel.GetBindTransformsAttr().Get()
        for i, nm in enumerate(names):
            model[nm] = Gf.Matrix4d(xf[i]).ExtractTranslation()
        info["middleFingerTip_bind_position"] = [round(c, 4) for c in model["middleFingerTip"]]
        info["thumbTip_bind_position"] = [round(c, 4) for c in model["thumbTip"]]
        info["joint_count"] = len(names)
        report[side] = info
        print(side, json.dumps(info, indent=1))

        poses = {}
        for pname in ["rest"] + POSE_ORDER:
            poses[pname] = {nm: {"local": q_xyzw(table[pname][nm]["local"]),
                                 "delta": q_xyzw(table[pname][nm]["delta"])} for nm in names}
        worst_delta = 0.0
        for pname in POSE_ORDER:
            for nm in names:
                q = table["rest"][nm]["local"] @ table[pname][nm]["delta"]
                worst_delta = max(worst_delta, 1.0 - abs(q.normalized().dot(table[pname][nm]["local"])))
        report[side]["local_equals_rest_times_delta_max_error"] = worst_delta
        print(side, "local = rest * delta, worst error", worst_delta)
        hands[side] = {"joints": names, "parents": parents, "rest": rest_out, "poses": poses}
        r = subprocess.run(["/usr/bin/usdchecker", "--arkit", usdz], capture_output=True, text=True)
        print("usdchecker", side, r.returncode, r.stdout.strip(), r.stderr.strip())
        report[side]["usdchecker"] = (r.stdout + r.stderr).strip()

    doc = {
        "about": "Hand poses for glove_right.usdz and glove_left.usdz.",
        "units": "metres",
        "upAxis": "Y",
        "frame": "Y up, fingers toward -Z, palm toward -Y, right thumb toward -X, wrist joint at the origin",
        "quaternion": "x, y, z, w",
        "local": "joint rotation relative to its parent joint, the same space as the USD skeleton restTransforms",
        "delta": "rotation relative to the rest pose, in the joint's own frame: local = rest.r * delta",
        "jointFrame": "Y along the bone toward the child joint, Z out of the back of the hand (nail side)",
        "poseOrder": ["rest"] + POSE_ORDER,
        "hands": hands,
    }
    with open(os.path.join(OUT, "poses.json"), "w") as f:
        json.dump(doc, f, indent=1)
    with open(os.path.join(WORK, "export_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)

# ---------------------------------------------------------------------------
# Stage: preview. Night lit Cycles stills of the right hand.
# ---------------------------------------------------------------------------

def stage_preview():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(OUT, "gloves.blend"))
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = int(os.environ.get("GLOVE_SAMPLES", "96"))
    scn.cycles.use_denoising = True
    scn.render.resolution_x, scn.render.resolution_y = 1200, 900
    scn.render.image_settings.file_format = "PNG"
    try:
        scn.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    scn.view_settings.exposure = 0.3
    for o in bpy.data.objects:
        if o.name.endswith("_left"):
            o.hide_render = True
    world = bpy.data.worlds.new("night")
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.006, 0.008, 0.016, 1)
    bg.inputs["Strength"].default_value = 1.0
    scn.world = world

    def light(name, kind, loc, target, color, power, size=0.3):
        ld = bpy.data.lights.new(name, kind)
        ld.color = color
        ld.energy = power
        if kind == "AREA":
            ld.size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        scn.collection.objects.link(lo)
    light("moon", "AREA", (0.35, -0.25, 0.75), (0, 0.06, 0), (0.62, 0.72, 1.0), 13.0, 0.8)
    light("sodium", "AREA", (0.30, 0.45, 0.10), (0, 0.06, 0), (1.0, 0.55, 0.22), 1.4, 0.3)
    light("neon", "AREA", (-0.35, 0.05, -0.15), (0, 0.06, 0), (0.7, 0.06, 0.10), 0.8, 0.3)

    cam_d = bpy.data.cameras.new("eye")
    cam = bpy.data.objects.new("eye", cam_d)
    scn.collection.objects.link(cam)
    scn.camera = cam
    views = {
        # roughly where the player's eye sits for a right hand held out at chest height
        "firstperson": ((-0.08, -0.26, 0.27), (0.0, 0.09, -0.02), 38.0),
        "threequarter": ((-0.27, 0.30, 0.16), (0.0, 0.065, -0.01), 50.0),
    }
    rig = bpy.data.objects["rig_right"]
    pdir = os.path.join(OUT, "preview")
    os.makedirs(pdir, exist_ok=True)
    shots = [(p, v) for p in ("relaxed", "grip", "point") for v in views]
    shots.append(("relaxed", "detail"))
    views["detail"] = ((0.05, 0.02, 0.17), (-0.005, 0.085, 0.0), 60.0)
    only = os.environ.get("GLOVE_SHOTS")
    for pose, vname in shots:
        if only and f"{pose}_{vname}" not in only.split(","):
            continue
        apply_pose(rig, pose)
        eye, tgt, lens = views[vname]
        cam.location = eye
        cam.rotation_euler = (Vector(tgt) - Vector(eye)).to_track_quat("-Z", "Y").to_euler()
        cam_d.lens = lens
        scn.render.filepath = os.path.join(pdir, f"{pose}_{vname}.png")
        bpy.ops.render.render(write_still=True)
        print("wrote", scn.render.filepath)


if STAGE == "model":
    stage_model()
elif STAGE == "check":
    stage_check()
elif STAGE == "bake":
    stage_bake()
elif STAGE == "export":
    stage_export()
elif STAGE == "preview":
    stage_preview()
