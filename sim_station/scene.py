"""MuJoCo digital twin of the Everest dual-YAM water-cup station.

Every measured number comes from `layout.py`. This module only turns those numbers
into an MJCF model:

* both YAM arms (vendor geometry and inertias, torque-limited joints with rotor
  inertia, friction and viscous damping),
* collision pads taken from the real fingertip mesh (V-funnel + flat tip), a gripper
  housing box and capsule proxies for the links, so hands, wrists and forearms are real
  obstacles,
* hollow tapered 16-stave paper/plastic cups (no solid lid), nested inverted stacks
  made of one free top cup and one free rest-of-stack body,
* the dispenser: octagonal tank, horizontal barrel, vertical valve body and a passive
  swivel + lift handle (no actuators on the handle),
* a walnut coaster with a funnel recess that self-centres a lowered cup.
"""

from __future__ import annotations

import os
import warnings
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from sim_station import layout as L

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Collision bit masks. Two geoms collide when (type_a & aff_b) | (type_b & aff_a).
ENV_T, ENV_A = 1, 7          # table, mat, dispenser, coaster, cups, stacks, handle
CUP_T, CUP_A = 2, 5          # cup arm parts collide with env + tap arm
TAP_T, TAP_A = 4, 3          # tap arm parts collide with env + cup arm
ARM_MASK = {"cup": (CUP_T, CUP_A), "tap": (TAP_T, TAP_A)}

# Contact softness. Light cups under tens of newtons need stiff, fast contacts.
CONTACT_SOLREF = "0.003 1"
CONTACT_SOLIMP = "0.96 0.995 0.0004"
STACK_COUNT_MAX = L.NEST_MAX - 1          # rest-of-stack carries up to this many cups


# ---------------------------------------------------------------------------
# small XML / math helpers
# ---------------------------------------------------------------------------

def _s(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.7g}"
    return " ".join(_s(v) for v in np.asarray(x, dtype=float).reshape(-1))


def E(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    return ET.SubElement(parent, tag, {k: _s(v) for k, v in attrs.items() if v is not None})


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def rot_axes(*cols: np.ndarray) -> np.ndarray:
    return np.stack(cols, axis=1)


def yaw_quat(yaw: float) -> str:
    return f"{np.cos(0.5 * yaw):.7f} 0 0 {np.sin(0.5 * yaw):.7f}"


# ---------------------------------------------------------------------------
# procedural meshes and textures
# ---------------------------------------------------------------------------

def _obj(path, vertices, faces, uvs=None) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# station mesh\n")
        for x, y, z in vertices:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        if uvs:
            for u, v in uvs:
                fh.write(f"vt {u:.6f} {v:.6f}\n")
        for a, b, c in faces:
            fh.write(f"f {a}/{a} {b}/{b} {c}/{c}\n" if uvs else f"f {a} {b} {c}\n")


def _ring(radius, z, n, phase):
    return [(radius * np.cos(phase + 2 * np.pi * i / n), radius * np.sin(phase + 2 * np.pi * i / n), z) for i in range(n)]


def _tube(r_out, r_in, z0, z1, n=8, phase=np.pi / 8):
    """Open-top faceted acrylic shell (octagonal)."""
    rings = _ring(r_out, z0, n, phase) + _ring(r_out, z1, n, phase) + _ring(r_in, z1, n, phase) + _ring(r_in, z0, n, phase)
    verts = rings + [(0.0, 0.0, z0 + 0.004)]
    floor_c = len(rings)
    faces = []

    def tri(a, b, c):
        faces.append((a + 1, b + 1, c + 1))

    for i in range(n):
        j = (i + 1) % n
        ob, ot, it, ib = i, n + i, 2 * n + i, 3 * n + i
        jb, jt, kt, kb = j, n + j, 2 * n + j, 3 * n + j
        tri(ob, jb, jt); tri(ob, jt, ot)
        tri(it, kt, kb); tri(it, kb, ib)
        tri(ot, jt, kt); tri(ot, kt, it)
        tri(ib, floor_c, kb)
    return verts, faces


def _prism(radius, z0, z1, n=8, phase=np.pi / 8):
    """Closed convex prism (collision hull of the tank)."""
    bot = _ring(radius, z0, n, phase)
    top = _ring(radius, z1, n, phase)
    verts = bot + top + [(0, 0, z0), (0, 0, z1)]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i + 1, j + 1, n + j + 1)); faces.append((i + 1, n + j + 1, n + i + 1))
        faces.append((2 * n + 1, j + 1, i + 1))
        faces.append((2 * n + 2, n + i + 1, n + j + 1))
    return verts, faces


def _frustum_open(r0, z0, r1, z1, n=48):
    verts, uvs = [], []
    for radius, z, v in ((r0, z0, 0.0), (r1, z1, 1.0)):
        for i in range(n):
            a = 2 * np.pi * i / n
            verts.append((radius * np.cos(a), radius * np.sin(a), z))
            uvs.append((i / n, v))
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i + 1, i + n + 1, j + n + 1))
        faces.append((i + 1, j + n + 1, j + 1))
    return verts, uvs, faces


def _frustum_solid(r0, z0, r1, z1, n=24):
    verts = [(r0 * np.cos(2 * np.pi * i / n), r0 * np.sin(2 * np.pi * i / n), z0) for i in range(n)]
    verts += [(r1 * np.cos(2 * np.pi * i / n), r1 * np.sin(2 * np.pi * i / n), z1) for i in range(n)]
    verts += [(0, 0, z0), (0, 0, z1)]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i + 1, j + 1, j + n + 1)); faces.append((i + 1, j + n + 1, i + n + 1))
        faces.append((2 * n + 1, j + 1, i + 1))
        faces.append((2 * n + 2, i + n + 1, j + n + 1))
    return verts, faces


def _annulus(r_in, r_out, z0, z1, n=48):
    verts = []
    for r, z in ((r_out, z0), (r_out, z1), (r_in, z1), (r_in, z0)):
        verts += [(r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), z) for i in range(n)]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        for k in range(4):
            a = k * n
            b = ((k + 1) % 4) * n
            faces.append((a + i + 1, a + j + 1, b + j + 1)); faces.append((a + i + 1, b + j + 1, b + i + 1))
    return verts, faces


def _disc(radius, z, n=48):
    verts = [(0, 0, z)] + [(radius * np.cos(2 * np.pi * i / n), radius * np.sin(2 * np.pi * i / n), z) for i in range(n)]
    faces = [(1, 2 + (i + 1) % n, 2 + i) for i in range(n)]
    return verts, faces


def _ensure_assets() -> None:
    if os.environ.get("SIM_ASSETS_READY") == "1":          # parallel workers reuse what the parent already wrote
        return
    os.makedirs(ASSET_DIR, exist_ok=True)
    half = L.CUP_HEIGHT * 0.5
    upper, upper_f = _tube(L.TANK_CIRCUM_RADIUS, L.TANK_CIRCUM_RADIUS - 0.0035, 0.094, 0.380)
    lower, lower_f = _tube(L.TANK_CIRCUM_RADIUS - 0.007, L.TANK_CIRCUM_RADIUS - 0.0105, 0.008, 0.096)
    _obj(os.path.join(ASSET_DIR, "tank_upper.obj"), upper, upper_f)
    _obj(os.path.join(ASSET_DIR, "tank_lower.obj"), lower, lower_f)
    sv, sf = _prism(L.TANK_CIRCUM_RADIUS - 0.002, 0.008, 0.380)
    _obj(os.path.join(ASSET_DIR, "tank_solid.obj"), sv, sf)
    wv, wuv, wf = _frustum_open(L.CUP_BASE_RADIUS, -half, L.CUP_RIM_RADIUS, half)
    iv, iuv, iff = _frustum_open(L.CUP_BASE_RADIUS - 0.0022, -half + 0.0035, L.CUP_RIM_RADIUS - 0.0022, half - 0.0006)
    iff = [(a, c, b) for a, b, c in iff]
    _obj(os.path.join(ASSET_DIR, "cup_wall.obj"), wv, wf, wuv)
    _obj(os.path.join(ASSET_DIR, "cup_inner.obj"), iv, iff, iuv)
    rv, rf = _annulus(L.CUP_RIM_RADIUS - 0.0026, L.CUP_RIM_RADIUS + 0.0016, half - 0.0035, half + 0.0006)
    _obj(os.path.join(ASSET_DIR, "cup_rim.obj"), rv, rf)
    fv, ff = _disc(L.CUP_BASE_RADIUS - 0.0018, -half + 0.0035)
    _obj(os.path.join(ASSET_DIR, "cup_floor.obj"), fv, ff)
    # inverted-cup collision pieces: rim at z = 0, base at z = h (radius shrinks with height)
    k = L.CUP_TAPER
    bv, bf = _frustum_solid(L.STACK_BAND_RADIUS, 0.0, L.STACK_BAND_RADIUS - k * L.NEST_STEP, L.NEST_STEP)
    _obj(os.path.join(ASSET_DIR, "band_frustum.obj"), bv, bf)
    fv2, ff2 = _frustum_solid(L.STACK_BAND_RADIUS, 0.0, L.STACK_BAND_RADIUS - k * L.CUP_HEIGHT, L.CUP_HEIGHT)
    _obj(os.path.join(ASSET_DIR, "full_frustum.obj"), fv2, ff2)
    _write_textures()


def _write_textures() -> None:
    ripple = os.path.join(ASSET_DIR, "ripple.png")
    wood = os.path.join(ASSET_DIR, "wood.png")
    mat = os.path.join(ASSET_DIR, "mat.png")
    coaster = os.path.join(ASSET_DIR, "coaster.png")
    height, width = 512, 256
    ys = np.linspace(0, 1, height)[:, None]
    xs = np.linspace(0, 1, width)[None, :]
    wave = np.sin((ys * 46.0 + 0.42 * np.sin(xs * 2 * np.pi * 8)) * np.pi)
    shade = 0.66 + 0.34 * (0.5 + 0.5 * wave)
    tan = np.array([198, 164, 116], dtype=float)
    image = np.clip(tan * shade[..., None], 0, 255).astype(np.uint8)
    image[:24, :, :] = 244
    image[-34:, :, :] = 242
    Image.fromarray(image, mode="RGB").save(ripple)

    rng = np.random.default_rng(4)
    gh, gw = 256, 512
    gy = np.linspace(0, 16, gh)[:, None]
    gx = np.linspace(0, 30, gw)[None, :]
    grain = np.sin(gy * 0.9 + np.sin(gx * 0.15) * 1.2)
    fine = rng.normal(0, 0.025, (gh, gw))
    value = np.clip(0.88 + 0.09 * grain + fine, 0.55, 1.0)
    for out, base in ((wood, (124, 76, 42)), (coaster, (150, 98, 54))):
        board = np.zeros((gh, gw, 3), np.uint8)
        for c, b in enumerate(base):
            board[..., c] = np.clip(b * value, 0, 255)
        Image.fromarray(board, mode="RGB").save(out)

    side = 1024
    img = Image.new("RGB", (side, side), (74, 98, 108))
    draw = ImageDraw.Draw(img)
    step = 32
    for i in range(0, side, step // 4):
        draw.line([(i, 0), (i, side - 1)], fill=(84, 108, 118), width=1)
        draw.line([(0, i), (side - 1, i)], fill=(84, 108, 118), width=1)
    for i in range(0, side, step):
        draw.line([(i, 0), (i, side - 1)], fill=(206, 216, 220), width=1)
        draw.line([(0, i), (side - 1, i)], fill=(206, 216, 220), width=1)
    for i in range(0, side, step * 5):
        draw.line([(i, 0), (i, side - 1)], fill=(236, 242, 244), width=2)
        draw.line([(0, i), (side - 1, i)], fill=(236, 242, 244), width=2)
    cx, cy = int(side * 0.5), int(side * 0.5)
    for r in (150, 260, 380):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(180, 196, 202), width=1)
    for deg in range(0, 180, 15):
        a = np.radians(deg)
        draw.line([(cx - 400 * np.cos(a), cy - 400 * np.sin(a)), (cx + 400 * np.cos(a), cy + 400 * np.sin(a))],
                  fill=(150, 170, 178), width=1)
    img.save(mat)


# ---------------------------------------------------------------------------
# vendor arm: fragments, collision proxies, pads
# ---------------------------------------------------------------------------

_REF_ATTRS = ("name", "mesh", "joint", "joint1", "joint2", "body1", "body2", "site", "material", "texture")
_VENDOR_CACHE: Dict[str, object] = {}


def _prefix(elem: ET.Element, prefix: str) -> None:
    for node in elem.iter():
        for attr in _REF_ATTRS:
            value = node.get(attr)
            if value:
                node.set(attr, prefix + value)


def _vendor():
    """Compiled vendor model at q = 0 for frame algebra, plus PCA capsules of the link meshes."""
    if _VENDOR_CACHE:
        return _VENDOR_CACHE
    path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)

    def mesh_verts_in_body(gid: int) -> np.ndarray:
        mid = int(model.geom_dataid[gid])
        vs, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        v = np.array(model.mesh_vert[vs:vs + vn], dtype=float)
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, model.geom_quat[gid])
        return v @ Rg.reshape(3, 3).T + model.geom_pos[gid]

    capsules = {}
    for gid in range(model.ngeom):
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        body = model.body(int(model.geom_bodyid[gid])).name
        if body not in ("link2", "link3", "link4", "link5"):
            continue
        v = mesh_verts_in_body(gid)
        c = v.mean(axis=0)
        u, s, vt = np.linalg.svd(v - c, full_matrices=False)
        axis = vt[0]
        proj = (v - c) @ axis
        perp = (v - c) - np.outer(proj, axis)
        radius = float(np.percentile(np.linalg.norm(perp, axis=1), 80))
        lo, hi = float(proj.min()) + radius, float(proj.max()) - radius
        capsules[body] = (c + axis * lo, c + axis * hi, radius)

    site = data.site("grasp_site")
    _VENDOR_CACHE.update(
        model=model, data=data, capsules=capsules,
        site_p=np.array(site.xpos), site_R=np.array(site.xmat).reshape(3, 3),
    )
    return _VENDOR_CACHE


def site_to_body(body: str, pos_s: np.ndarray, R_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Express a pose given in the TCP (grasp_site) frame in a body's frame, at the closed q = 0 pose."""
    v = _vendor()
    data = v["data"]
    b = data.body(body)
    Rb = np.array(b.xmat).reshape(3, 3)
    pw = v["site_p"] + v["site_R"] @ np.asarray(pos_s, float)
    Rw = v["site_R"] @ np.asarray(R_s, float)
    return Rb.T @ (pw - np.array(b.xpos)), Rb.T @ Rw


def _arm_fragments(prefix: str):
    path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    root = ET.parse(path).getroot()
    assets = list(root.find("asset") or [])
    bodies = list(root.find("worldbody") or [])
    eqs = list(root.find("equality") or [])
    contacts = list(root.find("contact") or [])
    for group in (assets, bodies, eqs, contacts):
        for node in group:
            _prefix(node, prefix)
    arm_name = "cup" if prefix == "cup_" else "tap"
    for node in bodies:
        for geom in node.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("group", "1")
        for joint in node.iter("joint"):
            name = joint.get("name", "")
            idx = int(name[-1])
            if joint.get("type", "hinge") == "hinge":
                lim = float(L.ARM_TORQUE_LIMIT[idx - 1])
                joint.set("actuatorfrcrange", f"-{lim} {lim}")
                if idx == 4:
                    joint.set("range", f"-1.69297 {L.JOINT_HIGH[3]}")
                joint.set("armature", _s(L.JOINT_ARMATURE[idx - 1]))
                joint.set("damping", _s(L.JOINT_DAMPING[idx - 1]))
                joint.set("frictionloss", _s(L.JOINT_FRICTIONLOSS[idx - 1]))
            else:
                joint.set("armature", _s(L.FINGER_ARMATURE))
                joint.set("damping", _s(L.FINGER_DAMPING))
                joint.set("frictionloss", _s(L.FINGER_FRICTIONLOSS))
    _paint_arm(bodies, prefix)
    _add_collision(bodies, prefix, arm_name)
    return assets, bodies, eqs, contacts


def _paint_arm(bodies, prefix: str) -> None:
    shell = "shell_gray" if prefix == "cup_" else "shell_white"
    for node in bodies:
        for geom in node.iter("geom"):
            mesh = geom.get("mesh") or ""
            geom.attrib.pop("rgba", None)
            geom.set("material", shell if (mesh.endswith("link2") or mesh.endswith("link3")) else "arm_black")
        for site in node.iter("site"):
            site.set("rgba", "0 0 0 0")
            site.set("size", "0.001")
        for body in node.iter("body"):
            if not str(body.get("name", "")).endswith("gripper"):
                continue
            E(body, "geom", name=f"{prefix}camera", type="box", pos="0.028 0.012 -0.02", size="0.018 0.022 0.016",
              material="arm_black", contype=0, conaffinity=0, group=1)
            mark = prefix == "cup_"
            if mark:
                E(body, "geom", name=f"{prefix}mark", type="box", pos="0.047 0.012 -0.02", size="0.0012 0.010 0.010",
                  material="mark_white", contype=0, conaffinity=0, group=1)
            else:
                E(body, "geom", name=f"{prefix}mark_v", type="box", pos="0.047 0.012 -0.02", size="0.0012 0.003 0.011",
                  material="mark_white", contype=0, conaffinity=0, group=1)
                E(body, "geom", name=f"{prefix}mark_h", type="box", pos="0.0475 0.012 -0.02", size="0.001 0.011 0.003",
                  material="mark_white", contype=0, conaffinity=0, group=1)


def _add_collision(bodies, prefix: str, arm: str) -> None:
    """Link capsules, gripper housing box, funnel fingertip pads."""
    v = _vendor()
    ct, ca = ARM_MASK[arm]
    common = dict(contype=ct, conaffinity=ca, group=3, rgba="0.9 0.2 0.2 0.18")
    by_name = {}
    for node in bodies:
        for body in node.iter("body"):
            by_name[body.get("name", "")] = body
    for link, (p0, p1, r) in v["capsules"].items():
        host = by_name.get(prefix + link)
        if host is not None:
            E(host, "geom", name=f"{prefix}col_{link}", type="capsule", fromto=np.concatenate([p0, p1]), size=r,
              friction="0.6 0.005 0.0001", solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP, **common)
    gripper = by_name[prefix + "gripper"]
    pos_b, R_b = site_to_body("gripper", np.array([0.0, 0.0, -0.1185]), np.eye(3))
    E(gripper, "geom", name=f"{prefix}col_housing", type="box", pos=pos_b, quat=mat_to_quat(R_b),
      size="0.032 0.050 0.037", friction="0.6 0.005 0.0001", solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP, **common)
    # Fingertip pads from the real tip geometry: flat tip pad + 18 degree funnel (see layout.pad_boxes_site).
    for side_name, name, center, Rb, half in L.pad_boxes_site():
        body = by_name[prefix + f"tip_{side_name}"]
        pad = dict(contype=ct, conaffinity=ca, group=3, rgba="0.2 0.6 0.9 0.35", friction=f"{L.PAD_FRICTION} 0.01 0.0001",
                   condim=4, solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP, margin=0.0)
        p, R = site_to_body(f"tip_{side_name}", center, Rb)
        E(body, "geom", name=f"{prefix}pad_{side_name}_{name}", type="box", pos=p, quat=mat_to_quat(R), size=list(half), **pad)


# ---------------------------------------------------------------------------
# cups, stacks, coaster
# ---------------------------------------------------------------------------

def _stave_geoms(parent: ET.Element, prefix: str, r0: float, z0: float, r1: float, z1: float, n: int, thick: float,
                 mat_attrs: dict) -> None:
    """Hollow tapered shell as n tilted boxes. Outer surface runs (r0,z0)->(r1,z1)."""
    alpha = np.arctan2(r1 - r0, z1 - z0)
    length = float(np.hypot(r1 - r0, z1 - z0))
    r_mid, z_mid = 0.5 * (r0 + r1), 0.5 * (z0 + z1)
    nr, nz = np.cos(alpha), -np.sin(alpha)
    tr, tz = np.sin(alpha), np.cos(alpha)
    c_r = r_mid - 0.5 * thick * nr
    c_z = z_mid - 0.5 * thick * nz
    half_w = r_mid * np.tan(np.pi / n) + 0.0006
    for i in range(n):
        phi = 2 * np.pi * i / n
        rh = np.array([np.cos(phi), np.sin(phi), 0.0])
        ph = np.array([-np.sin(phi), np.cos(phi), 0.0])
        zh = np.array([0.0, 0.0, 1.0])
        xl = nr * rh + nz * zh
        zl = tr * rh + tz * zh
        R = rot_axes(xl, ph, zl)
        pos = c_r * rh + c_z * zh
        E(parent, "geom", name=f"{prefix}stave{i}", type="box", pos=pos, quat=mat_to_quat(R),
          size=[0.5 * thick, half_w, 0.5 * length], **mat_attrs)


def _cup_body(name: str, wall: str, inner: str, mass: float) -> ET.Element:
    body = ET.Element("body", {"name": name, "pos": "0 0 -0.6", "gravcomp": "1"})
    E(body, "freejoint", name=f"{name}_free")
    half = L.CUP_HEIGHT * 0.5
    phys = dict(contype=ENV_T, conaffinity=ENV_A, friction=f"{L.CUP_FRICTION} 0.01 0.0002", condim=4,
                solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP, group=3, rgba="0.9 0.7 0.2 0.0", margin=0.0)
    _stave_geoms(body, f"{name}_", L.CUP_BASE_RADIUS, -half, L.CUP_RIM_RADIUS, half, L.CUP_STAVE_COUNT, L.CUP_WALL_THICK, phys)
    E(body, "geom", name=f"{name}_floor", type="cylinder", pos=f"0 0 {-half + 0.0035}", size=[L.CUP_BASE_RADIUS - 0.0016, 0.0020],
      **phys)
    E(body, "inertial", pos="0 0 -0.006", mass=mass, diaginertia=[mass * 0.00155, mass * 0.00155, mass * 0.00108])
    clear = inner == "cup_clear"
    E(body, "geom", name=f"{name}_shell", type="mesh", mesh="cup_wall", material=wall, contype=0, conaffinity=0, group=1)
    if not clear:
        E(body, "geom", name=f"{name}_inside", type="mesh", mesh="cup_inner", material=inner, contype=0, conaffinity=0, group=1)
        E(body, "geom", name=f"{name}_bottom", type="mesh", mesh="cup_floor", material=inner, contype=0, conaffinity=0, group=1)
    E(body, "geom", name=f"{name}_lip", type="mesh", mesh="cup_rim", material="cup_white" if not clear else "cup_clear",
      contype=0, conaffinity=0, group=1)
    return body


def _rest_body(name: str, wall: str, inner: str, xy) -> ET.Element:
    body = ET.Element("body", {"name": name, "pos": f"{xy[0]:.4f} {xy[1]:.4f} {L.MAT_SURFACE_Z:.4f}"})
    E(body, "freejoint", name=f"{name}_free")
    E(body, "inertial", pos="0 0 0.06", mass=0.08, diaginertia="0.0002 0.0002 0.0001")
    col = dict(contype=ENV_T, conaffinity=ENV_A, friction=f"{L.CUP_FRICTION} 0.01 0.0002", condim=4,
               solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP, group=3, rgba="0.9 0.7 0.2 0.0", margin=0.0)
    clear = inner == "cup_clear"
    for i in range(STACK_COUNT_MAX):
        z = i * L.NEST_STEP
        E(body, "geom", name=f"{name}_vis{i}", type="mesh", mesh="cup_wall", pos=f"0 0 {z + L.CUP_HEIGHT * 0.5:.5f}",
          quat="0 1 0 0", material=wall, contype=0, conaffinity=0, group=1)
        if not clear:
            E(body, "geom", name=f"{name}_in{i}", type="mesh", mesh="cup_inner", pos=f"0 0 {z + L.CUP_HEIGHT * 0.5:.5f}",
              quat="0 1 0 0", material=inner, contype=0, conaffinity=0, group=1)
        E(body, "geom", name=f"{name}_lip{i}", type="mesh", mesh="cup_rim", pos=f"0 0 {z + L.CUP_HEIGHT * 0.5:.5f}",
          quat="0 1 0 0", material="cup_white" if not clear else "cup_clear", contype=0, conaffinity=0, group=1)
        E(body, "geom", name=f"{name}_band{i}", type="mesh", mesh="band_frustum", pos=f"0 0 {z:.5f}", mass=0.001, **col)
        E(body, "geom", name=f"{name}_full{i}", type="mesh", mesh="full_frustum", pos=f"0 0 {z:.5f}", mass=0.001, **col)
    return body


def _material(asset: ET.Element, name: str, **attrs) -> None:
    E(asset, "material", name=name, **attrs)


def build_xml() -> str:
    _ensure_assets()
    root = ET.Element("mujoco", {"model": "watercup_station"})
    E(root, "compiler", angle="radian", meshdir=ASSET_DIR, autolimits="true")
    E(root, "option", timestep=0.001, gravity="0 0 -9.81", integrator="implicitfast", cone="elliptic",
      impratio=8, solver="Newton", iterations=40, tolerance=1e-9, noslip_iterations=3, noslip_tolerance=1e-7)
    E(root, "size", memory="64M")
    visual = E(root, "visual")
    E(visual, "global", offwidth=1920, offheight=1200)
    E(visual, "quality", shadowsize=3072, offsamples=4)
    E(visual, "headlight", ambient="0.36 0.36 0.37", diffuse="0.42 0.42 0.42", specular="0.18 0.18 0.18")
    E(visual, "rgba", haze="0.92 0.91 0.89 1")
    E(visual, "map", znear=0.01, zfar=12, shadowclip=2.0, shadowscale=0.6)

    asset = E(root, "asset")
    E(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.90 0.91 0.92", rgb2="0.68 0.71 0.75",
      width=256, height=256)
    for tex, fn in (("wood", "wood.png"), ("coaster_tex", "coaster.png"), ("mat_tex", "mat.png"), ("ripple", "ripple.png")):
        E(asset, "texture", name=tex, type="2d", file=os.path.join(ASSET_DIR, fn))
    _material(asset, "wood", texture="wood", specular=0.15, shininess=0.15, reflectance=0.03)
    _material(asset, "coaster_wood", texture="coaster_tex", specular=0.28, shininess=0.28, reflectance=0.04)
    _material(asset, "mat", texture="mat_tex", specular=0.06, shininess=0.05, reflectance=0.01)
    _material(asset, "ripple", texture="ripple", specular=0.12, shininess=0.1)
    _material(asset, "acrylic", rgba="0.84 0.92 0.96 0.30", specular=0.96, shininess=0.82, reflectance=0.22)
    _material(asset, "chrome", rgba="0.80 0.83 0.86 1", specular=1.0, shininess=0.96, reflectance=0.55)
    _material(asset, "chrome_dark", rgba="0.46 0.49 0.53 1", specular=0.95, shininess=0.9, reflectance=0.4)
    _material(asset, "plastic_white", rgba="0.95 0.95 0.93 1", specular=0.35, shininess=0.28)
    _material(asset, "cup_white", rgba="0.96 0.95 0.92 1", specular=0.25, shininess=0.2)
    _material(asset, "cup_cream", rgba="0.96 0.93 0.85 1", specular=0.15, shininess=0.1)
    _material(asset, "cup_clear", rgba="0.86 0.93 0.96 0.22", specular=0.95, shininess=0.82, reflectance=0.2)
    _material(asset, "shell_gray", rgba="0.60 0.62 0.65 1", specular=0.35, shininess=0.28)
    _material(asset, "shell_white", rgba="0.95 0.95 0.93 1", specular=0.28, shininess=0.2)
    _material(asset, "arm_black", rgba="0.10 0.10 0.11 1", specular=0.45, shininess=0.35)
    _material(asset, "mark_white", rgba="0.96 0.96 0.95 1", specular=0.2, shininess=0.15)
    _material(asset, "aluminum", rgba="0.78 0.80 0.82 1", specular=0.7, shininess=0.45, reflectance=0.15)
    _material(asset, "wall", rgba="0.93 0.92 0.90 1", specular=0.02, shininess=0.02)
    _material(asset, "water", rgba="0.52 0.74 0.90 0.55", specular=0.6, shininess=0.7)
    for name in ("tank_upper", "tank_lower", "tank_solid", "cup_wall", "cup_inner", "cup_rim", "cup_floor", "band_frustum", "full_frustum"):
        shell = name in ("tank_upper", "tank_lower", "cup_wall", "cup_inner", "cup_floor")
        E(asset, "mesh", name=name, file=f"{name}.obj", inertia="shell" if shell else None)

    world = E(root, "worldbody")
    E(world, "light", name="key", pos="0.25 -0.75 1.7", dir="0.05 0.35 -1", diffuse="0.88 0.85 0.80", specular="0.35 0.35 0.33",
      castshadow="true")
    E(world, "light", name="fill", pos="1.0 0.75 1.3", dir="-0.5 -0.4 -1", diffuse="0.40 0.43 0.48", castshadow="false")
    E(world, "light", name="rim", pos="-0.75 0.25 1.0", dir="0.7 -0.1 -0.4", diffuse="0.26 0.24 0.22", castshadow="false")
    E(world, "geom", name="floor", type="plane", pos="0.4 0 -0.78", size="4 4 0.1", rgba="0.60 0.58 0.55 1", contype=0, conaffinity=0)
    E(world, "geom", name="back_wall", type="box", pos="1.4 0 0.4", size="0.02 1.6 1.0", material="wall", contype=0, conaffinity=0)

    # table, mat, rail
    env = dict(contype=ENV_T, conaffinity=ENV_A)
    E(world, "geom", name="tabletop", type="box", pos="0.48 -0.05 -0.038", size="0.62 0.64 0.038", material="wood",
      friction="1.1 0.02 0.001", **env)
    E(world, "geom", name="cutting_mat", type="box", pos=[L.MAT_CENTER_XY[0], L.MAT_CENTER_XY[1], 0.0015],
      size=[L.MAT_HALF_SIZE[0], L.MAT_HALF_SIZE[1], 0.0015], material="mat", friction=f"{L.MAT_FRICTION} 0.02 0.001",
      solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP, **env)
    rail = E(world, "body", name="rail", pos=f"-0.012 {0.5 * (L.CUP_MOUNT[1] + L.TAP_MOUNT[1]):.4f} 0.0")
    half_rail = 0.5 * abs(L.CUP_MOUNT[1] - L.TAP_MOUNT[1]) + 0.16
    E(rail, "geom", type="box", pos="0 0 0.010", size=[0.020, half_rail, 0.010], material="aluminum", contype=0, conaffinity=0)
    E(rail, "geom", type="box", pos="0 0 0.0205", size=[0.012, half_rail, 0.0008], rgba="0.2 0.21 0.23 1", contype=0, conaffinity=0)
    E(rail, "geom", type="box", pos="-0.01 0 0.040", size=[0.044, 0.030, 0.012], rgba="0.10 0.12 0.14 1", contype=0, conaffinity=0)
    E(rail, "geom", type="cylinder", pos="-0.01 0 0.053", size="0.016 0.002", rgba="0.22 0.24 0.26 1", contype=0, conaffinity=0)
    for cy in (-half_rail + 0.02, half_rail - 0.02):
        E(rail, "geom", type="box", pos=f"0.012 {cy} 0.018", size="0.030 0.016 0.018", rgba="0.92 0.36 0.10 1", contype=0, conaffinity=0)

    equality = E(root, "equality")
    contact = E(root, "contact")
    actuator = E(root, "actuator")

    for arm, prefix in (("cup", "cup_"), ("tap", "tap_")):
        assets_, bodies_, eqs_, contacts_ = _arm_fragments(prefix)
        for node in assets_:
            asset.append(node)
        mount_p, yaw = L.MOUNTS[arm]
        mount = E(world, "body", name=f"{prefix}mount", pos=mount_p, quat=yaw_quat(yaw))
        for node in bodies_:
            mount.append(node)
        for node in eqs_:
            equality.append(node)
        for node in contacts_:
            contact.append(node)
        for j in range(1, 7):
            E(actuator, "position", name=f"{prefix}a{j}", joint=f"{prefix}joint{j}", kp=L.KP_ARM[j - 1], kv=L.KD_ARM[j - 1],
              forcelimited="true", forcerange=[-L.ARM_TORQUE_LIMIT[j - 1], L.ARM_TORQUE_LIMIT[j - 1]])
        for j in (7, 8):
            E(actuator, "position", name=f"{prefix}g{j}", joint=f"{prefix}joint{j}", kp=L.FINGER_KP, kv=L.FINGER_KD,
              forcelimited="true", forcerange=[-L.FORCE_CAP_PAPER_CUP_N, L.FORCE_CAP_PAPER_CUP_N])

    _dispenser(world)
    _coaster(world)

    # cups and stacks
    for kind, wall, inner, mass, xy in (
        ("paper", "ripple", "cup_cream", L.PAPER_MASS, L.PARKED_PAPER_STACK_XY),
        ("plastic", "cup_clear", "cup_clear", L.PLASTIC_MASS, L.PARKED_PLASTIC_STACK_XY),
    ):
        world.append(_rest_body(f"{kind}_rest", wall, inner, xy))
        world.append(_cup_body(f"{kind}_cup", wall, inner, mass))
    return ET.tostring(root, encoding="unicode")


def _dispenser(world: ET.Element) -> None:
    disp = E(world, "body", name="dispenser", pos=[L.DISPENSER_XY[0], L.DISPENSER_XY[1], 0.0])
    E(disp, "inertial", pos="0 0 0.15", mass=0.55, diaginertia="0.004 0.004 0.003")
    E(disp, "joint", name="disp_x", type="slide", axis="1 0 0", range="-0.06 0.06", damping=40, frictionloss=L.DISPENSER_SLIDE_FRICTION_N)
    E(disp, "joint", name="disp_y", type="slide", axis="0 1 0", range="-0.06 0.06", damping=40, frictionloss=L.DISPENSER_SLIDE_FRICTION_N)
    env = dict(contype=ENV_T, conaffinity=ENV_A, solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP)
    E(disp, "geom", name="tank_upper", type="mesh", mesh="tank_upper", material="acrylic", contype=0, conaffinity=0, group=1)
    E(disp, "geom", name="tank_lower", type="mesh", mesh="tank_lower", material="acrylic", contype=0, conaffinity=0, group=1)
    E(disp, "geom", name="tank_collar", type="cylinder", pos="0 0 0.095", size="0.080 0.004", material="acrylic", contype=0, conaffinity=0, group=1)
    E(disp, "geom", name="tank_top_rim", type="cylinder", pos="0 0 0.378", size="0.086 0.003", material="acrylic", contype=0, conaffinity=0, group=1)
    E(disp, "geom", name="tank_hit", type="mesh", mesh="tank_solid", friction="0.6 0.01 0.001", group=3, rgba="0.2 0.6 0.9 0.12", **env)
    for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        E(world, "geom", name=f"tape_{sx}_{sy}", type="box",
          pos=[L.DISPENSER_XY[0] + 0.075 * sx, L.DISPENSER_XY[1] + 0.075 * sy, L.MAT_SURFACE_Z + 0.0010],
          size="0.016 0.011 0.001", material="plastic_white", contype=0, conaffinity=0, group=1)

    faucet = E(disp, "body", name="faucet", pos=L.FAUCET_POS)
    E(faucet, "geom", name="gasket_nut", type="cylinder", pos="0.010 0 0", size="0.017 0.008", quat="0.707107 0 0.707107 0",
      material="plastic_white", contype=0, conaffinity=0, group=1)
    E(faucet, "geom", name="chrome_flange", type="cylinder", pos="0.001 0 0", size="0.0135 0.0035", quat="0.707107 0 0.707107 0",
      material="chrome", contype=0, conaffinity=0, group=1)
    vx = L.VALVE_LOCAL_X
    E(faucet, "geom", name="valve_barrel", type="capsule", fromto=[-0.003, 0, 0, vx, 0, 0], size=0.0105, material="chrome",
      friction="0.5 0.01 0.001", **env, group=1)
    top_z = L.LEVER_PIVOT_LOCAL[2] - 0.002
    bot_z = L.MOUTH_LOCAL[2] + 0.008
    E(faucet, "geom", name="valve_body", type="cylinder", pos=[vx, 0, 0.5 * (top_z + bot_z)], size=[0.0105, 0.5 * (top_z - bot_z)],
      material="chrome", friction="0.5 0.01 0.001", **env, group=1)
    E(faucet, "geom", name="spout_tip", type="cylinder", pos=[vx, 0, L.MOUTH_LOCAL[2] + 0.004], size=[0.0068, 0.004],
      material="chrome", friction="0.5 0.01 0.001", **env, group=1)
    E(faucet, "geom", name="spout_mouth", type="cylinder", pos=[vx, 0, L.MOUTH_LOCAL[2] + 0.0005], size=[0.0048, 0.0006],
      rgba="0.12 0.13 0.14 1", contype=0, conaffinity=0, group=1)

    turret = E(faucet, "body", name="lever_turret", pos=L.LEVER_PIVOT_LOCAL)
    E(turret, "joint", name="lever_yaw", type="hinge", axis="0 0 1", range=[L.YAW_MIN, L.YAW_MAX], damping=0.02,
      frictionloss=L.LEVER_YAW_FRICTION, armature=2e-5, solreflimit="0.004 1")
    E(turret, "geom", name="lever_cap", type="cylinder", pos="0 0 -0.001", size=[0.0125, 0.0025], material="chrome_dark",
      mass=0.004, contype=0, conaffinity=0, group=1)
    lever = E(turret, "body", name="lever", pos="0 0 0")
    E(lever, "joint", name="lever", type="hinge", axis="0 -1 0", range=[L.LEVER_MIN, L.LEVER_MAX], damping=0.02,
      frictionloss=L.LEVER_LIFT_FRICTION, armature=2e-5, solreflimit="0.004 1")
    hc = dict(contype=ENV_T, conaffinity=ENV_A, friction="1.0 0.02 0.001", solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP)
    t = L.LEVER_HANDLE_THICK
    w = L.LEVER_HANDLE_WIDTH
    E(lever, "geom", name="lever_stem", type="box", pos=[0.008, 0, 0.0015 + 0.5 * t], size=[0.010, 0.5 * 0.011, 0.5 * t],
      material="chrome", mass=0.004, **hc)
    E(lever, "geom", name="lever_paddle", type="box", pos=[0.026, 0, 0.0015 + 0.5 * t], size=[0.5 * (L.LEVER_HANDLE_LEN - 0.008), 0.5 * w, 0.5 * t],
      material="chrome", mass=0.006, **hc)
    E(lever, "geom", name="lever_boss", type="cylinder", pos=[0, 0, 0.0015 + 0.5 * t], size=[0.0085, 0.5 * t],
      material="chrome", mass=0.002, contype=0, conaffinity=0, group=1)
    E(lever, "site", name="lever_site", pos=[L.LEVER_GRIP_DIST, 0, 0.0015 + 0.5 * t], size=0.004, rgba="0 0 0 0")

    stream_top = L.MOUTH_OUTLET_Z - 0.002
    E(world, "geom", name="stream", type="cylinder",
      pos=[L.MOUTH_WORLD[0], L.MOUTH_WORLD[1], 0.5 * (stream_top + L.MAT_SURFACE_Z + 0.02)],
      size=[0.0038, 0.5 * (stream_top - (L.MAT_SURFACE_Z + 0.02))], material="water", rgba="0.52 0.74 0.90 0.0",
      contype=0, conaffinity=0, group=1)


def _coaster(world: ET.Element) -> None:
    body = E(world, "body", name="coaster", pos=[L.COASTER_XY[0], L.COASTER_XY[1], L.MAT_SURFACE_Z])
    h_slab = L.COASTER_FLOOR_Z - L.MAT_SURFACE_Z
    env = dict(contype=ENV_T, conaffinity=ENV_A, friction=f"{L.CUP_FRICTION} 0.01 0.0005", solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP)
    E(body, "geom", name="coaster_slab", type="cylinder", pos=[0, 0, 0.5 * h_slab], size=[L.COASTER_OUTER_RADIUS, 0.5 * h_slab],
      material="coaster_wood", **env)
    # funnel ring: boxes whose inner faces run from (r_bottom, floor) up to (r_top, rim)
    r0, r1 = L.COASTER_FUNNEL_R_BOTTOM, L.COASTER_FUNNEL_R_TOP
    z0, z1 = L.COASTER_FLOOR_Z - L.MAT_SURFACE_Z, L.COASTER_RIM_Z - L.MAT_SURFACE_Z
    thick = 0.006
    alpha = np.arctan2(r1 - r0, z1 - z0)
    length = float(np.hypot(r1 - r0, z1 - z0))
    nr, nz = np.cos(alpha), -np.sin(alpha)
    tr, tz = np.sin(alpha), np.cos(alpha)
    n = 18
    # inner (cup-facing) surface normal points toward the axis; build the box on the outer side of that surface
    c_r = 0.5 * (r0 + r1) + 0.5 * thick * nr
    c_z = 0.5 * (z0 + z1) + 0.5 * thick * nz
    half_w = c_r * np.tan(np.pi / n) + 0.0008
    for i in range(n):
        phi = 2 * np.pi * i / n
        rh = np.array([np.cos(phi), np.sin(phi), 0.0])
        ph = np.array([-np.sin(phi), np.cos(phi), 0.0])
        zh = np.array([0.0, 0.0, 1.0])
        R = rot_axes(nr * rh + nz * zh, ph, tr * rh + tz * zh)
        E(body, "geom", name=f"coaster_wall{i}", type="box", pos=c_r * rh + c_z * zh, quat=mat_to_quat(R),
          size=[0.5 * thick, half_w, 0.5 * length], material="coaster_wood", **env)
    # visual details: rounded square top plate, tab, marker holes and grooves
    E(body, "geom", name="coaster_tab", type="box", pos=[-0.028, 0.028, 0.5 * h_slab], size=[0.028, 0.028, 0.5 * h_slab],
      material="coaster_wood", contype=0, conaffinity=0, group=1)
    for hx, hy in ((-0.044, 0.018), (-0.018, 0.044)):
        E(body, "geom", type="cylinder", pos=[hx, hy, h_slab + 0.0003], size="0.0065 0.0004", rgba="0.22 0.14 0.08 1",
          contype=0, conaffinity=0, group=1)
    for radius, shade in ((0.048, "0.42 0.25 0.12 1"), (0.0290, "0.50 0.30 0.15 1")):
        E(body, "geom", type="cylinder", pos=[0, 0, h_slab + 0.0002], size=[radius, 0.0003], rgba=shade, contype=0, conaffinity=0, group=1)
