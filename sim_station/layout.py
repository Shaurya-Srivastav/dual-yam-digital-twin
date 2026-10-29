"""Single source of truth for station geometry, calibration, physics parameters,
real Raspberry Pi taught waypoints, and level configurations.

Everything that is a *measured number* lives here. The scene (`scene.py`), the
plant (`plant.py`), the controller/skills and the evaluation suite all read from
this module so there is exactly one place to correct a calibration.

Frames
------
World: +x forward from the T-slot rail across the mat, +y to the left, +z up,
mat surface at z = MAT_SURFACE_Z, table top at z = 0.
Both arm bases sit on the rail at x = 0.06, yaw 0. All "cup frame" numbers are
the Pi's forward-kinematics outputs in the cup arm's base frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 1. Bench and arm base calibration
# ---------------------------------------------------------------------------

MAT_SURFACE_Z = 0.003
MAT_CENTER_XY = np.array([0.460, -0.0533])
MAT_HALF_SIZE = np.array([0.3048, 0.4572, 0.0015])        # 24 in (x) by 36 in (y), 3 mm

CUP_MOUNT = np.array([0.0600, 0.2500, 0.0000])   # YAM base plate sits on the table plane (base mesh min z = 0)
CUP_YAW = 0.0

# Tap-arm base. The station photos (overhead of the rail, both wrists on one extrusion,
# 24x36 inch gridded mat) show the bases closer together than the official 0.61 m
# i2rt dual-arm jig. The mount that reaches a stack standing in the coaster, with no
# relative yaw, is the stack-pin solution: +67 mm in x and -480 mm in y in the cup
# frame. The rim-grab subtraction (-0.1 mm, -606.6 mm) is that other jig; its lever
# arc reaches, but the tap hand cannot pin a stack at the coaster.
TAP_BASE_OFFSET_IN_CUP_FRAME = np.array([0.067, -0.480])
TAP_MOUNT = np.array(
    [CUP_MOUNT[0] + TAP_BASE_OFFSET_IN_CUP_FRAME[0], CUP_MOUNT[1] + TAP_BASE_OFFSET_IN_CUP_FRAME[1], CUP_MOUNT[2]]
)
TAP_YAW = 0.0

MOUNTS = {"cup": (CUP_MOUNT, CUP_YAW), "tap": (TAP_MOUNT, TAP_YAW)}


def solve_tap_base_candidates() -> Dict[str, Tuple[np.ndarray, float]]:
    """Mounts that were on the table. The running one is `stack_reach`.

    The pin pose was a retreat (the tap arm backed 14.6 cm off the stack), so it does
    not locate the base by itself. The overhead photo is what picks this candidate:
    both bases share the rail, and the gap is visibly under the 0.61 m jig.
    """
    p_cup_xy = np.array([0.4872, 0.0077])
    p_tap_xy = np.array([0.4199, 0.4878])
    d_parallel = p_cup_xy - p_tap_xy
    parallel = np.array([CUP_MOUNT[0] + d_parallel[0], CUP_MOUNT[1] + d_parallel[1], CUP_MOUNT[2]])
    r = float(np.hypot(*p_tap_xy))
    phi = float(np.arctan2(p_tap_xy[1], p_tap_xy[0]))
    psi = float(np.arccos(p_cup_xy[0] / r) - phi)
    c, s = np.cos(psi), np.sin(psi)
    dy = float(p_cup_xy[1] - (s * p_tap_xy[0] + c * p_tap_xy[1]))
    same_rail = np.array([CUP_MOUNT[0], CUP_MOUNT[1] + dy, CUP_MOUNT[2]])
    rim = np.array([CUP_MOUNT[0] - 0.0001, CUP_MOUNT[1] - 0.6066, CUP_MOUNT[2]])
    return {"parallel": (parallel, 0.0), "same_rail": (same_rail, psi),
            "rim_grab_0.61m": (rim, 0.0), "stack_reach": (TAP_MOUNT.copy(), 0.0)}


def _rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def local_to_world(arm: str, local_xyz) -> np.ndarray:
    mount, yaw = MOUNTS[arm]
    return mount + _rotz(yaw) @ np.asarray(local_xyz, dtype=float)


def world_to_local(arm: str, world_xyz) -> np.ndarray:
    mount, yaw = MOUNTS[arm]
    return _rotz(yaw).T @ (np.asarray(world_xyz, dtype=float) - mount)


def cup_local_to_world(local_xyz) -> np.ndarray:
    return local_to_world("cup", local_xyz)


def tap_local_to_world(local_xyz) -> np.ndarray:
    return local_to_world("tap", local_xyz)


# ---------------------------------------------------------------------------
# 2. Cup, coaster, stacks, spigot
# ---------------------------------------------------------------------------

CUP_HEIGHT = 0.110
CUP_RIM_RADIUS = 0.038          # 76 mm rim
CUP_BASE_RADIUS = 0.027         # 54 mm base
CUP_WALL_THICK = 0.0018
CUP_STAVE_COUNT = 16
CUP_TAPER = (CUP_RIM_RADIUS - CUP_BASE_RADIUS) / CUP_HEIGHT     # radius gained per metre of height
PAPER_MASS = 0.012
PLASTIC_MASS = 0.008
NEST_STEP = 0.0125                                              # exposed rim step per nested cup
STACK_BAND_RADIUS = 0.0355                                      # body under the rolled bead: what the pin pads pinch
NEST_DEFAULT = 8
NEST_MAX = 11                                                    # tallest stack the sim builds
STACK_BREAK_FORCE_N = 12.0                                       # nested cups stick; straight pull needs ~12 N
STACK_FATIGUE_LEN_M = 0.010                                      # path length of relative wiggle that halves... e-folds the bond

CUP_HOLD_WIDTH_ON_AXIS = 0.0674                # jaw gap when a side grasp sits on the axis at hold height (calibration cup)
CUP_RADIUS = 0.5 * (CUP_RIM_RADIUS + CUP_BASE_RADIUS)           # legacy name: mid radius
CUP_RIM = CUP_RIM_RADIUS
CUP_BASE = CUP_BASE_RADIUS


def cup_radius_at(height_above_base: float) -> float:
    return CUP_BASE_RADIUS + CUP_TAPER * float(height_above_base)


# Pi forward kinematics in the cup arm's base frame (measured)
L1_COASTER_LOCAL_XY = np.array([0.5047, 0.0051])
SPOUT_LOCAL_XY = np.array([0.4680, -0.2831])
PAPER_STACK_LOCAL_XY = np.array([0.4872, 0.0077])
PLASTIC_STACK_LOCAL_XY = np.array([0.4872, -0.0673])
SIDE_PLACE_LOCAL_XY = np.array([0.4648, -0.2164])

COASTER_XY = CUP_MOUNT[:2] + L1_COASTER_LOCAL_XY
# Levels 1–4 start where the coaster is. The nested stack stands on that spot; the
# wooden dish itself is set aside for the pour (one dish cannot hold the remaining
# stack and the served cup). Level 5 is the only level whose cup starts somewhere else.
PAPER_STACK_XY = COASTER_XY.copy()
PLASTIC_STACK_XY = COASTER_XY.copy()
SIDE_PLACE_XY = CUP_MOUNT[:2] + SIDE_PLACE_LOCAL_XY
L2_COASTER_XY = np.array([0.400, -0.080])
LEVEL1_CUP_XY = COASTER_XY.copy()
PARKED_PAPER_STACK_XY = np.array([0.6400, 0.3550])
PARKED_PLASTIC_STACK_XY = np.array([0.7000, 0.2800])

COASTER_OUTER_RADIUS = 0.056
COASTER_FLOOR_Z = 0.0100                    # recess floor; the real taught hold TCP z=0.0642 = 0.010 + 0.055 - 0.001
COASTER_RIM_Z = 0.0150
COASTER_FUNNEL_R_BOTTOM = 0.0470            # recess floor radius: the real L1 grasp sits 2.0 cm off the placed axis, so the dish has >= 2 cm of slack
COASTER_FUNNEL_R_TOP = 0.0530

# Dispenser. The tank is centred 13.6 cm behind the valve axis along +x.
MOUTH_WORLD_XY = CUP_MOUNT[:2] + SPOUT_LOCAL_XY
DISPENSER_XY = np.array([MOUTH_WORLD_XY[0] + 0.1360, MOUTH_WORLD_XY[1]])
TANK_CIRCUM_RADIUS = 0.090                  # octagon circumradius
TANK_APOTHEM = TANK_CIRCUM_RADIUS * np.cos(np.pi / 8.0)
FAUCET_POS = np.array([-0.0820, 0.0, 0.1500])
VALVE_LOCAL_X = -0.0540
MOUTH_OUTLET_Z = 0.1290
LEVER_PIVOT_Z = 0.1750
MOUTH_LOCAL = np.array([VALVE_LOCAL_X, 0.0, MOUTH_OUTLET_Z - FAUCET_POS[2]])
LEVER_PIVOT_LOCAL = np.array([VALVE_LOCAL_X, 0.0, LEVER_PIVOT_Z - FAUCET_POS[2]])
LEVER_HANDLE_LEN = 0.0440
LEVER_GRIP_DIST = 0.0300                    # grip point distance from the pivot along the handle
LEVER_HANDLE_THICK = 0.006
LEVER_HANDLE_WIDTH = 0.015

MOUTH_WORLD = np.array([MOUTH_WORLD_XY[0], MOUTH_WORLD_XY[1], MOUTH_OUTLET_Z])
LEVER_PIVOT_WORLD = np.array([MOUTH_WORLD_XY[0], MOUTH_WORLD_XY[1], LEVER_PIVOT_Z])
LEVER_WORLD = LEVER_PIVOT_WORLD + np.array([LEVER_GRIP_DIST, 0.0, 0.0])

LEVER_YAW_DEFAULT = np.radians(45.0)         # where the handle sits at the start of Levels 1, 2, 3 and 5
LEVER_CLOSED = 0.0
LEVER_OPEN = 1.15
LEVER_FLOW_THRESHOLD = 0.45
LEVER_MIN = 0.0
LEVER_MAX = 1.25
YAW_MIN = -2.0 * np.pi
YAW_MAX = 2.0 * np.pi              # the swivel turns freely through a full circle
LEVER_YAW_FRICTION = 0.05                   # N m
LEVER_LIFT_FRICTION = 0.07                  # N m
DISPENSER_SLIDE_FRICTION_N = 9.0            # taped to the mat: slides only if pulled harder than this

# Carry geometry: a held cup's centre sits at TCP height; its base floats 8 mm above the mat at the spout.
CUP_GRASP_HEIGHT_ABOVE_BASE = 0.054         # real taught hold: TCP at mid cup
SPOUT_BASE_CLEARANCE = 0.008
UNDER_SPOUT = np.array(
    [MOUTH_WORLD[0], MOUTH_WORLD[1], MAT_SURFACE_Z + SPOUT_BASE_CLEARANCE + CUP_GRASP_HEIGHT_ABOVE_BASE]
)

# ---------------------------------------------------------------------------
# 3. Gripper and controller model (real i2rt numbers)
# ---------------------------------------------------------------------------

KP_ARM = np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
KD_ARM = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
GRAV_COMP_FACTOR_REAL = np.array([1.0, 1.1, 1.1, 1.2, 1.0, 1.0])   # yam_v1.yml; covers real cables/payload the model lacks
# The sim model has no unmodelled load, so the over-compensation would just lift every pose by 1-2 cm.
GRAV_COMP_FACTOR = np.ones(6)
COULOMB_FRICTION = np.array([0.3, 0.3, 0.3, 0.06, 0.06, 0.06])
# DM4340 (joints 1-3) peak 27 N m; DM4310 (joints 4-6) peak 10 N m. The vendor MJCF caps every joint at 10 N m,
# which cannot hold the arm at full reach (about 11 N m of gravity at joint 2), and the real arm does hold it.
ARM_TORQUE_LIMIT = np.array([27.0, 27.0, 27.0, 10.0, 10.0, 10.0])
JOINT_LOW = np.array([-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944])
JOINT_HIGH = np.array([3.14159, 3.66519, 3.14159, 1.72, 1.5708, 2.0944])   # J4 reaches 1.67 in the Pi log, beyond the vendor 1.5708

# Sim joint dynamics (rotor inertia reflected through the gearbox, viscous + Coulomb friction)
JOINT_ARMATURE = np.array([0.045, 0.045, 0.045, 0.0011, 0.0011, 0.0011])
JOINT_DAMPING = np.array([0.30, 0.30, 0.30, 0.05, 0.03, 0.03])
JOINT_FRICTIONLOSS = 0.5 * COULOMB_FRICTION

GRIPPER_MAX_OPEN_M = 0.0776                 # calibrated can0 span 5.314 rad -> 77.6 mm
FINGER_MAX_M = 0.5 * GRIPPER_MAX_OPEN_M
GRIPPER_MOTOR_STROKE_RAD = 6.57
GRIPPER_STROKE_M = 0.096
GRIPPER_NM_PER_N = GRIPPER_STROKE_M / GRIPPER_MOTOR_STROKE_RAD       # 0.0146 N m per N of jaw force
GRIPPER_KP_MOTOR = 20.0
GRIPPER_KD_MOTOR = 0.5
GRIPPER_RATE_M_S = 0.12                     # jaw gap speed (open/close)
FINGER_ARMATURE = 0.02
FINGER_DAMPING = 3.0
FINGER_FRICTIONLOSS = 0.3
# Servo gains in jaw-force units. The physical value is kp*2/0.0146^2 = 1.9e5 N/m, which a 1 kHz
# explicit loop cannot hold; this is the stiffest conditioned stand-in. Force saturates at the cap
# after 0.5 mm of overdrive, as the real force limiter does.
FINGER_KP = 40000.0
FINGER_KD = 120.0

FORCE_CAP_STACK_TOP_N = 15.0
FORCE_CAP_STACK_PIN_N = 18.0
FORCE_CAP_PAPER_CUP_N = 9.0
FORCE_CAP_PLASTIC_CUP_N = 5.0
FORCE_CAP_RIM_AND_LEVER_N = 14.0
FORCE_CAP_PROBE_N = 3.0

PAD_FRICTION = 1.3
CUP_FRICTION = 1.1
MAT_FRICTION = 1.2

# ---------------------------------------------------------------------------
# 4. Motion limits
# ---------------------------------------------------------------------------

TRANSIT_SPEED_M_S = 0.25
CARRY_SPEED_M_S = 0.15
CARRY_ACCEL_M_S2 = 0.80
APPROACH_SPEED_M_S = 0.03
PECK_SPEED_M_S = 0.05
SWEEP_SPEED_M_S = 0.025
CONTACT_PAUSE_S = 0.18
MAX_JOINT_VEL = np.array([1.6, 1.4, 1.6, 2.0, 2.2, 2.6])
MAX_JOINT_ACC = np.array([5.0, 4.5, 5.0, 8.0, 9.0, 10.0])
CONTROL_DT = 0.002

REST_Q = np.array([0.0, 0.094, 0.098, 0.021, -0.054, 0.0])      # the Pi's rest pose: arm folded flat
READY_Q = {
    "cup": np.array([0.05, 1.60, 1.10, 0.70, 0.0, 0.0]),
    "tap": np.array([0.05, 1.60, 1.10, 0.70, 0.0, 0.0]),
}
TABLE_LOOK = np.array([0.46, -0.04, 0.12])

# ---------------------------------------------------------------------------
# 5. Levels
# ---------------------------------------------------------------------------


@dataclass
class LevelLayout:
    level: int
    coaster_xy: np.ndarray
    cup_xy: np.ndarray
    paper_stack_xy: np.ndarray
    plastic_stack_xy: np.ndarray
    cup_in_coaster: bool = False
    inverted_stack: bool = False
    stack_height_known: bool = True
    lever_yaw_known: bool = True
    cup_location_known: bool = True
    description: str = ""


def level_layout(level: int, material: str = "paper") -> LevelLayout:
    plastic = material == "plastic"
    if level == 1:
        return LevelLayout(1, COASTER_XY.copy(), COASTER_XY.copy(), PARKED_PAPER_STACK_XY.copy(),
                           PARKED_PLASTIC_STACK_XY.copy(), cup_in_coaster=True, description="Cup upright in the coaster")
    if level in (2, 3):
        paper = PAPER_STACK_XY.copy() if not plastic else PARKED_PAPER_STACK_XY.copy()
        plas = PLASTIC_STACK_XY.copy() if plastic else PARKED_PLASTIC_STACK_XY.copy()
        stack = plas if plastic else paper
        return LevelLayout(level, L2_COASTER_XY.copy(), stack.copy(), paper, plas, inverted_stack=True,
                           stack_height_known=(level == 2), description="Inverted nested stack")
    if level == 4:
        return LevelLayout(4, COASTER_XY.copy(), COASTER_XY.copy(), PARKED_PAPER_STACK_XY.copy(),
                           PARKED_PLASTIC_STACK_XY.copy(), cup_in_coaster=True, lever_yaw_known=False,
                           description="Handle swivel angle unknown")
    if level == 5:
        return LevelLayout(5, COASTER_XY.copy(), np.array([0.45, 0.10]), PARKED_PAPER_STACK_XY.copy(),
                           PARKED_PLASTIC_STACK_XY.copy(), cup_location_known=False,
                           description="Cup anywhere on the mat")
    raise ValueError(f"Level must be 1..5, got {level}")


# Free-space on the mat where Level 5 may drop the cup (clear of the dispenser, stacks, coaster, rail, edges)
L5_REGION_X = (0.24, 0.50)
L5_REGION_Y = (-0.20, 0.34)
L5_KEEPOUT = [
    (COASTER_XY, 0.11),
    (PARKED_PAPER_STACK_XY, 0.10),
    (PARKED_PLASTIC_STACK_XY, 0.10),
    (MOUTH_WORLD_XY, 0.12),
    # folded arms at REST_Q with the tap base at the stack-reach mount (tcp about (0.38, -0.22) and (0.32, 0.26))
    (np.array([0.27, -0.19]), 0.18),
    (np.array([0.38, -0.16]), 0.11),
    (np.array([0.36, -0.08]), 0.10),
    (np.array([0.20, 0.30]), 0.19),
    (np.array([0.32, 0.24]), 0.11),
]

# ---------------------------------------------------------------------------
# 6. Real taught waypoints from the Pi log (joint space, used ONLY for replay / equivalence tests)
# ---------------------------------------------------------------------------

REAL_L1_CUP_WAYPOINTS: Dict[str, np.ndarray] = {
    "approach": np.array([0.0250, 2.1666, 0.4881, 1.6703, 0.0265, 0.0147, 0.9971]),
    "grasp": np.array([0.0700, 2.2765, 0.7200, 1.5532, 0.2810, 0.0410, 0.9971]),
    "paper_hold": np.array([0.0647, 2.2448, 0.7040, 1.4910, 0.2245, 0.0223, 0.7797]),
    "travel_clear": np.array([0.0841, 2.1395, 0.6765, 1.4822, 0.2577, 0.0257, 0.8421]),
    "lower_for_spout": np.array([-0.1658, 2.2715, 0.7132, 1.5333, 0.2092, 0.0326, 0.7162]),
    "under_spout": np.array([-0.4789, 2.4008, 0.9154, 1.5379, 0.1986, 0.0296, 0.7317]),
    "spout_exit": np.array([-0.2195, 2.3447, 0.8394, 1.5005, 0.0700, 0.0063, 0.7411]),
    "lift_out": np.array([-0.2359, 2.0514, 0.6083, 1.5002, 0.0776, 0.0036, 0.7558]),
    "above_coaster": np.array([0.0391, 2.2391, 0.7303, 1.5669, 0.0788, 0.0826, 0.8417]),
    "in_coaster": np.array([0.0383, 2.3436, 0.7185, 1.6463, 0.0788, 0.0826, 0.8417]),
    "release": np.array([0.0185, 2.3428, 0.7189, 1.6489, -0.0158, -0.0536, 0.9986]),
    "rest": np.array([-0.1848, 0.0940, 0.0978, 0.0212, -0.0536, -0.0017, 0.9973]),
}

REAL_L2_CUP_WAYPOINTS: Dict[str, np.ndarray] = {
    "stack_approach": np.concatenate([np.deg2rad([-2.3, 100.3, 25.8, 83.4, -9.2, -1.3]), [0.997]]),
    "stack_grasp": np.concatenate([np.deg2rad([4.0, 109.8, 36.5, 87.8, 9.0, 4.3]), [0.996]]),
    "stack_hold": np.concatenate([np.deg2rad([2.9, 113.4, 37.3, 90.9, 5.4, 0.4]), [0.685]]),
    "stack_lift": np.concatenate([np.deg2rad([4.7, 85.2, 64.2, 31.3, 8.5, -1.5]), [0.681]]),
    "stack_carry": np.array([-0.1787, 1.3445, 0.4927, 0.9535, 0.3023, -1.4960, 0.6867]),
    "side_place": np.array([-0.4660, 2.4424, 0.7780, 1.6573, -0.0868, -1.4960, 0.6855]),
    "side_release": np.array([-0.4660, 2.4424, 0.7780, 1.6573, -0.0868, -1.4960, 0.9970]),
    "side_clear": np.array([0.0082, 1.5547, 0.0860, 1.5154, 0.0715, 0.0746, 0.9964]),
    "regrasp": np.array([-0.7479, 2.2253, 0.3580, 1.6707, 0.2096, 0.0177, 0.9958]),
}

REAL_L2_TAP_WAYPOINTS: Dict[str, np.ndarray] = {
    "stack_pin_release": np.concatenate([np.deg2rad([54.3, 144.6, 80.6, 68.0, 18.2, 0.2]), [0.99]]),
    "rim_approach": np.array([0.2287, 2.3596, 1.1549, 0.9897, -1.0657, 1.1149, 0.9933]),
    "rim_grab": np.array([0.6285, 2.2818, 0.7715, 1.5150, 0.0624, 1.4620, 0.0482]),
    "rim_flip_up": np.array([0.6292, 2.0895, 1.6207, -1.0248, -0.0681, 1.6306, 0.0454]),
    "rim_release": np.array([0.6094, 1.9717, 1.7164, -1.1027, -0.1013, 1.8469, 0.8366]),
    "lever_clear": np.array([0.4618, 0.0006, 0.1089, -0.1665, 0.2127, -0.0002, 0.8118]),
}

# ---------------------------------------------------------------------------
# 7. Dance Mode (official embeds only; no audio ships with the repo)
# ---------------------------------------------------------------------------

DANCE_TRACKS: Dict[str, dict] = {
    "rickroll": {
        "id": "rickroll", "title": "Never Gonna Give You Up", "artist": "Rick Astley",
        "youtube_id": "dQw4w9WgXcQ", "bpm": 113.0, "downbeat_s": 0.18, "style": "wave",
    },
    "crab": {
        "id": "crab", "title": "Crab Rave", "artist": "Noisestorm (Monstercat)",
        "youtube_id": "LDU_Txk06tM", "bpm": 125.0, "downbeat_s": 0.0, "style": "crab",
    },
    "stayin": {
        "id": "stayin", "title": "Stayin' Alive", "artist": "Bee Gees",
        "youtube_id": "fNFzfwLM72c", "bpm": 104.0, "downbeat_s": 0.12, "style": "strut",
    },
    "robot": {
        "id": "robot", "title": "Robot groove (built-in click track)", "artist": "Digital Twin",
        "youtube_id": "", "bpm": 96.0, "downbeat_s": 0.0, "style": "robot",
    },
}

DANCE_BASE_Q = np.array([0.15, 1.15, 1.15, -0.35, 0.15, 0.0])


# ---------------------------------------------------------------------------
# 8. Fingertip pad geometry in the TCP frame (z = approach, y = jaw axis, x = third axis), jaws closed
# ---------------------------------------------------------------------------

PAD_TIP_HALF = (0.0070, 0.0045, 0.0065)       # flat tip pad, inner face on the jaw midplane
PAD_FUNNEL_HALF = (0.0095, 0.0040, 0.0105)    # 18 degree lead-in behind the tip, so a cup is guided between the jaws
PAD_FUNNEL_DEG = 18.0


def pad_boxes_site():
    """[(side, name, center, R(columns = box axes), half sizes)] in the TCP frame at zero jaw opening."""
    phi = np.radians(PAD_FUNNEL_DEG)
    out = []
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        out.append((side, "tip", np.array([0.0, sgn * PAD_TIP_HALF[1], -PAD_TIP_HALF[2]]), np.eye(3), PAD_TIP_HALF))
        n = np.array([0.0, sgn * np.cos(phi), np.sin(phi)])
        e1 = np.array([1.0, 0.0, 0.0])
        e3 = np.cross(e1, n)
        face_c = np.array([0.0, sgn * PAD_FUNNEL_HALF[2] * np.sin(phi), -0.013 - PAD_FUNNEL_HALF[2] * np.cos(phi)])
        center = face_c + n * PAD_FUNNEL_HALF[1]
        out.append((side, "funnel", center, np.stack([e1, n, e3], axis=1), PAD_FUNNEL_HALF))
    return out


def pad_lowest_offset(R_tcp: np.ndarray) -> float:
    """Lowest z (relative to the TCP) of the closed fingertip blade for a TCP orientation R_tcp."""
    low = 0.0
    for side, name, c, R, h in pad_boxes_site():
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner = c + R @ (np.array([sx * h[0], sy * h[1], sz * h[2]]))
                    low = min(low, float((R_tcp @ corner)[2]))
    return low
