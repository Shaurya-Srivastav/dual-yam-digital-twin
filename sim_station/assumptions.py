"""Where each important number came from. None of these were remeasured for this tree.

Categories: recorded (from a log already in the repo), derived (computed from those),
assumed (chosen for the scene), fitted (changed so a replay would pass).
"""

from __future__ import annotations

CATEGORIES = ("recorded", "derived", "assumed", "fitted")

ENTRIES = (
    ("CUP_RIM_RADIUS", "assumed", "76 mm nominal. An earlier note said about 90 mm. Not remeasured."),
    ("CUP_BASE_RADIUS", "assumed", "54 mm nominal base. Not remeasured."),
    ("CUP_HEIGHT", "assumed", "110 mm nominal height. Not remeasured."),
    ("CUP_WALL_THICK", "assumed", "Wall thickness used by the hollow cup mesh."),
    ("PAPER_MASS", "assumed", "Paper cup mass used by the simulator."),
    ("PLASTIC_MASS", "assumed", "Plastic cup mass used by the simulator."),
    ("NEST_STEP", "assumed", "Exposed rim step of a nested cup."),
    ("STACK_BREAK_FORCE_N", "assumed", "Scripted tether break force, not a pull-test measurement."),
    ("STACK_FATIGUE_LEN_M", "assumed", "Scripted wiggle length that weakens the tether."),
    ("CUP_MOUNT", "recorded", "Cup-arm base used by the scene."),
    ("TAP_BASE_OFFSET_IN_CUP_FRAME", "derived", "Tap base from the two-grasp calculation, not a rail survey."),
    ("COASTER_FUNNEL_R_BOTTOM", "fitted", "Widened so the recorded grasp, 2 cm off the placed axis, can sit in the dish."),
    ("COASTER_FUNNEL_R_TOP", "fitted", "Matched to the widened dish."),
    ("MAT_HALF_SIZE", "assumed", "24 by 36 inch mat extent used by the scene."),
    ("L5_REGION_X", "assumed", "Reachable search interval. Not the whole mat."),
    ("L5_REGION_Y", "assumed", "Reachable search interval. Not the whole mat."),
    ("REAL_L1_CUP_WAYPOINTS", "recorded", "Taught Level 1 joint vectors already in the repo."),
)
