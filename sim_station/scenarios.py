"""Nominal station checks and the labeled hypothetical rim. Nothing here changes layout.py."""

from __future__ import annotations

from sim_station import layout as L

# The model rim is a radius. 76 mm is the diameter that radius implies.
NOMINAL_RIM_DIAMETER_M = 2.0 * L.CUP_RIM_RADIUS
# An earlier note said about 90 mm. It was not remeasured. It is not the model.
HYPOTHETICAL_RIM_DIAMETER_M = 0.090


def jaw_can_span(diameter_m: float, open_span_m: float = L.GRIPPER_MAX_OPEN_M) -> bool:
    return float(diameter_m) <= float(open_span_m) + 1e-9


def hypothetical_90mm() -> dict:
    ok = jaw_can_span(HYPOTHETICAL_RIM_DIAMETER_M)
    return {
        "name": "hypothetical_90mm_rim",
        "rim_diameter_m": HYPOTHETICAL_RIM_DIAMETER_M,
        "jaw_open_m": L.GRIPPER_MAX_OPEN_M,
        "feasible_for_this_gripper": ok,
        "status": "unsupported" if not ok else "untested",
        "note": "Not a measurement. Not substituted for the 76 mm nominal rim.",
    }


def nominal_scenario() -> dict:
    return {
        "id": "nominal_76mm_paper_seed0",
        "rim_diameter_m": NOMINAL_RIM_DIAMETER_M,
        "base_diameter_m": 2.0 * L.CUP_BASE_RADIUS,
        "cup_height_m": L.CUP_HEIGHT,
        "jaw_open_m": L.GRIPPER_MAX_OPEN_M,
        "materials": ("paper", "plastic"),
        "pour": "dry_timer",
        "physical_validation": False,
        "jaw_fits_nominal_rim": jaw_can_span(NOMINAL_RIM_DIAMETER_M),
    }


def layout_problems() -> list:
    """Configurations the current numbers would make impossible. Empty means the nominal scene is internally possible."""
    problems = []
    if L.PAPER_MASS <= 0 or L.PLASTIC_MASS <= 0:
        problems.append("cup mass must be positive")
    if L.CUP_HEIGHT <= 0 or L.CUP_RIM_RADIUS <= 0 or L.CUP_BASE_RADIUS <= 0:
        problems.append("cup dimensions must be positive")
    if L.CUP_BASE_RADIUS > L.CUP_RIM_RADIUS:
        problems.append("base radius larger than rim radius")
    if not jaw_can_span(NOMINAL_RIM_DIAMETER_M):
        problems.append("nominal rim does not fit the jaw opening")
    if L.NEST_MAX < 1 or L.NEST_DEFAULT < 1 or L.NEST_DEFAULT > L.NEST_MAX:
        problems.append("stack count is not in 1..NEST_MAX")
    if L.L5_REGION_X[0] >= L.L5_REGION_X[1] or L.L5_REGION_Y[0] >= L.L5_REGION_Y[1]:
        problems.append("search region is empty")
    return problems
