"""Level plans: dependency graphs of skills. Both arms run concurrently wherever the graph allows it."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from sim_station import layout as L
from sim_station import skills as S
from sim_station import skills_l2 as S2
from sim_station import skills_search as S3
from sim_station.runtime import Ctx, StepDef

GROUPS = ("Pick", "Pour", "Serve")


def _step(id_, title, group, arm, deps, fn, check=""):
    return StepDef(id=id_, title=title, group=group, arm=arm, deps=tuple(deps), make=fn, check=check)


def serve_steps(ctx: Ctx, prefix: str, after: str, seat_z: float, tap_after=()) -> List[StepDef]:
    """Common tail: spout, lever, pour, coaster, rest. `after` is the id of the step that leaves the cup in the hand, lifted."""
    p = prefix
    return [
        _step(f"{p}cup.carry", "Carry the cup beside the dispenser and lower it to spout height", "Pour", "cup", [after],
              lambda c: S.s_carry_to_spout(c), "cup base 8 mm above the mat in the corridor"),
        _step(f"{p}tap.ready", "Tap arm: open fingers beside the handle", "Pour", "tap", list(tap_after),
              lambda c: S.s_lever_ready(c), "fingers open 5 cm, pinch axis on the hinge axis"),
        _step(f"{p}cup.under", "Slide the cup under the outlet", "Pour", "cup", [f"{p}cup.carry"],
              lambda c: S.s_slide_under(c), "cup axis within 5 mm of the outlet"),
        _step(f"{p}tap.hold", "Tap arm: close the fingers on the handle", "Pour", "tap", [f"{p}tap.ready", f"{p}cup.under"],
              lambda c: S.s_lever_hold(c), "jaws stall near 15 mm, force below cap"),
        _step(f"{p}tap.lift", "Tap arm: lift the handle open", "Pour", "tap", [f"{p}tap.hold"],
              lambda c: S.s_lever_open(c), "handle above the flow angle"),
        _step(f"{p}pour", "Hold the valve open for the pour", "Pour", None, [f"{p}tap.lift"],
              lambda c: S.s_pour_wait(c), "flow clock from the handle angle"),
        _step(f"{p}tap.lower", "Tap arm: lower the handle shut", "Pour", "tap", [f"{p}pour"],
              lambda c: S.s_lever_close(c), "handle back on its stop"),
        _step(f"{p}tap.release", "Tap arm: let go and lift off the handle", "Pour", "tap", [f"{p}tap.lower"],
              lambda c: S.s_lever_release(c), "fingers open, hand clear"),
        _step(f"{p}cup.exit", "Slide the cup out from under the outlet", "Serve", "cup", [f"{p}tap.release"],
              lambda c: S.s_exit_spout(c), "moved sideways first, then up"),
        _step(f"{p}cup.to_coaster", "Carry the cup above the coaster", "Serve", "cup", [f"{p}cup.exit"],
              lambda c: S.s_to_coaster(c, seat_z), "3.5 cm above the seat"),
        _step(f"{p}cup.place", "Lower the cup into the coaster", "Serve", "cup", [f"{p}cup.to_coaster"],
              lambda c: S.s_place_cup(c, seat_z), "slow guarded descent, funnel centres the cup"),
        _step(f"{p}cup.release", "Open the jaws and let go", "Serve", "cup", [f"{p}cup.place"],
              lambda c: S.s_release(c), "jaws fully open"),
        _step(f"{p}cup.retreat", "Back the hand out clear", "Serve", "cup", [f"{p}cup.release"],
              lambda c: S.s_retreat(c), "moved away along the approach axis"),
        _step(f"{p}cup.rest", "Cup arm: return to rest", "Serve", "cup", [f"{p}cup.retreat"],
              lambda c: S.s_rest_over_cup(c), "folded"),
        _step(f"{p}tap.rest", "Tap arm: return to rest", "Serve", "tap", [f"{p}tap.release"],
              lambda c: S.s_goto_rest(c, "tap"), "folded"),
    ]


def pick_upright(ctx: Ctx, prefix: str, xy, base_z: float, first_deps=()) -> List[StepDef]:
    p = prefix
    xy_of = xy if callable(xy) else (lambda c, _xy=xy: _xy)
    return [
        _step(f"{p}cup.open", "Open the cup arm's jaws", "Pick", "cup", first_deps, lambda c: S.s_open(c, "cup"), "jaws at 77.6 mm"),
        _step(f"{p}cup.approach", "Move to a standoff in front of the cup", "Pick", "cup", [f"{p}cup.open"],
              lambda c: S.s_approach_cup(c, xy_of(c), base_z), "5 cm from the cup, hand level with its middle"),
        _step(f"{p}cup.slide_in", "Slide the open fingers in around the cup", "Pick", "cup", [f"{p}cup.approach"],
              lambda c: S.s_slide_in(c), "3 cm/s, jam guard 5 N"),
        _step(f"{p}cup.close", "Close the jaws on the cup wall", "Pick", "cup", [f"{p}cup.slide_in"],
              lambda c: S.s_close_cup(c), "jaws stall at a cup-sized width"),
        _step(f"{p}cup.lift", "Lift the cup out of its seat", "Pick", "cup", [f"{p}cup.close"],
              lambda c: S.s_lift_cup(c, 0.05), "5 cm up, no slip"),
    ]


def unstack_steps(ctx: Ctx, prefix: str = "", first_deps=()) -> List[StepDef]:
    p = prefix
    return [
        _step(f"{p}cup.approach", "Cup arm: roll the wrist the other way, then move in", "Pick", "cup", [f"{p}cup.open"],
              lambda c: S2.s_approach_stack(c), "hand upside down before the fingers reach the cup"),
        _step(f"{p}cup.slide_in", "Cup arm: slide the open fingers around the top cup", "Pick", "cup", [f"{p}cup.approach"],
              lambda c: S2.s_slide_in_stack(c), "3 cm/s, jam guard 5 N"),
        _step(f"{p}cup.close", "Cup arm: close on the top cup (gentle 15 N cap)", "Pick", "cup", [f"{p}cup.slide_in"],
              lambda c: S2.s_close_stack(c), "jaws stall near 67 mm on the middle band"),
        _step(f"{p}tap.pin_approach", "Tap arm: move beside the lower stack", "Pick", "tap", [f"{p}cup.close"],
              lambda c: S2.s_pin_approach(c), "pads open 77.6 mm, level with the band under the top rim"),
        _step(f"{p}tap.pin", "Tap arm: pin the lower stack against the mat", "Pick", "tap", [f"{p}tap.pin_approach"],
              lambda c: S2.s_pin_hold(c), "pads stall near 70 mm, 18 N cap"),
        _step(f"{p}cup.unstack", "Cup arm: wiggle the top cup free and lift it", "Pick", "cup", [f"{p}tap.pin"],
              lambda c: S2.s_unstack(c), "pull force collapses when the nested cups let go"),
        _step(f"{p}tap.unpin", "Tap arm: let go of the lower stack and back off", "Pick", "tap", [f"{p}cup.unstack"],
              lambda c: S2.s_pin_release(c), "pads open, hand clear"),
        _step(f"{p}tap.stow", "Tap arm: fold away from the stack", "Pick", "tap", [f"{p}tap.unpin"],
              lambda c: S.s_goto_rest(c, "tap"), "tap arm folded, out of the way"),
        _step(f"{p}cup.flip_lift", "Cup arm: roll the wrist back to its normal position", "Pick", "cup", [f"{p}tap.stow", f"{p}cup.unstack"],
              lambda c: S2.s_flip_lift(c), "cup upright in the ordinary pour grip, never set down"),
    ]


def build_level(level: int, ctx: Ctx) -> List[StepDef]:
    seat = L.COASTER_FLOOR_Z
    if level == 1:
        return pick_upright(ctx, "", ctx.bel.cup_xy, seat) + serve_steps(ctx, "", "cup.lift", seat)
    if level == 2:
        first = [_step("cup.open", "Cup arm: open the jaws", "Pick", "cup", [], lambda c: S.s_open(c, "cup"), "jaws open")]
        steps = first + unstack_steps(ctx, "")
        steps += serve_steps(ctx, "", "cup.flip_lift", seat, tap_after=("cup.flip_lift",))
        return steps
    if level == 3:
        steps = [
            _step("cup.blade", "Cup arm: close the fingertips into a blade", "Pick", "cup", [], lambda c: S3.s_close_fingertips(c), "jaws closed"),
            _step("cup.height", "Cup arm: peck down on the stack to measure its height", "Pick", "cup", ["cup.blade"],
                  lambda c: S3.s_find_height(c), "n = round((h - 11.0 cm) / 1.25 cm) + 1"),
            _step("cup.open", "Cup arm: open the jaws", "Pick", "cup", ["cup.height"], lambda c: S.s_open(c, "cup"), "jaws open"),
        ]
        steps += unstack_steps(ctx, "")
        steps += serve_steps(ctx, "", "cup.flip_lift", seat, tap_after=("cup.flip_lift",))
        return steps
    if level == 4:
        # The tap arm walks the whole circle of the valve. The cup arm stays folded until that walk is
        # finished, which is what keeps the two arms from meeting over the mat.
        steps = pick_upright(ctx, "", ctx.bel.cup_xy, seat, first_deps=("tap.search",)) + serve_steps(ctx, "", "cup.lift", seat, tap_after=("tap.search",))
        steps.insert(0, _step("tap.search", "Tap arm: pinch around the valve until the jaws stall on the handle", "Pour", "tap", [],
                              lambda c: S3.s_lever_search(c), "a close that stalls at 15 mm has the paddle; the dispenser is not pushed"))
        return steps
    if level == 5:
        steps = [
            _step("cup.stamp", "Left arm: open the gripper flat", "Pick", "cup", [],
                  lambda c: S3.s_flat_hand(c, "cup"), "jaws open, black pads parallel to the mat"),
            _step("cup.scan", "Left arm: stomp its half of the mat", "Pick", "cup", ["cup.stamp"],
                  lambda c: S3.s_scan_half(c, "cup"), "flat black pads, then the centre of the patch that stood up"),
            _step("tap.stamp", "Right arm: open the gripper flat", "Pick", "tap", ["cup.scan"],
                  lambda c: S3.s_flat_hand(c, "tap"), "starts only after the left arm has folded clear"),
            _step("tap.scan", "Right arm: stomp its half of the mat", "Pick", "tap", ["tap.stamp"],
                  lambda c: S3.s_scan_right(c), "the other half; the arms do not cross"),
        ]
        steps += pick_upright(ctx, "", lambda c: c.bel.cup_xy, L.MAT_SURFACE_Z, first_deps=("tap.scan",))
        # The tap arm stays folded until the cup is in the hand. Moving it to the lever during the stomp
        # is how the two arms met.
        steps += serve_steps(ctx, "", "cup.lift", seat, tap_after=("cup.lift",))
        return steps
    raise ValueError(f"level {level} not built yet")
