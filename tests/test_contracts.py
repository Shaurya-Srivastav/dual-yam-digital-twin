"""Contracts that do not need a full level. The graph check does load the simulator once.

    python tests/test_contracts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sim_station import assumptions, layout as L
from sim_station.api_checks import check_reset, check_song, check_speed, check_trials
from sim_station.audio import SONGS
from sim_station.evaluate import write_report
from sim_station.jsonsafe import json_ready
from sim_station.motion import PlanError, cart_path
from sim_station.runtime import StepDef, validate_steps
from sim_station.sensing import jacobian_reliable, world_force_from_tau


def _step(i, deps=()):
    return StepDef(id=i, title=i, group="Pick", arm="cup", deps=tuple(deps), make=lambda c: iter(()))


def test_graph_rejects_a_cycle_and_a_missing_dep():
    try:
        validate_steps([_step("a", ("b",)), _step("b", ("a",))])
        raise AssertionError("cycle was accepted")
    except ValueError as e:
        assert "cycle" in str(e)
    try:
        validate_steps([_step("a", ("missing",))])
        raise AssertionError("missing dep was accepted")
    except ValueError as e:
        assert "missing" in str(e)
    validate_steps([_step("a"), _step("b", ("a",))])


def test_levels_build():
    from sim_station.runner import Session
    S = Session()
    for level in (1, 2, 3, 4, 5):
        S.reset(level, "paper", seed=0)
        assert len(S.steps) > 5
        validate_steps(S.steps)


def test_stack_assist_can_be_turned_off():
    from sim_station.runner import Session
    S = Session()
    S.reset(2, "paper", seed=0)
    P = S.plant
    assert P.stack_assist is True
    b = P.body[P.active_rest]
    P.data.xfrc_applied[:] = 0.0
    P.stack_assist = False
    P._stabilize_stacks()
    assert np.allclose(P.data.xfrc_applied[b], 0.0)


def test_torque_residual_sign():
    Jp = np.zeros((3, 6))
    Jp[:, :3] = np.eye(3)
    force = world_force_from_tau(Jp, np.array([1.0, 0, 0, 0, 0, 0]), np.eye(3))
    assert force[0] < -0.9
    assert jacobian_reliable(Jp)
    assert not jacobian_reliable(np.zeros((3, 6)))


def test_short_cartesian_path():
    try:
        cart_path(None, [np.eye(4)], np.zeros(6))
        raise AssertionError("one pose was accepted")
    except PlanError:
        pass


def test_json_ready_drops_nan():
    raw = json.dumps(json_ready({"a": float("nan"), "b": np.float64(1.5), "c": [float("inf")]}), allow_nan=False)
    got = json.loads(raw)
    assert got == {"a": None, "b": 1.5, "c": [None]}


def test_api_bounds():
    check_reset(1, "paper", 0)
    for bad in (
        lambda: check_reset(9, "paper", 0),
        lambda: check_reset(1, "glass", 0),
        lambda: check_speed(0.0),
        lambda: check_speed(float("nan")),
        lambda: check_trials([6], 10),
        lambda: check_trials([1], 0),
        lambda: check_song("nope", SONGS),
    ):
        try:
            bad()
            raise AssertionError("accepted")
        except ValueError:
            pass
    check_song("rick", SONGS)


def test_hardware_stays_off():
    import os
    os.environ.pop("SIM_ALLOW_HARDWARE", None)
    from sim_station.sim_robot import get_robot
    try:
        get_robot("cup", hardware=True)
        raise AssertionError("hardware was allowed")
    except RuntimeError as e:
        assert "real arms" in str(e)


def test_skills_do_not_read_cup_pose():
    root = Path(__file__).resolve().parents[1] / "sim_station"
    for name in ("skills.py", "skills_l2.py", "skills_search.py"):
        text = (root / name).read_text()
        assert "cup_state(" not in text, name
        assert "outlet_xy_true(" not in text, name


def test_assumptions_name_real_attributes():
    for name, cat, note in assumptions.ENTRIES:
        assert cat in assumptions.CATEGORIES
        assert hasattr(L, name), name
        assert note


def test_report_writer(tmp_path=None):
    import tempfile
    dest = Path(tempfile.mkdtemp()) / "out.json"
    write_report(str(dest), {1: {"results": [{"ok": False, "sim_time_s": float("nan")}], "passed": 0}})
    json.loads(dest.read_text())
    row = json.loads(dest.with_suffix(".jsonl").read_text().strip())
    assert row["sim_time_s"] is None


def main() -> int:
    bad = 0
    tests = [test_graph_rejects_a_cycle_and_a_missing_dep, test_torque_residual_sign, test_short_cartesian_path, test_json_ready_drops_nan,
             test_api_bounds, test_hardware_stays_off, test_skills_do_not_read_cup_pose, test_assumptions_name_real_attributes,
             test_report_writer, test_levels_build, test_stack_assist_can_be_turned_off]
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception as e:
            bad += 1
            print("FAIL", t.__name__, type(e).__name__, e)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
