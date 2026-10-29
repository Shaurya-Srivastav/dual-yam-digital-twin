"""Checks that do not load MuJoCo.  python tests/test_offline.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sim_station.api_checks import check_camera, check_pick
from sim_station.audio import SONGS, analyze, click_track, wav_bytes
from sim_station.beliefgeom import feasible_region, grid_can_miss_rim, grid_half_diagonal
from sim_station.benchmark_spec import (
    DEV_SEEDS, PUBLISHED_RATES, PUBLISHED_SEEDS, STRESS_SEEDS, TEST_SEEDS, partitions_are_disjoint,
)
from sim_station.collision_policy import FATAL_CONTACT, still_leaving
from sim_station.dance import STYLE_PLAN
from sim_station.episode import EpisodeRecord
from sim_station.physics_manifest import OPTION
from sim_station.schedule import lanes_allow
from sim_station.frames import compose_pose, invert_pose, pose_ok
from sim_station.obs import ArmObservation, ReplayObservations, reject_action
from sim_station.pathcheck import PathSpecError, cartesian_waypoints
from sim_station.scenarios import hypothetical_90mm, layout_problems, nominal_scenario
from sim_station import layout as L


def test_grid_does_not_cover_the_rim():
    gap = grid_half_diagonal(0.060)
    assert gap > L.CUP_RIM_RADIUS
    assert grid_can_miss_rim(0.060, L.CUP_RIM_RADIUS)
    assert not grid_can_miss_rim(0.040, L.CUP_RIM_RADIUS)


def test_feasible_region_radius():
    reg = feasible_region([[0.0, 0.0], [0.02, 0.0]])
    assert reg["n"] == 2 and abs(reg["radius_m"] - 0.01) < 1e-9


def test_repeated_pose_is_rejected_and_a_turn_is_kept():
    eye = np.eye(4)
    try:
        cartesian_waypoints([eye, eye.copy()])
        raise AssertionError("zero motion accepted")
    except PathSpecError:
        pass
    turned = eye.copy()
    turned[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert len(cartesian_waypoints([eye, turned])) == 2
    bad = eye.copy()
    bad[0, 3] = np.nan
    try:
        cartesian_waypoints([eye, bad])
        raise AssertionError("nan accepted")
    except PathSpecError:
        pass


def test_leaving_expires_after_separation():
    start = {("pad", "cup")}
    assert still_leaving(start, {("pad", "cup")}) == start
    assert still_leaving(start, set()) == set()
    assert still_leaving(set(), {("pad", "cup")}) == set()


def test_pose_round_trip():
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [0.1, -0.2, 0.3]
    assert pose_ok(T)
    back = compose_pose(T, invert_pose(T))
    assert np.allclose(back, np.eye(4), atol=1e-9)


def test_nominal_rim_fits_and_90mm_does_not():
    assert layout_problems() == []
    nom = nominal_scenario()
    assert nom["jaw_fits_nominal_rim"] and nom["physical_validation"] is False
    hyp = hypothetical_90mm()
    assert hyp["feasible_for_this_gripper"] is False
    assert hyp["status"] == "unsupported"


def test_seed_partitions_do_not_overlap_the_published_window():
    assert partitions_are_disjoint()
    assert set(DEV_SEEDS).isdisjoint(PUBLISHED_SEEDS)
    assert set(TEST_SEEDS).isdisjoint(PUBLISHED_SEEDS)
    assert set(STRESS_SEEDS).isdisjoint(PUBLISHED_SEEDS)
    assert PUBLISHED_RATES[1] == (19, 24)


def test_replay_goes_stale_and_limits_reject_nan():
    pkt = ArmObservation(0.0, np.zeros(6), np.zeros(6), np.zeros(6), 0.05, 0.0)
    replay = ReplayObservations([pkt])
    assert replay.read() is pkt and replay.stale
    low, high = np.full(6, -1.0), np.full(6, 1.0)
    assert reject_action(np.zeros(6), low, high) is None
    assert reject_action(np.full(6, np.nan), low, high)


def test_click_track_tempo():
    bpm = analyze(click_track(120.0, seconds=8.0)).bpm
    assert 100.0 < bpm < 145.0


def test_global_step_does_not_share_an_arm():
    assert lanes_allow(None, {"cup": None, "tap": None}, False)
    assert not lanes_allow(None, {"cup": "cup.lift", "tap": None}, False)
    assert not lanes_allow("tap", {"cup": None, "tap": None}, True)
    assert lanes_allow("cup", {"cup": None, "tap": "tap.ready"}, False)


def test_fatal_contacts_are_the_referee_set():
    assert "arm_mat" not in FATAL_CONTACT
    assert set(FATAL_CONTACT) == {"arm_arm", "arm_dispenser", "arm_table", "cup_dispenser", "arm_stack"}


def test_short_and_nonfinite_audio_do_not_crash():
    short = analyze(np.zeros(16))
    assert short.beat_pos(1.0) == 0.0
    assert not np.isfinite(short.bpm)
    noisy = analyze(np.array([np.nan, np.inf, 0.0]))
    assert noisy.beat_pos(0.2) == 0.0
    try:
        click_track(0)
        raise AssertionError("zero bpm accepted")
    except ValueError:
        pass


def test_click_track_is_a_wav_and_has_a_dance():
    assert SONGS["click"]["generated"] is True
    assert "click" in STYLE_PLAN
    blob = wav_bytes(click_track(120.0, seconds=1.0))
    assert blob[:4] == b"RIFF" and len(blob) > 1000


def test_camera_and_pick_reject_nonfinite():
    check_camera(10.0, -20.0, 1.5, [0.4, 0.0, 0.1])
    check_pick(0.4, 0.0)
    for bad in (
        lambda: check_camera(float("nan"), 0.0, 1.0, None),
        lambda: check_camera(0.0, 120.0, 1.0, None),
        lambda: check_pick(float("inf"), 0.0),
    ):
        try:
            bad()
            raise AssertionError("accepted")
        except ValueError:
            pass


def test_physics_option_is_the_documented_solver():
    assert OPTION["timestep"] == 0.001
    assert OPTION["integrator"] == "implicitfast"
    assert OPTION["solver"] == "Newton"
    rec = EpisodeRecord(level=1, seed=0)
    assert rec.physical_validation is False and rec.privileged_true_pose is None


def test_workflow_cannot_start_the_arms():
    text = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "sim.yml").read_text()
    assert "--hardware" not in text
    assert "SIM_ALLOW_HARDWARE" not in text


def main() -> int:
    bad = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception as exc:
            bad += 1
            print("FAIL", t.__name__, type(exc).__name__, exc)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
