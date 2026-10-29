"""Numeric guards that do not need a full level run.

    python tests/test_numerics.py
"""
from __future__ import annotations

import sys

import numpy as np

from sim_station.audio import SR, analyze
from sim_station.evaluate import wilson_interval
from sim_station.motion import _slerp_R
from sim_station.skills import belief_walk_ok
from sim_station.skills_search import _on_rim


def test_wilson_contains_rate():
    lo, hi = wilson_interval(19, 24)
    assert 0.0 <= lo <= 19 / 24 <= hi <= 1.0
    assert hi - lo > 0.2          # 24 trials is a wide interval
    assert wilson_interval(0, 0)[0] != wilson_interval(0, 0)[0]


def test_belief_walk_yields_to_the_dish():
    assert belief_walk_ok([0.0, 0.0], [0.01, 0.0])
    assert not belief_walk_ok([0.0, 0.0], [0.001, 0.0])
    # dish pushed +y, belief wants -y
    assert not belief_walk_ok([0.0, 0.004], [0.0, -0.01])
    assert belief_walk_ok([0.0, -0.004], [0.0, -0.01])


def test_mouth_miss_is_not_a_rim_miss():
    centre = np.array([0.4, 0.1])
    assert not _on_rim(centre, centre, 0.040)
    assert _on_rim(centre + np.array([0.034, 0.0]), centre, 0.040)
    assert not _on_rim(centre + np.array([0.080, 0.0]), centre, 0.040)


def test_slerp_near_pi_is_a_rotation():
    R0 = np.eye(3)
    R1 = np.diag([1.0, -1.0, -1.0])          # 180 deg about x
    Ru = _slerp_R(R0, R1, 0.5)
    assert abs(np.linalg.det(Ru) - 1.0) < 1e-6
    assert np.allclose(Ru.T @ Ru, np.eye(3), atol=1e-6)
    assert np.allclose(_slerp_R(R0, R0, 0.3), R0)


def test_click_track_bpm():
    bpm = 120.0
    n = int(SR * 8)
    y = np.zeros(n)
    step = int(SR * 60.0 / bpm)
    for i in range(0, n - 200, step):
        y[i:i + 80] = np.hanning(80)
    a = analyze(y)
    assert 100.0 < a.bpm < 145.0, a.bpm
    assert len(a.beats) >= 8


def test_silence_does_not_crash():
    a = analyze(np.zeros(SR))
    assert np.isfinite(a.bpm)
    assert a.duration > 0.5


def main() -> int:
    bad = 0
    for t in (test_wilson_contains_rate, test_belief_walk_yields_to_the_dish, test_mouth_miss_is_not_a_rim_miss,
              test_slerp_near_pi_is_a_rotation, test_click_track_bpm, test_silence_does_not_crash):
        try:
            t()
            print("PASS", t.__name__)
        except Exception as e:
            bad += 1
            print("FAIL", t.__name__, type(e).__name__, e)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
