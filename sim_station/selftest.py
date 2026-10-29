"""Nominal smoke test. Exit code 0 means all five levels pass the strict referee with no randomization,
the pose-aligned Level 1 waypoint replay passes, and FK/IK self-consistency passes.

It does not mean the randomized rates are at target, and it does not mean the scene matches the
physical station. The scene-as-reset replay is printed either way and is not part of the exit code:
that one leaves the cup where the level reset put it.

    python -m sim_station.selftest
"""

from __future__ import annotations

import os
import sys
import time


from sim_station import proofs
from sim_station.runner import Session


def main() -> int:
    S = Session()
    bad = 0
    for level in (1, 2, 3, 4, 5):
        S.reset(level, "paper", seed=0)
        t0 = time.time()
        S.run()
        r = S.check_pass()
        print(f"level {level}: {'PASS' if r['ok'] else 'FAIL'}  sim {r['sim_time_s']:.0f}s  wall {time.time() - t0:.0f}s  "
              f"cup {r['coaster_offset_mm']:.0f} mm from centre, flow {r['flow_s']:.2f}s  {r['reasons'][:2] if not r['ok'] else ''}")
        bad += 0 if r["ok"] else 1
    rep = proofs.replay_l1()
    print(f"pose-aligned L1 replay: {'PASS' if rep['ok'] else 'FAIL'}  under outlet {rep['under_outlet_mm']} mm, back in coaster {rep['returned_to_coaster_mm']} mm")
    scene = proofs.replay_l1_scene()
    print(f"scene-as-reset L1 replay (not a pass criterion): {'PASS' if scene['ok'] else 'FAIL'}  "
          f"under outlet {scene['under_outlet_mm']} mm, back in coaster {scene['returned_to_coaster_mm']} mm")
    eq = proofs.equivalence_l1()
    print(f"FK/IK self-consistency: {'PASS' if eq['ok'] else 'FAIL'}")
    bad += 0 if rep["ok"] else 1
    bad += 0 if eq["ok"] else 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
