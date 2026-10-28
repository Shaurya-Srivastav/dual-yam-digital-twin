"""Smoke test: every level once, nominal, then the real-data replay. Exit code 0 only if all of it passes.

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
    print(f"real L1 replay: {'PASS' if rep['ok'] else 'FAIL'}  under outlet {rep['under_outlet_mm']} mm, back in coaster {rep['returned_to_coaster_mm']} mm")
    eq = proofs.equivalence_l1()
    print(f"IK reproduces the real poses: {'PASS' if eq['ok'] else 'FAIL'}")
    bad += 0 if rep["ok"] else 1
    bad += 0 if eq["ok"] else 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
