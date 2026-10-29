"""The published evaluation window, and seed partitions that have not been scored yet.

Seeds 0 through 23 are the window behind 79/42/42/21/12 on commit 32b5d08.
The dev, test and stress tuples below do not overlap that window and have not been run.
"""

from __future__ import annotations

SPEC_VERSION = "sim-2026-10-28"

THRESHOLDS = {
    "cup_upright_deg": 10.0,
    "coaster_mm": 15.0,
    "flow_s": (6.8, 7.2),
    "dispenser_creep_mm": 2.0,
}

PUBLISHED_SEEDS = tuple(range(24))
PUBLISHED_RATES = {1: (19, 24), 2: (10, 24), 3: (10, 24), 4: (5, 24), 5: (3, 24)}
PUBLISHED_COMMIT = "32b5d08"

# Held out from the published window on purpose. Do not quote a rate for these until evaluate finishes.
DEV_SEEDS = tuple(range(100, 108))
TEST_SEEDS = tuple(range(1000, 1024))
STRESS_SEEDS = tuple(range(5000, 5024))
NOMINAL_SEED = (0,)


def partitions_are_disjoint() -> bool:
    groups = (PUBLISHED_SEEDS, DEV_SEEDS, TEST_SEEDS, STRESS_SEEDS)
    seen = set()
    for group in groups:
        if seen & set(group):
            return False
        seen |= set(group)
    return NOMINAL_SEED[0] in PUBLISHED_SEEDS
