"""Which contacts a planned path is allowed to keep.

A pair that is already touching at the first sample may stay in contact. Once that
pair separates, touching it again is a new collision. This does not read the simulator.
"""

from __future__ import annotations

from typing import Iterable, Set, Tuple

Pair = Tuple[str, str]

# Recorded by the referee. arm_mat is logged and is not in this list.
FATAL_CONTACT = ("arm_arm", "arm_dispenser", "arm_table", "cup_dispenser", "arm_stack")


def still_leaving(active: Iterable[Pair], present: Iterable[Pair]) -> Set[Pair]:
    return set(active) & set(present)
