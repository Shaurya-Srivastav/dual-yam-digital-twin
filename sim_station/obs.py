"""What a skill is allowed to see. True cup and lever poses are not fields of this packet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ArmObservation:
    t_s: float
    q: np.ndarray
    qd: np.ndarray
    effort: np.ndarray
    grip_width_m: float
    grip_effort_n: float

    def finite(self) -> bool:
        return all(np.isfinite(v).all() if isinstance(v, np.ndarray) else np.isfinite(v)
                   for v in (self.t_s, self.q, self.qd, self.effort, self.grip_width_m, self.grip_effort_n))


class ReplayObservations:
    """A fixed list of packets. Stepping past the end returns the last one and sets stale."""

    def __init__(self, packets: list) -> None:
        if not packets:
            raise ValueError("replay needs at least one observation")
        self.packets = list(packets)
        self.i = 0
        self.stale = False

    def read(self) -> ArmObservation:
        pkt = self.packets[self.i]
        if self.i < len(self.packets) - 1:
            self.i += 1
            self.stale = False
        else:
            self.stale = True
        return pkt


def reject_action(q: np.ndarray, low: np.ndarray, high: np.ndarray) -> Optional[str]:
    q = np.asarray(q, dtype=float)
    if q.shape != low.shape or not np.isfinite(q).all():
        return "action must be a finite joint vector"
    if np.any(q < low) or np.any(q > high):
        return "action outside joint limits"
    return None
