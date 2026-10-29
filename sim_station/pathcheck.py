"""Cartesian waypoint checks that do not need a robot model."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class PathSpecError(ValueError):
    pass


def cartesian_waypoints(poses: Sequence[np.ndarray], trans_eps: float = 1e-9, rot_eps: float = 1e-9) -> List[np.ndarray]:
    """Drop consecutive poses that do not move. A pure rotation is kept. No motion is an error."""
    if len(poses) < 2:
        raise PathSpecError("a cartesian path needs at least two poses")
    cleaned: List[np.ndarray] = []
    for raw in poses:
        T = np.asarray(raw, dtype=float)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise PathSpecError("cartesian poses must be finite 4x4 matrices")
        if cleaned and _same(cleaned[-1], T, trans_eps, rot_eps):
            continue
        cleaned.append(T)
    if len(cleaned) < 2:
        raise PathSpecError("cartesian path has no motion")
    return cleaned


def _same(a: np.ndarray, b: np.ndarray, trans_eps: float, rot_eps: float) -> bool:
    if float(np.linalg.norm(a[:3, 3] - b[:3, 3])) > trans_eps:
        return False
    return float(np.linalg.norm(a[:3, :3] - b[:3, :3])) <= rot_eps
