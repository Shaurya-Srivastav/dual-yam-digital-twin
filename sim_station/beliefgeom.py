"""Geometry of the Level 5 search. These are consequences of the grid and the rim, not new measurements."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def grid_half_diagonal(spacing_m: float) -> float:
    return 0.5 * float(spacing_m) * np.sqrt(2.0)


def grid_can_miss_rim(spacing_m: float, rim_radius_m: float) -> bool:
    """True when a cup centre can sit in a cell so that no grid point lies inside the rim disk.

    The farthest point in a square cell is the centre, at half the diagonal from every corner.
    """
    return grid_half_diagonal(spacing_m) > float(rim_radius_m)


def feasible_region(points: Sequence[Sequence[float]]) -> dict:
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.size == 0 or pts.shape[-1] < 2:
        return {"n": 0, "centre": None, "radius_m": None}
    centre = pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(pts - centre, axis=1)))
    return {
        "n": int(len(pts)),
        "centre": centre,
        "radius_m": radius,
        "min_xy": pts.min(axis=0),
        "max_xy": pts.max(axis=0),
    }
