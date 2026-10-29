"""Bounds for the local dashboard. Invalid requests fail before they touch the simulation."""

from __future__ import annotations

import math
from typing import Optional, Sequence


def check_reset(level: Optional[int], material: Optional[str], seed: Optional[int]) -> None:
    if level is not None and int(level) not in (1, 2, 3, 4, 5):
        raise ValueError("level must be 1, 2, 3, 4 or 5")
    if material is not None and material not in ("paper", "plastic"):
        raise ValueError("material must be paper or plastic")
    if seed is not None and not 0 <= int(seed) <= 1_000_000:
        raise ValueError("seed out of range")


def check_speed(speed: float) -> None:
    if not math.isfinite(float(speed)) or not 0.05 <= float(speed) <= 8.0:
        raise ValueError("speed out of range")


def check_trials(levels: Sequence[int], trials: int) -> None:
    if not levels or any(int(x) not in (1, 2, 3, 4, 5) for x in levels):
        raise ValueError("levels must be chosen from 1..5")
    if not 1 <= int(trials) <= 200:
        raise ValueError("trials must be from 1 to 200")


def check_camera(az: Optional[float], el: Optional[float], dist: Optional[float], lookat: Optional[Sequence[float]]) -> None:
    for name, value in (("az", az), ("el", el), ("dist", dist)):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if el is not None and not -89.5 <= float(el) <= 89.5:
        raise ValueError("elevation out of range")
    if dist is not None and not 0.15 <= float(dist) <= 8.0:
        raise ValueError("distance out of range")
    if az is not None and not -3600.0 <= float(az) <= 3600.0:
        raise ValueError("azimuth out of range")
    if lookat is not None and (len(lookat) != 3 or any(not math.isfinite(float(v)) for v in lookat)):
        raise ValueError("lookat must be three finite numbers")


def check_pick(x: float, y: float) -> None:
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        raise ValueError("pick point must be finite")
    if not -1.0 <= float(x) <= 2.0 or not -1.5 <= float(y) <= 1.5:
        raise ValueError("pick point is off the table")


def check_song(song: str, known: Sequence[str]) -> None:
    if song not in known:
        raise ValueError("unknown song")
