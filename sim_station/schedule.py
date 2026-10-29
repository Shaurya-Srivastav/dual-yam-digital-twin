"""When a step is allowed to start. A global step does not share the arms with another step."""

from __future__ import annotations

from typing import Optional


def lanes_allow(arm: Optional[str], lane: dict, global_running: bool) -> bool:
    if arm is None:
        if global_running:
            return False
        return lane.get("cup") is None and lane.get("tap") is None
    if global_running:
        return False
    return lane.get(arm) is None
