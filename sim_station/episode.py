"""Record of one simulated episode. The true object pose is stored beside the observation, not inside it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

SCHEMA = "episode-sim-1"


@dataclass
class EpisodeRecord:
    schema: str = SCHEMA
    level: int = 1
    seed: int = 0
    material: str = "paper"
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    privileged_true_pose: Optional[dict] = None
    passed: Optional[bool] = None
    first_failure: str = ""
    physical_validation: bool = False
