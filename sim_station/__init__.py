"""Digital twin of the Everest dual-YAM water-cup station (MuJoCo)."""

import os
import sys
from pathlib import Path

# The i2rt SDK supplies the YAM arm and gripper models. Point I2RT_PATH at a clone of it (default: ./third_party/i2rt).
_I2RT = os.environ.get("I2RT_PATH", str(Path(__file__).resolve().parents[1] / "third_party" / "i2rt"))
if os.path.isdir(_I2RT) and _I2RT not in sys.path:
    sys.path.insert(0, _I2RT)
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
