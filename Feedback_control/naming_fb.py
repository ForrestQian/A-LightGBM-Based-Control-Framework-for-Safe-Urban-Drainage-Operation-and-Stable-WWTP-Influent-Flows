"""Re-export anonymized IDs for the feedback module."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from naming import (  # noqa: E402,F401
    PUMP_HJT,
    PUMP_TJQ,
    PUMP_WJT,
    PUMP_XHJT,
    PUMP_XL,
    WWTP_1,
    WWTP_2,
)
