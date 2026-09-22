"""Local aliases so controloptimize can import section / pump short keys without
depending on package install paths.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from naming import (  # noqa: E402,F401
    KEY_HJT,
    KEY_TJQ,
    KEY_WJT,
    KEY_XHJT,
    KEY_XL,
    SEC_HJT_XL,
    SEC_TJQ_WWTP2,
    SEC_WJT_TJQ,
    SEC_XHJT_WWTP2,
    SEC_XL_WWTP1,
)
