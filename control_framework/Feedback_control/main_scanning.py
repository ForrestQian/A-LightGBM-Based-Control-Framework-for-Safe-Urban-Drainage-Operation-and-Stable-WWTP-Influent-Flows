"""Feedback-control entry point (offline demo).

1) Scan latest levels against configured limits.
2) Run the rule-based decision tree.
3) Print / optionally save recommended corrections.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ruamel import yaml

from decision_tree import FeedbackTree, scan_limits
from naming_fb import (
    PUMP_HJT,
    PUMP_TJQ,
    PUMP_WJT,
    PUMP_XHJT,
    PUMP_XL,
    WWTP_1,
    WWTP_2,
)

HERE = Path(__file__).resolve().parent


def demo_levels(seed: int = 0) -> pd.Series:
    """Synthetic latest levels for a walk-through (not field data)."""
    rng = np.random.default_rng(seed)
    return pd.Series(
        {
            PUMP_XL: rng.uniform(-1.5, 1.0),
            PUMP_HJT: rng.uniform(-2.0, 1.0),
            PUMP_XHJT: rng.uniform(-3.5, 1.0),
            PUMP_TJQ: rng.uniform(-1.5, 1.5),
            PUMP_WJT: rng.uniform(-1.5, 1.0),
            WWTP_1: rng.uniform(0.2, 2.2),
            WWTP_2: rng.uniform(0.2, 2.2),
        }
    )


def main() -> None:
    with open(HERE / "config.yml", "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.RoundTripLoader)

    levels = demo_levels()
    warnings = scan_limits(levels, dict(config["scanning"]["level_limit"]["pump"]))
    print("Warnings:")
    print(warnings.to_string(index=False) if not warnings.empty else "(none)")

    tree = FeedbackTree(config=config, levels=levels)
    strategies = tree.run(warnings) if not warnings.empty else {}
    print("\nStrategies:")
    for k, v in strategies.items():
        print(f"  {k}: {v}")

    if config.get("write_output"):
        out = HERE / str(config.get("output_csv") or "feedback_actions.csv")
        pd.DataFrame(
            [{"code": k, "action": v, "ctm": datetime.now().isoformat(timespec="seconds")}
             for k, v in strategies.items()]
        ).to_csv(out, index=False)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
