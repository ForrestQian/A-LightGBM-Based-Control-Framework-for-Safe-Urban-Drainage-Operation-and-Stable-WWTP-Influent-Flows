"""Minute-scale scanning and rule-based decision-tree feedback.

warning_type:
  1 = low limit
  2 = high limit

Actions are expressed as anonymized pump tokens (Pump-XL, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from naming_fb import (
    PUMP_HJT,
    PUMP_TJQ,
    PUMP_WJT,
    PUMP_XHJT,
    PUMP_XL,
    WWTP_1,
    WWTP_2,
)

MANUAL = "No automatic action; manual intervention required."


@dataclass
class FeedbackTree:
    """Hand-crafted decision tree for corrective pump recommendations."""

    config: dict
    levels: pd.Series  # latest levels indexed by station ID
    strategies: dict[str, str] = field(default_factory=dict)

    def _lim(self, station: str) -> list[float]:
        return list(self.config["control"]["level_limit"]["pump"][station])

    def run(self, warnings: pd.DataFrame) -> dict[str, str]:
        self.strategies = {}
        for code, gdf in warnings.groupby("code"):
            if code in (WWTP_1, WWTP_2):
                self._wwtp_control(code, gdf)
            elif code == PUMP_XL:
                self._xl_control(code, gdf)
            elif code == PUMP_TJQ:
                self._tjq_control(code, gdf)
            elif code == PUMP_XHJT:
                self._xhjt_control(code, gdf)
            elif code == PUMP_HJT:
                self._hjt_control(code, gdf)
            elif code == PUMP_WJT:
                self._wjt_control(code, gdf)
        return self.strategies

    def _wwtp_control(self, code: str, idf: pd.DataFrame) -> None:
        """WWTP high/low -> adjust Pump-XL / Pump-HJT / Pump-XHJT."""
        wtype = int(idf["warning_type"].iloc[0])
        xl_l, hjt_l, xhjt_l = self._lim(PUMP_XL), self._lim(PUMP_HJT), self._lim(PUMP_XHJT)
        xl, hjt, xhjt = self.levels[PUMP_XL], self.levels[PUMP_HJT], self.levels[PUMP_XHJT]

        if wtype == 2:  # high: try to store upstream
            if xl <= xl_l[0] and hjt <= hjt_l[0]:
                self.strategies[code] = f"{PUMP_XL}: reduce 1 unit; {PUMP_HJT}: reduce 1 unit."
            elif xl <= xl_l[0]:
                self.strategies[code] = f"{PUMP_XL}: reduce 1 unit."
            elif hjt <= hjt_l[0]:
                self.strategies[code] = f"{PUMP_HJT}: reduce 1 unit."
            elif xhjt <= xhjt_l[0]:
                self.strategies[code] = f"{PUMP_XHJT}: reduce 1 unit."
            else:
                self.strategies[code] = MANUAL
        elif wtype == 1:  # low: release upstream storage
            if xhjt >= xhjt_l[1]:
                self.strategies[code] = f"{PUMP_XHJT}: increase 1 unit."
            elif xl >= xl_l[1] and hjt >= hjt_l[1]:
                self.strategies[code] = f"{PUMP_XL}: increase 1 unit; {PUMP_HJT}: increase 1 unit."
            elif xl >= xl_l[1]:
                self.strategies[code] = f"{PUMP_XL}: increase 1 unit."
            elif hjt >= hjt_l[1]:
                self.strategies[code] = f"{PUMP_HJT}: increase 1 unit."
            else:
                self.strategies[code] = MANUAL

    def _xl_control(self, code: str, idf: pd.DataFrame) -> None:
        wtype = int(idf["warning_type"].iloc[0])
        hjt_l, wwtp_l = self._lim(PUMP_HJT), self._lim(WWTP_1)
        hjt, wwtp = self.levels[PUMP_HJT], self.levels[WWTP_1]
        if wtype == 2:
            if wwtp <= wwtp_l[0]:
                self.strategies[code] = f"{PUMP_XL}: increase 1 unit."
            elif hjt <= hjt_l[0]:
                self.strategies[code] = f"{PUMP_HJT}: reduce 1 unit."
            else:
                self.strategies[code] = MANUAL
        elif wtype == 1:
            if wwtp >= wwtp_l[1]:
                self.strategies[code] = f"{PUMP_XL}: reduce 1 unit."
            elif hjt >= hjt_l[1]:
                self.strategies[code] = f"{PUMP_HJT}: increase 1 unit."
            else:
                self.strategies[code] = MANUAL

    def _tjq_control(self, code: str, idf: pd.DataFrame) -> None:
        wtype = int(idf["warning_type"].iloc[0])
        wjt_l, wwtp_l = self._lim(PUMP_WJT), self._lim(WWTP_2)
        wjt, wwtp = self.levels[PUMP_WJT], self.levels[WWTP_2]
        if wtype == 2:
            if wwtp <= wwtp_l[0]:
                self.strategies[code] = f"{PUMP_TJQ}: increase 1 unit."
            elif wjt <= wjt_l[0]:
                self.strategies[code] = f"{PUMP_WJT}: reduce 1 unit."
            else:
                self.strategies[code] = MANUAL
        elif wtype == 1:
            if wwtp >= wwtp_l[1]:
                self.strategies[code] = f"{PUMP_TJQ}: reduce 1 unit."
            elif wjt >= wjt_l[1]:
                self.strategies[code] = f"{PUMP_WJT}: increase 1 unit."
            else:
                self.strategies[code] = MANUAL

    def _xhjt_control(self, code: str, idf: pd.DataFrame) -> None:
        wtype = int(idf["warning_type"].iloc[0])
        wwtp_l = self._lim(WWTP_2)
        wwtp = self.levels[WWTP_2]
        if wtype == 2:
            self.strategies[code] = (
                f"{PUMP_XHJT}: increase 1 unit." if wwtp < wwtp_l[0] else MANUAL
            )
        elif wtype == 1:
            self.strategies[code] = (
                f"{PUMP_XHJT}: reduce 1 unit." if wwtp > wwtp_l[1] else MANUAL
            )

    def _hjt_control(self, code: str, idf: pd.DataFrame) -> None:
        wtype = int(idf["warning_type"].iloc[0])
        xl_l = self._lim(PUMP_XL)
        xl = self.levels[PUMP_XL]
        if wtype == 2:
            self.strategies[code] = (
                f"{PUMP_HJT}: increase 1 unit." if xl <= xl_l[0] else MANUAL
            )
        elif wtype == 1:
            self.strategies[code] = (
                f"{PUMP_HJT}: reduce 1 unit." if xl >= xl_l[1] else MANUAL
            )

    def _wjt_control(self, code: str, idf: pd.DataFrame) -> None:
        wtype = int(idf["warning_type"].iloc[0])
        tjq_l = self._lim(PUMP_TJQ)
        tjq = self.levels[PUMP_TJQ]
        if wtype == 2:
            self.strategies[code] = (
                f"{PUMP_WJT}: increase 1 unit." if tjq <= tjq_l[0] else MANUAL
            )
        elif wtype == 1:
            self.strategies[code] = (
                f"{PUMP_WJT}: reduce 1 unit." if tjq >= tjq_l[1] else MANUAL
            )


def scan_limits(levels: pd.Series, limits: dict[str, list[float]]) -> pd.DataFrame:
    """Emit warning rows for stations outside [low, high]."""
    rows = []
    for code, (lo, hi) in limits.items():
        if code not in levels.index:
            continue
        val = float(levels[code])
        if val < lo:
            rows.append(
                {"code": code, "target_type": 1, "warning_type": 1, "current_val": val}
            )
        elif val > hi:
            rows.append(
                {"code": code, "target_type": 1, "warning_type": 2, "current_val": val}
            )
    return pd.DataFrame(rows)
