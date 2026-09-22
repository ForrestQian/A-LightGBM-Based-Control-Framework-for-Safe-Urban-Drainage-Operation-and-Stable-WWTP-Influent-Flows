"""Abstract data-source stubs.

Replace these callables with your own offline CSV loaders or secured clients.
This package intentionally ships with NO server addresses, credentials, or raw data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from naming import CONTROLLED_PUMPS, WWTPS


def load_level_flow_frame(
    start: datetime,
    end: datetime,
    freq: str = "5min",
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a datetime-indexed frame with columns like:
      Pump-XL_level, Pump-XL_flow, WWTP-1_level, WWTP-1_flow, ...

    If csv_path is provided, read it; otherwise return a synthetic demo frame
    (random walk) so algorithm pipelines can be exercised offline.
    """
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["tm"])
        df = df.set_index("tm").sort_index()
        return df.loc[(df.index >= start) & (df.index <= end)].copy()

    idx = pd.date_range(start, end, freq=freq)
    rng = np.random.default_rng(0)
    data = {}
    for code in CONTROLLED_PUMPS + WWTPS:
        data[f"{code}_level"] = -1.0 + rng.standard_normal(len(idx)).cumsum() * 0.01
        data[f"{code}_flow"] = np.clip(2000 + rng.standard_normal(len(idx)) * 100, 0, None)
    # Extra feeder flow columns used by LightGBM feature templates
    for feeder in ["Pump-U1", "Pump-U2", "Pump-U3", "Pump-U4", "Pump-U5", "Pump-U6"]:
        data[f"{feeder}_flow"] = np.clip(500 + rng.standard_normal(len(idx)) * 50, 0, None)
    return pd.DataFrame(data, index=idx)


def load_section_volume_curve(section_id: str, csv_path: Optional[str] = None) -> pd.DataFrame:
    """Return a level-volume lookup table with columns [level, volume]."""
    if csv_path:
        return pd.read_csv(csv_path)
    levels = np.linspace(-3.0, 2.0, 51)
    volumes = (levels - levels.min()) / (levels.max() - levels.min()) * 5000.0
    return pd.DataFrame({"level": levels, "volume": volumes})
