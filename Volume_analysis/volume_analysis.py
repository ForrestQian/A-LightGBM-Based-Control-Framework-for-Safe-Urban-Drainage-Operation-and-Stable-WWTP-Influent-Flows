"""Level-to-volume conversion and section storage estimation.

Converts observed levels to tank/pipe volumes via a monotonic level-volume
curve, then reports utilization against configured capacity.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from ruamel import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from data_stub import load_level_flow_frame, load_section_volume_curve  # noqa: E402

HERE = Path(__file__).resolve().parent


def level_to_volume(level: float, curve: pd.DataFrame) -> float:
    """Nearest-neighbor lookup on a level-volume curve."""
    idx = (curve["level"] - level).abs().idxmin()
    return float(curve.loc[idx, "volume"])


def estimate_section_usage(
    levels: pd.Series,
    section_id: str,
    capacity: float,
    curve_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Map the latest level (or a level series) to volume and utilization.

    Parameters
    ----------
    levels : Series
        Level samples (index = time). The last non-null value is used.
    section_id : str
        Anonymized section ID, e.g. HJT-XL.
    capacity : float
        Design volume of the section (m3).
    """
    curve = load_section_volume_curve(section_id, csv_path=curve_csv)
    val = float(levels.dropna().iloc[-1])
    vol = level_to_volume(val, curve)
    util = vol / capacity if capacity > 0 else float("nan")
    return pd.DataFrame(
        [
            {
                "section": section_id,
                "level": val,
                "volume": vol,
                "capacity": capacity,
                "utilization": util,
                "tm": levels.dropna().index[-1],
            }
        ]
    )


def main() -> None:
    with open(HERE / "config.yml", "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.RoundTripLoader)

    runtime = datetime.now().replace(second=0, microsecond=0)
    # Demo: pull synthetic levels for stations; map to linked sections
    hist = load_level_flow_frame(runtime - timedelta(hours=1), runtime)
    rows = []
    for section, meta in config["sections"].items():
        station = meta["level_station"]
        col = f"{station}_level"
        if col not in hist.columns:
            continue
        rows.append(
            estimate_section_usage(
                hist[col],
                section,
                capacity=float(meta["capacity"]),
            )
        )
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if config.get("write_output"):
        out = HERE / str(config.get("output_csv") or "section_volume.csv")
        result.to_csv(out, index=False)
        print(f"Wrote {out}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
