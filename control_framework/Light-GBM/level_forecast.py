"""Online recursive LightGBM level forecast (5-min steps, default 1-hour horizon).

Loads local model files from ./model/{StationID}.txt.
Writes predictions to a local CSV when write_output is true.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from ruamel import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from czcommon import TIMEFMT  # noqa: E402
from data_stub import load_level_flow_frame  # noqa: E402

HERE = Path(__file__).resolve().parent
pd.set_option("mode.chained_assignment", None)


def recursive_forecast(
    df: pd.DataFrame,
    station: str,
    related_flows: list[str],
    model: lgb.Booster,
    runtime: datetime,
    n_steps: int,
    lag_steps: int,
) -> pd.DataFrame:
    level_col = f"{station}_level"
    work = df[[level_col] + related_flows].copy()
    for col in related_flows:
        work[col] = work[col].rolling(window="1h", min_periods=1).mean()

    features = pd.DataFrame(index=work.index)
    for col in related_flows:
        for k in range(1, lag_steps + 1):
            lag_name = f"{col}_{k * 5}"
            features[lag_name] = work[col].shift(k)
            if np.isnan(features.loc[work.index[-1], lag_name]):
                features.loc[work.index[-1], lag_name] = work.loc[work.index[-1], col]

    merged = pd.concat([work, features], axis=1)
    feat_names = [c for c in merged.columns if c not in related_flows]

    level = float(merged.loc[merged.index[-1], level_col])
    rows = []
    ctm = datetime.now().strftime(TIMEFMT)
    for i in range(1, n_steps + 1):
        x = []
        for name in feat_names:
            if "level" in name:
                x.append(level)
            else:
                x.append(merged.loc[merged.index[-1], name])
        level = level + round(float(model.predict(np.array(x).reshape(1, -1))[0]), 3)
        rows.append(
            {
                "station": station,
                "tmst": (runtime + timedelta(minutes=(i - 1) * 5)).strftime(TIMEFMT),
                "tmed": (runtime + timedelta(minutes=i * 5)).strftime(TIMEFMT),
                "ubound": level,
                "lbound": level,
                "ctm": ctm,
            }
        )
    return pd.DataFrame(rows)


def model_filename(station: str) -> str:
    return station.replace("-", "_") + ".txt"


def main() -> None:
    import os

    os.chdir(HERE)
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.RoundTripLoader)

    if config.get("runtime") is None:
        runtime = datetime.now().replace(second=0, microsecond=0)
    else:
        y, m, d, h, mi = config["runtime"][:5]
        runtime = datetime(y, m, d, h, mi)

    hist = load_level_flow_frame(
        runtime - timedelta(hours=1),
        runtime,
        csv_path=config.get("offline_csv"),
    )

    n_steps = int(config["horizon"]["n_steps"])
    lag_steps = int(config["horizon"]["lag_steps"])
    outs = []
    for station in config["stations"]:
        model_path = HERE / "model" / model_filename(station)
        if not model_path.exists():
            print(f"[skip] missing model: {model_path.name}")
            continue
        # Load via model string to avoid C-API path encoding issues.
        model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
        related = list(config["data_relation"][f"{station}_level"])
        outs.append(
            recursive_forecast(hist, station, related, model, runtime, n_steps, lag_steps)
        )

    if not outs:
        print("No predictions produced (train models first).")
        return
    result = pd.concat(outs, ignore_index=True)
    if config.get("write_output"):
        out_csv = str(config.get("output_csv") or "predictions.csv")
        result.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")
    else:
        print(result.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
