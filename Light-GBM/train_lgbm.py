"""Offline LightGBM training for short-term level-difference models.

Each station predicts level_diff over a fixed horizon (default: 60 min for
one-shot, or recursive 5-min steps for online use). Feature template:
  - current level
  - related pump/WWTP flows and their 5-min lags (past ~30 min)

No database access. Provide a local CSV via config['offline_csv'] or rely on
the synthetic demo generator in common/data_stub.py.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from ruamel import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from data_stub import load_level_flow_frame  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"


def build_features(df: pd.DataFrame, level_col: str, flow_cols: list[str], lag_steps: int) -> tuple[pd.DataFrame, list[str]]:
    """Construct lagged flow features; drop raw instantaneous flow columns from the feature set."""
    work = df[[level_col] + flow_cols].copy()
    features = pd.DataFrame(index=work.index)
    for col in flow_cols:
        for k in range(1, lag_steps + 1):
            features[f"{col}_{k * 5}"] = work[col].shift(k)
    merged = pd.concat([work, features], axis=1).dropna()
    feat_names = [c for c in merged.columns if c not in flow_cols]
    return merged, feat_names


def add_diff_label(df: pd.DataFrame, level_col: str, horizon_steps: int) -> pd.DataFrame:
    """Label = level(t + horizon) - level(t)."""
    out = df.copy()
    out[f"{level_col}_diff"] = np.round(out[level_col].shift(-horizon_steps) - out[level_col], 3)
    return out.dropna()


def train_one_station(
    df: pd.DataFrame,
    station: str,
    related_flows: list[str],
    params: dict,
    lag_steps: int = 6,
    horizon_steps: int = 12,
    train_ratio: float = 0.7,
) -> lgb.Booster:
    level_col = f"{station}_level"
    merged, feat_names = build_features(df, level_col, related_flows, lag_steps)
    labeled = add_diff_label(merged, level_col, horizon_steps)
    label = f"{level_col}_diff"

    n = len(labeled)
    split = int(n * train_ratio)
    train_df = labeled.iloc[:split]
    val_df = labeled.iloc[split:]

    dtrain = lgb.Dataset(train_df[feat_names], train_df[label])
    dval = lgb.Dataset(val_df[feat_names], val_df[label], reference=dtrain)

    booster = lgb.train(
        {
            "learning_rate": params.get("learning_rate", 0.05),
            "bagging_fraction": params.get("bagging_fraction", 0.9),
            "feature_fraction": params.get("feature_fraction", 0.8),
            "num_leaves": params.get("num_leaves", 64),
            "bagging_freq": params.get("bagging_freq", 4),
            "boosting_type": "gbdt",
            "objective": "mse",
            "metric": "mse",
            "verbose": -1,
            "seed": 2222,
            "feature_pre_filter": False,
        },
        dtrain,
        num_boost_round=int(params.get("n_estimators", 200)),
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(int(params.get("early_stopping_rounds", 20))),
            lgb.log_evaluation(0),
        ],
    )
    return booster


def model_filename(station: str) -> str:
    """Filesystem-safe model name (hyphens -> underscores)."""
    return station.replace("-", "_") + ".txt"


def main() -> None:
    # Work in this folder so LightGBM writes relative ASCII paths only.
    os.chdir(HERE)
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.RoundTripLoader)

    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(days=30)
    df = load_level_flow_frame(start, end, csv_path=config.get("offline_csv"))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lag = int(config["horizon"]["lag_steps"])
    # Train one-shot 60-min difference by default (12 x 5-min steps)
    horizon = int(config["horizon"]["n_steps"])

    for station in config["stations"]:
        level_col = f"{station}_level"
        related = list(config["data_relation"][level_col])
        missing = [c for c in [level_col] + related if c not in df.columns]
        if missing:
            print(f"[skip] {station}: missing columns {missing}")
            continue
        model = train_one_station(
            df, station, related, dict(config["lgbm"]), lag_steps=lag, horizon_steps=horizon
        )
        # Avoid LightGBM C-API write on non-ASCII absolute paths: dump via Python I/O.
        out_path = MODEL_DIR / model_filename(station)
        out_path.write_text(model.model_to_string(), encoding="utf-8")
        print(f"[ok] saved {out_path.name}")


if __name__ == "__main__":
    main()
