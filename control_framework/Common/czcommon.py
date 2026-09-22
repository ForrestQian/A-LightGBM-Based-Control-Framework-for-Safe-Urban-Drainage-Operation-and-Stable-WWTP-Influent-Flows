"""Shared time helpers and lightweight DataFrame I/O utilities.

No hostnames, credentials, or site-specific endpoints are stored here.
Any remote access must be injected by the caller and is out of scope for
this anonymized package.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from string import Template
from typing import Any, Mapping, Optional

import pandas as pd

TIMEFMT = "%Y-%m-%d %H:%M:%S"


def datetime_minute(tm: datetime) -> datetime:
    return tm.replace(second=0, microsecond=0)


def datetime_hour(tm: datetime) -> datetime:
    return tm.replace(minute=0, second=0, microsecond=0)


def datetime_day(tm: datetime, start_hour: int = 0) -> datetime:
    return tm.replace(hour=start_hour, minute=0, second=0, microsecond=0)


def env_or_none(key: str) -> Optional[str]:
    """Read an optional environment variable; return None if unset/empty."""
    val = os.environ.get(key)
    return val if val else None


def query_to_df(client: Any, sql: str, parameters: Mapping[str, Any]) -> pd.DataFrame:
    """
    Execute a parameterized SQL string on a client that exposes
    ``execute(..., columnar=True, with_column_types=True)``.
    """
    sql_to_run = Template(sql).substitute(**parameters)
    data, columns = client.execute(sql_to_run, columnar=True, with_column_types=True)
    return pd.DataFrame({re.sub(r"\W", "_", col[0]): d for d, col in zip(data, columns)})


def df_to_sql_table(db: Any, df_in: pd.DataFrame, table_name: str, method: str = "REPLACE") -> None:
    """Insert rows into a SQL table through a DB-API connection (generic)."""
    cursor = db.cursor()
    cols = ",".join(df_in.columns)
    placeholders = "}', '${".join(df_in.columns)
    for _, row in df_in.iterrows():
        sql = (
            f"{method} INTO {table_name} ({cols}) "
            f"VALUES ('${{{placeholders}}}')"
        )
        try:
            cursor.execute(Template(sql).substitute(**row.to_dict()))
            db.commit()
        except Exception:
            db.rollback()


def repair_spike(series: pd.Series, max_diff: float) -> tuple[pd.Series, list[int]]:
    """Replace isolated near-zero spikes when neighbors are larger than max_diff."""
    adjusted = series.copy()
    changed: list[int] = []
    for i in range(1, len(series) - 1):
        if (
            abs(series.iloc[i]) < 0.1
            and series.iloc[i + 1] > max_diff
            and series.iloc[i - 1] > max_diff
        ):
            adjusted.iloc[i] = (series.iloc[i - 1] + series.iloc[i + 1]) / 2.0
            changed.append(i)
    return adjusted, changed
