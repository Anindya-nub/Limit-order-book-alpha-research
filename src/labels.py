"""Three-class event-horizon future mid labels and purged temporal split."""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_labels(df: pd.DataFrame, horizons: tuple[int, ...] = (10, 50, 100), flat_threshold_ticks: float = 0.5, tick_size: float = 0.01) -> pd.DataFrame:
    """-1 down / 0 unchanged / +1 up. No label crosses a day boundary."""
    out = df.copy()
    for h in horizons:
        future = out.groupby("day", sort=False)["mid"].shift(-h)
        movement = future - out["mid"]
        threshold = flat_threshold_ticks * tick_size
        label = np.select([movement > threshold, movement < -threshold], [1, -1], default=0).astype(float)
        label[future.isna() | out["mid"].isna()] = np.nan
        out[f"label_{h}"] = label
        out[f"future_mid_{h}"] = future
    return out


def temporal_split(df: pd.DataFrame, horizon: int, proportions=(0.6, 0.2, 0.2)) -> dict[str, np.ndarray]:
    """Split by trading days when >=3, else split within the day.

    Purge rows whose t+h label would reach a new split or new trading day.
    The test set is held out entirely for final reporting.
    """
    days = df["day"].drop_duplicates().tolist()
    n = len(df)
    if n < max(150, 5 * horizon):
        raise ValueError("Insufficient events for train/validation/test split")
    if len(days) >= 3:
        a = max(1, int(len(days) * proportions[0]))
        b = min(len(days) - 1, max(a + 1, int(len(days) * (proportions[0] + proportions[1]))))
        if b >= len(days):
            b = len(days) - 1
        segment = np.where(df["day"].isin(days[:a]), 0, np.where(df["day"].isin(days[a:b]), 1, 2))
    else:
        # Diagnostic only: test shares a calendar day with training observations.
        segment = np.zeros(n, dtype=int)
        segment[int(n * proportions[0]):int(n * (proportions[0] + proportions[1]))] = 1
        segment[int(n * (proportions[0] + proportions[1])):] = 2
    future_idx = np.arange(n) + horizon
    future_idx_safe = np.minimum(future_idx, n - 1)
    same_segment = (future_idx < n) & (segment == segment[future_idx_safe])
    same_day = (df["day"].to_numpy() == df["day"].to_numpy()[future_idx_safe])
    book_valid = df["valid_book"].to_numpy()
    valid = (same_segment & same_day & df[f"label_{horizon}"].notna().to_numpy()
             & book_valid & book_valid[future_idx_safe])
    # Drop first 50 events per day; momentum features otherwise contain insufficient history.
    enough_history = df.groupby("day", sort=False).cumcount().to_numpy() >= 50
    valid &= enough_history
    result = {name: np.flatnonzero(valid & (segment == i)) for i, name in enumerate(["train", "validation", "test"])}
    if any(len(x) < 25 for x in result.values()):
        raise ValueError("Too few valid rows in at least one partition")
    return result
