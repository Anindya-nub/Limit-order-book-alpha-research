"""Causal market microstructure features (only use events at or before t)."""
from __future__ import annotations
import numpy as np
import pandas as pd

BASE_FEATURES = [
    "spread_bps", "obi_1", "obi_5", "microprice_deviation_bps",
    "ofi_normalized", "signed_trade_pressure_10", "signed_trade_pressure_50",
    "signed_cancel_pressure_50", "trade_intensity_50", "cancel_intensity_50",
    "depth_ratio_5", "log_depth_5", "mid_return_10_bps", "mid_return_50_bps",
]


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return event-aligned features; assumes rows sorted within each `day`.

    Uses visible executed order direction reversed to infer aggressor side:
    sell resting order (-1) executed => buyer-initiated trade (+).
    """
    if "day" not in df:
        df = df.assign(day="single_day")
    pieces = []
    for _, x in df.groupby("day", sort=False):
        x = x.copy().reset_index(drop=True)
        a, b = x["ask_price_1"], x["bid_price_1"]
        az, bz = x["ask_size_1"].astype(float), x["bid_size_1"].astype(float)
        x["mid"] = (a + b) / 2
        x["spread"] = a - b
        x["spread_bps"] = 1e4 * _safe_div(x["spread"], x["mid"])
        x["obi_1"] = _safe_div(bz - az, bz + az)
        x["microprice"] = _safe_div(a * bz + b * az, bz + az)
        x["microprice_deviation_bps"] = 1e4 * _safe_div(x["microprice"] - x["mid"], x["mid"])
        levels = min(5, sum(c.startswith("ask_size_") for c in x.columns))
        bid_depth = sum(x[f"bid_size_{i}"].astype(float) for i in range(1, levels + 1))
        ask_depth = sum(x[f"ask_size_{i}"].astype(float) for i in range(1, levels + 1))
        x["obi_5"] = _safe_div(bid_depth - ask_depth, bid_depth + ask_depth)
        x["depth_ratio_5"] = np.log((bid_depth + 1) / (ask_depth + 1))
        x["log_depth_5"] = np.log1p(bid_depth + ask_depth)
        prev_a, prev_b = a.shift(1), b.shift(1)
        prev_az, prev_bz = az.shift(1), bz.shift(1)
        e_bid = (b >= prev_b).astype(int) * bz - (b <= prev_b).astype(int) * prev_bz
        e_ask = (a <= prev_a).astype(int) * az - (a >= prev_a).astype(int) * prev_az
        x["ofi"] = (e_bid - e_ask).fillna(0)
        prior_depth = (az + bz).rolling(50, min_periods=1).mean().shift(1)
        x["ofi_normalized"] = _safe_div(x["ofi"].rolling(10, min_periods=1).sum(), prior_depth)
        event = x["event_type"]
        x["trade_signed_size"] = np.where(event.isin([4, 5]), -x["direction"] * x["size"], 0)
        x["cancel_signed_size"] = np.where(event.isin([2, 3]), -x["direction"] * x["size"], 0)
        for window in [10, 50]:
            x[f"signed_trade_pressure_{window}"] = _safe_div(
                x["trade_signed_size"].rolling(window, min_periods=1).sum(),
                x["size"].where(event.isin([4, 5]), 0).rolling(window, min_periods=1).sum() + 1,
            )
        x["signed_cancel_pressure_50"] = _safe_div(
            x["cancel_signed_size"].rolling(50, min_periods=1).sum(),
            x["size"].where(event.isin([2, 3]), 0).rolling(50, min_periods=1).sum() + 1,
        )
        x["trade_intensity_50"] = event.isin([4, 5]).astype(float).rolling(50, min_periods=1).mean()
        x["cancel_intensity_50"] = event.isin([2, 3]).astype(float).rolling(50, min_periods=1).mean()
        for lag in [10, 50]:
            x[f"mid_return_{lag}_bps"] = 1e4 * np.log(x["mid"] / x["mid"].shift(lag))
        x["valid_book"] = (
            (x["bid_size_1"] > 0) & (x["ask_size_1"] > 0) &
            (x["bid_price_1"] < x["ask_price_1"]) &
            x["spread_bps"].between(0, 500)
        )
        pieces.append(x)
    out = pd.concat(pieces, ignore_index=True)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

FEATURE_SETS = {
    "all": BASE_FEATURES,
    "obi": ["obi_1", "obi_5", "spread_bps", "depth_ratio_5", "log_depth_5"],
    "ofi": ["ofi_normalized", "spread_bps", "depth_ratio_5", "log_depth_5"],
    "trade_flow": ["signed_trade_pressure_10", "signed_trade_pressure_50", "trade_intensity_50",
                   "spread_bps", "depth_ratio_5", "log_depth_5"],
}
