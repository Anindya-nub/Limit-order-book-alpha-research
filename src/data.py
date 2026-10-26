"""Official LOBSTER paired CSV loader and integrity checks."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

MESSAGE_COLUMNS = ["time", "event_type", "order_id", "size", "price", "direction"]
PRICE_SCALE = 10_000.0


def read_lobster(message_path: str | Path, orderbook_path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load row-aligned LOBSTER messages + snapshots (prices converted to USD).

    Row t is the book AFTER message t. Do not shift snapshots by one row.
    """
    msg = pd.read_csv(message_path, header=None, names=MESSAGE_COLUMNS, nrows=max_rows)
    book = pd.read_csv(orderbook_path, header=None, nrows=max_rows)
    if len(msg) != len(book):
        raise ValueError(f"Misaligned files: {len(msg)} messages, {len(book)} snapshots")
    if book.shape[1] % 4:
        raise ValueError("Orderbook CSV must have 4 columns per price level")
    levels = book.shape[1] // 4
    book.columns = [f"{kind}_{val}_{depth}" for depth in range(1, levels + 1)
                    for kind, val in [("ask", "price"), ("ask", "size"), ("bid", "price"), ("bid", "size")]]
    msg = msg.reset_index(drop=True)
    book = book.reset_index(drop=True)
    for col in [c for c in book if "_price_" in c]:
        book[col] = book[col].astype(float) / PRICE_SCALE
        side_size = col.replace("_price_", "_size_")
        book.loc[book[side_size].eq(0) | book[col].abs().gt(1_000_000), col] = np.nan
    msg["price"] = msg["price"].astype(float) / PRICE_SCALE
    out = pd.concat([msg, book], axis=1)
    out["event_index"] = np.arange(len(out))
    return out


def inspect_snapshots(df: pd.DataFrame) -> dict:
    """Basic integrity diagnostics; never silently repair invalid book states."""
    valid = (df["ask_size_1"] > 0) & (df["bid_size_1"] > 0)
    valid &= df["ask_price_1"].notna() & df["bid_price_1"].notna()
    crossed = valid & (df["bid_price_1"] >= df["ask_price_1"])
    monotonic = np.zeros(len(df), dtype=bool)
    n = sum(c.startswith("ask_price_") for c in df.columns)
    for depth in range(1, n):
        a0, a1 = df[f"ask_price_{depth}"], df[f"ask_price_{depth+1}"]
        b0, b1 = df[f"bid_price_{depth}"], df[f"bid_price_{depth+1}"]
        monotonic |= (a0.notna() & a1.notna() & a0.ge(a1)).to_numpy()
        monotonic |= (b0.notna() & b1.notna() & b0.le(b1)).to_numpy()
    return {
        "rows": int(len(df)),
        "levels": int(n),
        "missing_top_level_rows": int((~valid).sum()),
        "locked_or_crossed_rows": int(crossed.sum()),
        "non_monotonic_depth_rows": int(monotonic.sum()),
        "unexpected_event_type_rows": int((~df["event_type"].isin([1, 2, 3, 4, 5, 6, 7])).sum()),
        "time_reversal_rows": int((df["time"].diff() < 0).sum()),
    }


def check_visible_event_deltas(df: pd.DataFrame) -> dict:
    """Check visible volume transitions without claiming complete L3 reconstruction.

    A check is eligible only when the event price is visible in BOTH successive
    snapshots. This permits a useful audit even with LOBSTER's truncated depth.
    """
    n = len(df)
    if n < 2:
        return {"eligible_events": 0, "matched_events": 0, "mismatched_events": 0, "first_mismatch_rows": []}
    px = df["price"].to_numpy()
    event = df["event_type"].to_numpy()
    direction = df["direction"].to_numpy()
    before = np.zeros(n, dtype=np.int64)
    after = np.zeros(n, dtype=np.int64)
    seen_before = np.zeros(n, dtype=bool)
    seen_after = np.zeros(n, dtype=bool)
    nlevels = sum(c.startswith("ask_price_") for c in df.columns)
    for side, code in [("bid", 1), ("ask", -1)]:
        target_side = direction == code
        for level in range(1, nlevels + 1):
            prices = df[f"{side}_price_{level}"].to_numpy()
            sizes = df[f"{side}_size_{level}"].to_numpy(dtype=np.int64)
            current = target_side & (prices == px)
            previous = np.roll(prices, 1)
            previous_sizes = np.roll(sizes, 1)
            prev_match = target_side & (previous == px)
            seen_after |= current
            seen_before |= prev_match
            after += np.where(current, sizes, 0)
            before += np.where(prev_match, previous_sizes, 0)
    expected = np.where(event == 1, df["size"].to_numpy(), -df["size"].to_numpy())
    eligible = np.isin(event, [1, 2, 3, 4]) & seen_before & seen_after
    eligible[0] = False
    mismatches = np.flatnonzero(eligible & ((after - before) != expected))
    return {
        "eligible_events": int(eligible.sum()),
        "matched_events": int(eligible.sum() - len(mismatches)),
        "mismatched_events": int(len(mismatches)),
        "first_mismatch_rows": [int(x) for x in mismatches[:5]],
        "note": "Only events at prices visible in both adjacent snapshots were eligible; this does not prove full order-ID history.",
    }
