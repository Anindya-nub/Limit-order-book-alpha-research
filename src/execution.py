"""Conservative-ish top-of-book *aggressive* execution research baseline.

NOT a production backtester: assumes immediate full fill for sufficiently small
marketable orders at historical displayed bid/ask; lacks impact and venue routing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def aggressive_backtest(df: pd.DataFrame, indices: np.ndarray, probs: np.ndarray,
                        horizon: int, latency_events: int = 0,
                        shares: int = 10, fee_per_share: float = 0.003,
                        min_confidence: float = 0.40, margin: float = 0.07) -> tuple[pd.DataFrame, dict]:
    """Non-overlapping one-position trades, strictly executed *after* signal event.

    decision t => aggressive entry at t+1+latency_events => exit at entry+h.
    Both entry and exit cross the spread; inventory flat after exit.
    All fill prices are post-event quotes, not future midprices.
    """
    if len(indices) != len(probs):
        raise ValueError("indices/probabilities length mismatch")
    if latency_events < 0 or shares <= 0:
        raise ValueError("latency must be >=0 and shares must be >0")
    rows = []
    occupied_until = -1
    day = df["day"].to_numpy()
    bids = df["bid_price_1"].to_numpy()
    asks = df["ask_price_1"].to_numpy()
    bid_sizes = df["bid_size_1"].to_numpy()
    ask_sizes = df["ask_size_1"].to_numpy()
    times = df["time"].to_numpy()
    idx_sorted = np.argsort(indices)
    for z in idx_sorted:
        signal_idx = int(indices[z])
        if signal_idx <= occupied_until:
            continue
        p_down, p_flat, p_up = probs[z]
        if max(p_up, p_down) < min_confidence or abs(p_up - p_down) < margin:
            continue
        direction = 1 if p_up > p_down else -1
        entry = signal_idx + 1 + latency_events
        exit_ = entry + horizon
        if exit_ >= len(df) or day[signal_idx] != day[exit_]:
            continue
        if any((not np.isfinite(bids[i])) or (not np.isfinite(asks[i])) or
               bids[i] >= asks[i] for i in (entry, exit_)):
            continue
        # Conservative liquidity screen at entry and exit. (No impact/queue model.)
        if min(bid_sizes[entry], ask_sizes[entry], bid_sizes[exit_], ask_sizes[exit_]) < shares:
            continue
        # Buy at ask and sell at bid; vice versa for short.
        buy_price = asks[entry] if direction == 1 else asks[exit_]
        sell_price = bids[exit_] if direction == 1 else bids[entry]
        gross = shares * (sell_price - buy_price)
        fees = 2 * shares * fee_per_share
        pnl = gross - fees
        entry_mid = (asks[entry] + bids[entry]) / 2
        exit_mid = (asks[exit_] + bids[exit_]) / 2
        markout_bps = direction * 1e4 * (exit_mid / entry_mid - 1)
        rows.append({"signal_index": signal_idx, "entry_index": entry, "exit_index": exit_,
                     "entry_time": float(times[entry]), "exit_time": float(times[exit_]),
                     "side": "long" if direction == 1 else "short", "shares": shares,
                     "entry_price": float(asks[entry] if direction == 1 else bids[entry]),
                     "exit_price": float(bids[exit_] if direction == 1 else asks[exit_]),
                     "gross_pnl_usd": round(float(gross), 6), "fees_usd": round(float(fees), 6),
                     "net_pnl_usd": round(float(pnl), 6), "signed_mid_markout_bps": float(markout_bps),
                     "predicted_direction_probability": float(max(p_up, p_down))})
        occupied_until = exit_
    trades = pd.DataFrame(rows)
    if len(trades) == 0:
        return trades, {"trades": 0, "net_pnl_usd": 0, "warning": "No orders met thresholds or executable liquidity screens"}
    pnl = trades["net_pnl_usd"].to_numpy()
    equity = np.r_[0., np.cumsum(pnl)]
    dd = equity - np.maximum.accumulate(equity)
    return trades, {
        "trades": int(len(trades)), "net_pnl_usd": round(float(pnl.sum()), 4),
        "gross_pnl_usd": round(float(trades["gross_pnl_usd"].sum()), 4),
        "paid_fees_usd": round(float(trades["fees_usd"].sum()), 4),
        "win_rate": float((pnl > 0).mean()),
        "average_net_pnl_usd": float(pnl.mean()),
        "max_drawdown_usd": round(float(-dd.min()), 4),
        "mean_signed_mid_markout_bps": float(trades["signed_mid_markout_bps"].mean()),
        "execution_assumption": "full fill at displayed top-of-book, no impact; not live-executable evidence",
    }
