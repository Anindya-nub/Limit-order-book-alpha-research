"""Repeatable experiment: data -> causal features -> labels -> split -> ML -> backtest."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import read_lobster, inspect_snapshots, check_visible_event_deltas
from .features import make_features, FEATURE_SETS
from .labels import add_labels, temporal_split
from .models import build_estimator, probabilities_3, evaluate
from .execution import aggressive_backtest
from .visuals import save_charts, save_summary


def _json_safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    raise TypeError(type(v).__name__)


def run_experiment(message_paths: list[str], book_paths: list[str], output_dir: str = "reports",
                   horizons: tuple[int, ...] = (10, 50, 100), model: str = "logistic", feature_set: str = "all",
                   latency_events: int = 0, shares: int = 10,
                   fee_per_share: float = 0.003, max_rows: int | None = None,
                   tick_size: float = 0.01, flat_threshold_ticks: float = 0.5) -> dict:
    if len(message_paths) != len(book_paths) or not message_paths:
        raise ValueError("Provide one orderbook path for each message path")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    books = []
    checks = []
    for i, (mp, bp) in enumerate(zip(message_paths, book_paths)):
        df = read_lobster(mp, bp, max_rows)
        df["day"] = f"day_{i:03d}"  # each message/book pair is a separate session
        checks.append({"file": str(mp), **inspect_snapshots(df), "visible_event_delta_check": check_visible_event_deltas(df)})
        books.append(df)
    data = add_labels(make_features(pd.concat(books, ignore_index=True)), horizons,
                      tick_size=tick_size, flat_threshold_ticks=flat_threshold_ticks)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = {
        "mode": "HISTORICAL" if not any("synthetic" in p.lower() for p in message_paths) else "SYNTHETIC_DEMO_ONLY",
        "model": model, "feature_set": feature_set, "snapshot_checks": checks, "days": len(books),
        "single_day_warning": len(books) < 3,
        "important_caveat": "A one-day chronological split is NOT independent day-level out-of-sample evidence",
        "run_config": {"horizons": horizons, "latency_events": latency_events,
                       "shares": shares, "fee_per_share": fee_per_share,
                       "tick_size": tick_size, "flat_threshold_ticks": flat_threshold_ticks},
        "experiments": {},
    }
    for h in horizons:
        split = temporal_split(data, h)
        train, val, test = (split[k] for k in ("train", "validation", "test"))
        X = data[FEATURE_SETS[feature_set]].replace([np.inf, -np.inf], np.nan)
        y = data[f"label_{h}"].astype("Int64")
        train_labels = y.iloc[train].to_numpy(dtype=int)
        if len(np.unique(train_labels)) < 2:
            raise ValueError(f"h={h}: training partition has <2 classes; add better/full-day data")
        train_majority_label = int(pd.Series(train_labels).value_counts().index[0])
        estimator = build_estimator(model)
        estimator.fit(X.iloc[train], train_labels)
        probs_val = probabilities_3(estimator, X.iloc[val])
        probs_test = probabilities_3(estimator, X.iloc[test])
        ev_val = evaluate(y.iloc[val].to_numpy(dtype=int), probs_val)
        ev_test = evaluate(y.iloc[test].to_numpy(dtype=int), probs_test)
        ev_val["train_majority_baseline_accuracy"] = float((y.iloc[val].to_numpy(dtype=int) == train_majority_label).mean())
        ev_test["train_majority_baseline_accuracy"] = float((y.iloc[test].to_numpy(dtype=int) == train_majority_label).mean())
        trades, pnl_stats = aggressive_backtest(
            data, test, probs_test, horizon=h, latency_events=latency_events,
            shares=shares, fee_per_share=fee_per_share)
        trades.to_csv(outdir / f"trades_h{h}_{model}.csv", index=False)
        save_charts(outdir, f"h{h}_{model}_{feature_set}", y.iloc[test].to_numpy(dtype=int), probs_test, trades)
        # Calculate rank IC against realised raw forward price move, no in-sample fitting.
        forecast = probs_test[:, 2] - probs_test[:, 0]
        realised = data.iloc[test][f"future_mid_{h}"].to_numpy() - data.iloc[test]["mid"].to_numpy()
        ic = float(pd.Series(forecast).corr(pd.Series(realised), method="spearman"))
        results["experiments"][str(h)] = {
            "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
            "validation": ev_val, "test": ev_test, "test_spearman_ic": ic if np.isfinite(ic) else None,
            "aggressive_execution": pnl_stats,
        }
        print(f"h={h:3d} | test accuracy={ev_test['accuracy']:.3f} | "
              f"macro-F1={ev_test['macro_f1']:.3f} | test trades={pnl_stats['trades']} | "
              f"net P&L=${pnl_stats['net_pnl_usd']:.2f}")
    (outdir / "results.json").write_text(json.dumps(results, indent=2, default=_json_safe))
    save_summary(outdir, results)
    return results
