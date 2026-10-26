# LOB Alpha Research — Experiment Summary

Data mode: **SYNTHETIC_DEMO_ONLY**; model: **lightgbm**; feature set: **all**.

**Important:** Outputs from synthetic data demonstrate software behavior, NOT market predictability.
A single trading session does not provide independent day-level out-of-sample evidence.
The execution simulation assumes immediate full fills at displayed quotes and excludes market impact.

| Horizon (events) | Test accuracy | Test macro-F1 | Test log loss | Spearman IC | Trades | Net P&L (USD) |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.471 | 0.299 | 1.156 | -0.036 | 22 | -11.02 |

## Audit

See `results.json` for snapshot validation and per-horizon confusion matrices; the trade log contains event indices and actual entry/exit quote prices.
